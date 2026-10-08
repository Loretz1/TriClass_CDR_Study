#!/usr/bin/env python3
"""Build and validate static grounded confidence for pseudo interactions.

This module is intentionally independent from the CDR backbones. It reuses the
immutable target-pattern candidates, trains a source-only LightGCN, grounds
each candidate with source-similar overlapped users' target-domain behavior,
and learns a two-evidence low-capacity reliability calibrator.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.target_pattern_pseudo import (
    atomic_json,
    atomic_torch_save,
    file_sha256,
    interaction_sets,
    load_split,
    mapping_fingerprint,
    set_seed,
    split_lightgcn_edges,
    train_lightgcn_once,
)


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGGER = logging.getLogger("grounded-static-confidence")
FORMAT_VERSION = 1
PHASES = ("source-encoder", "evidence", "calibrate")


def deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path, dataset, domains, device_override="auto"):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    profiles = config.pop("domain_profiles", {}) or {}
    domain_key = "+".join(domains)
    if domain_key not in profiles:
        supported = ", ".join(sorted(profiles))
        raise ValueError(
            "No grounded-confidence profile for {}. Supported: {}".format(
                domain_key, supported
            )
        )
    deep_update(config, profiles[domain_key])
    split = config.pop("split", None)
    if not isinstance(split, dict):
        raise ValueError("The YAML must define a split mapping.")
    config.update(split)
    config["dataset"] = dataset
    config["domains"] = list(domains)
    data_path = config["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(SRC_ROOT, data_path))
    config["data_path"] = data_path
    requested_device = device_override
    if requested_device == "auto":
        requested_device = config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    config["device"] = str(requested_device)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build c_sem + c_ov confidence and validate it on simulated-cold "
            "overlapped users."
        )
    )
    parser.add_argument(
        "--dataset", required=True, choices=("Amazon2014", "Douban")
    )
    parser.add_argument("--domains", nargs=2, required=True, metavar=("SRC", "TGT"))
    parser.add_argument(
        "--config",
        default=os.path.join(
            SRC_ROOT, "configs", "model", "GroundedStaticConfidence.yaml"
        ),
    )
    parser.add_argument(
        "--phase", choices=("all",) + PHASES, default="all"
    )
    parser.add_argument("--force-phase", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    LOGGER.setLevel(logging.INFO)
    file_path = os.path.join(output_dir, "build.log")
    if not any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(handler.baseFilename) == os.path.abspath(file_path)
        for handler in root.handlers
    ):
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def atomic_npy(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".npy", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".npz", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def source_graph_fingerprint(id_mapping, edges):
    identity = {
        "src_id2user": id_mapping["src"]["id2user"],
        "src_id2item": id_mapping["src"]["id2item"],
        "edges_sha256": array_sha256(edges),
    }
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rankings(path, expected_users, num_items, width):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cache = np.load(path, allow_pickle=False)
    users = np.asarray(cache["user_ids"], dtype=np.int64)
    items = np.asarray(cache["item_ids"], dtype=np.int64)
    scores = np.asarray(cache["scores"], dtype=np.float32)
    if items.ndim != 2 or items.shape != scores.shape:
        raise ValueError("Ranking item_ids and scores must be equally shaped matrices.")
    if not np.array_equal(users, expected_users):
        raise ValueError("Ranking users are not aligned with the expected source IDs.")
    width = min(int(width), items.shape[1])
    if width <= 0:
        raise ValueError("candidate_width must be positive.")
    items = items[:, :width]
    scores = scores[:, :width]
    if np.any(items <= 0) or np.any(items > num_items):
        raise ValueError("Ranking cache contains invalid target item IDs.")
    return users, items, scores


def load_prerequisites(config, split_dir, all_users, id_mapping):
    source_dir = os.path.join(split_dir, config["source_subdir"])
    manifest_path = os.path.join(source_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Target-pattern manifest is missing: {}".format(manifest_path)
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    num_overlap = len(all_users["overlap_users"])
    num_users_src = len(id_mapping["src"]["id2user"]) - 1
    num_items_tgt = len(id_mapping["tgt"]["id2item"]) - 1
    expected = {
        "dataset": config["dataset"],
        "domains": list(config["domains"]),
        "num_users_overlap": num_overlap,
        "num_users_src": num_users_src,
        "num_items_tgt": num_items_tgt,
        "mapping_sha256": mapping_fingerprint(id_mapping),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError("Target-pattern manifest mismatch: {}".format(mismatches))

    width = int(config["evidence"]["candidate_width"])
    overlap_users = np.arange(1, num_overlap + 1, dtype=np.int64)
    cold_users = np.arange(num_overlap + 1, num_users_src + 1, dtype=np.int64)
    overlap = load_rankings(
        os.path.join(source_dir, "overlap_pseudo_rankings.npz"),
        overlap_users,
        num_items_tgt,
        width,
    )
    cold = load_rankings(
        os.path.join(source_dir, "pseudo_rankings.npz"),
        cold_users,
        num_items_tgt,
        width,
    )
    target_embedding_path = os.path.join(
        source_dir, "target_lightgcn_item_embeddings.npy"
    )
    if not os.path.isfile(target_embedding_path):
        raise FileNotFoundError(target_embedding_path)
    target_items = np.load(target_embedding_path, mmap_mode="r")
    if target_items.shape[0] != num_items_tgt + 1:
        raise ValueError("Target LightGCN item embeddings have incompatible rows.")
    return {
        "source_dir": source_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "overlap": overlap,
        "cold": cold,
        "target_items": target_items,
    }


def create_source_lightgcn(config, output_dir, train_src, id_mapping, force=False):
    embedding_path = os.path.join(output_dir, "source_lightgcn_user_embeddings.npy")
    checkpoint_path = os.path.join(output_dir, "source_lightgcn.pt")
    edges = np.unique(
        np.asarray(train_src[["user", "item"]], dtype=np.int64), axis=0
    )
    num_users = len(id_mapping["src"]["id2user"]) - 1
    num_items = len(id_mapping["src"]["id2item"]) - 1
    fingerprint = source_graph_fingerprint(id_mapping, edges)
    if force:
        for path in (embedding_path, checkpoint_path):
            if os.path.isfile(path):
                os.remove(path)
    if os.path.isfile(embedding_path) and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected = {
            "num_users": num_users,
            "num_items": num_items,
            "source_graph_sha256": fingerprint,
            "seed": int(config["seed"]),
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError("Cached source LightGCN mismatch: {}".format(mismatches))
        embeddings = np.load(embedding_path, mmap_mode="r")
        if embeddings.shape != (num_users + 1, int(checkpoint["embedding_dim"])):
            raise ValueError("Cached source user embeddings have an invalid shape.")
        LOGGER.info("Loaded frozen source LightGCN from %s", checkpoint_path)
        return embeddings, checkpoint

    positives = interaction_sets(edges, num_users)
    settings = config["lightgcn"]
    train_edges, validation_edges = split_lightgcn_edges(
        edges, float(settings["validation_ratio"]), int(config["seed"])
    )
    LOGGER.info(
        "Training source-only LightGCN with %d train and %d validation edges.",
        len(train_edges),
        len(validation_edges),
    )
    _model, best_epoch, best_loss = train_lightgcn_once(
        config,
        train_edges,
        validation_edges,
        positives,
        num_users,
        num_items,
        int(settings["max_epochs"]),
        True,
        int(config["seed"]),
    )
    LOGGER.info("Refitting source-only LightGCN for %d epochs.", best_epoch)
    final_model, _epoch, _loss = train_lightgcn_once(
        config,
        edges,
        np.empty((0, 2), dtype=np.int64),
        positives,
        num_users,
        num_items,
        best_epoch,
        False,
        int(config["seed"]),
    )
    final_model.eval()
    with torch.no_grad():
        users, _items = final_model.propagate()
        users = F.normalize(users, dim=-1).cpu().numpy().astype(np.float32)
    padded = np.zeros((num_users + 1, users.shape[1]), dtype=np.float32)
    padded[1:] = users
    atomic_npy(embedding_path, padded)
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": int(settings["embedding_dim"]),
        "layers": int(settings["layers"]),
        "selected_epochs": int(best_epoch),
        "selection_validation_bpr": float(best_loss),
        "train_interactions": int(len(edges)),
        "source_graph_sha256": fingerprint,
        "seed": int(config["seed"]),
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    LOGGER.info("Saved frozen source LightGCN to %s", checkpoint_path)
    return np.load(embedding_path, mmap_mode="r"), checkpoint


def build_overlap_target_histories(train_tgt, id_mapping, num_overlap, config):
    histories = defaultdict(set)
    src_user2id = id_mapping["src"]["user2id"]
    tgt_id2user = id_mapping["tgt"]["id2user"]
    for target_user, target_item in train_tgt[["user", "item"]].itertuples(
        index=False, name=None
    ):
        raw_user = tgt_id2user[int(target_user)]
        source_user = src_user2id.get(raw_user)
        if source_user is not None and 0 < int(source_user) <= num_overlap:
            histories[int(source_user)].add(int(target_item))
    missing = [user for user in range(1, num_overlap + 1) if not histories[user]]
    if missing:
        raise ValueError(
            "{} overlapped users have no target training interactions.".format(
                len(missing)
            )
        )
    maximum = int(config["evidence"]["max_neighbor_target_items"])
    if maximum <= 0:
        maximum = max(len(values) for values in histories.values())
    matrix = np.zeros((num_overlap + 1, maximum), dtype=np.int64)
    lengths = np.zeros(num_overlap + 1, dtype=np.int32)
    base_seed = int(config["seed"])
    for user in range(1, num_overlap + 1):
        items = np.asarray(sorted(histories[user]), dtype=np.int64)
        if len(items) > maximum:
            rng = np.random.default_rng(base_seed + user * 104729)
            items = np.sort(rng.choice(items, size=maximum, replace=False))
        matrix[user, : len(items)] = items
        lengths[user] = len(items)
    return matrix, lengths, histories


def normalize_semantic_scores(scores):
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    return ((scores - minimum) / np.maximum(maximum - minimum, 1e-8)).astype(
        np.float32
    )


def compute_overlap_support(
    config,
    query_users,
    candidate_items,
    source_users,
    target_items,
    target_history_matrix,
):
    settings = config["evidence"]
    device = torch.device(config["device"])
    num_overlap = target_history_matrix.shape[0] - 1
    neighbor_count = min(int(settings["neighbor_count"]), num_overlap - 1)
    if neighbor_count <= 0:
        raise ValueError("At least two overlapped users are required for grounding.")
    batch_size = int(settings["batch_size"])
    temperature = float(settings["neighbor_temperature"])
    if batch_size <= 0 or temperature <= 0:
        raise ValueError("Evidence batch size and neighbor temperature must be positive.")

    overlap_embeddings = torch.as_tensor(
        np.array(
            source_users[1 : num_overlap + 1], dtype=np.float32, copy=True
        ),
        dtype=torch.float32,
        device=device,
    )
    overlap_embeddings = F.normalize(overlap_embeddings, dim=-1)
    target_embeddings = torch.as_tensor(
        np.array(target_items, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    target_embeddings = F.normalize(target_embeddings, dim=-1)
    history_tensor = torch.as_tensor(
        target_history_matrix, dtype=torch.long, device=device
    )

    output = np.empty(candidate_items.shape, dtype=np.float32)
    all_neighbor_ids = np.empty(
        (len(query_users), neighbor_count), dtype=np.int64
    )
    all_neighbor_weights = np.empty(
        (len(query_users), neighbor_count), dtype=np.float32
    )
    for start in range(0, len(query_users), batch_size):
        stop = min(start + batch_size, len(query_users))
        users = np.asarray(query_users[start:stop], dtype=np.int64)
        query = torch.as_tensor(
            np.array(source_users[users], dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
        query = F.normalize(query, dim=-1)
        similarities = torch.matmul(query, overlap_embeddings.T)
        overlap_rows = np.flatnonzero(users <= num_overlap)
        if len(overlap_rows):
            rows = torch.as_tensor(overlap_rows, dtype=torch.long, device=device)
            columns = torch.as_tensor(
                users[overlap_rows] - 1, dtype=torch.long, device=device
            )
            similarities[rows, columns] = -torch.inf
        values, indices = torch.topk(
            similarities, k=neighbor_count, dim=1, largest=True, sorted=True
        )
        weights = torch.softmax(values / temperature, dim=1)
        neighbor_ids = indices + 1
        histories = history_tensor[neighbor_ids]
        history_mask = histories > 0
        history_embeddings = target_embeddings[histories]
        candidates = torch.as_tensor(
            candidate_items[start:stop], dtype=torch.long, device=device
        )
        candidate_embeddings = target_embeddings[candidates]
        pairwise = torch.einsum(
            "bkd,bnhd->bknh", candidate_embeddings, history_embeddings
        )
        pairwise = pairwise.masked_fill(~history_mask[:, None, :, :], -torch.inf)
        per_neighbor = pairwise.max(dim=-1).values
        no_history = ~history_mask.any(dim=-1)
        per_neighbor = per_neighbor.masked_fill(no_history[:, None, :], 0.0)
        support = (per_neighbor * weights[:, None, :]).sum(dim=-1)
        support = ((support + 1.0) * 0.5).clamp(0.0, 1.0)
        output[start:stop] = support.detach().cpu().numpy().astype(np.float32)
        all_neighbor_ids[start:stop] = neighbor_ids.detach().cpu().numpy()
        all_neighbor_weights[start:stop] = weights.detach().cpu().numpy()
        if stop == len(query_users) or (start // batch_size + 1) % 20 == 0:
            LOGGER.info(
                "Behavioral grounding: %d/%d users.", stop, len(query_users)
            )
    return output, all_neighbor_ids, all_neighbor_weights


def evidence_paths(output_dir, scope):
    return os.path.join(output_dir, "{}_static_evidence.npz".format(scope))


def build_scope_evidence(
    config,
    output_dir,
    scope,
    rankings,
    source_users,
    target_items,
    target_history_matrix,
    force=False,
):
    path = evidence_paths(output_dir, scope)
    if force and os.path.isfile(path):
        os.remove(path)
    if os.path.isfile(path):
        cache = np.load(path, allow_pickle=False)
        LOGGER.info("Loaded cached %s evidence from %s", scope, path)
        return {key: np.asarray(cache[key]) for key in cache.files}
    users, items, raw_semantic = rankings
    semantic = normalize_semantic_scores(raw_semantic)
    overlap, neighbor_ids, neighbor_weights = compute_overlap_support(
        config,
        users,
        items,
        source_users,
        target_items,
        target_history_matrix,
    )
    arrays = {
        "user_ids": users,
        "item_ids": items,
        "semantic_raw": raw_semantic,
        "semantic_evidence": semantic,
        "overlap_evidence": overlap,
        "neighbor_ids": neighbor_ids,
        "neighbor_weights": neighbor_weights,
    }
    atomic_npz(path, **arrays)
    LOGGER.info("Saved %s static evidence to %s", scope, path)
    return arrays


def create_labels(users, items, histories):
    labels = np.zeros(items.shape, dtype=np.float32)
    for row, user in enumerate(users.tolist()):
        positives = histories[int(user)]
        labels[row] = np.fromiter(
            (1.0 if int(item) in positives else 0.0 for item in items[row]),
            dtype=np.float32,
            count=items.shape[1],
        )
    return labels


def split_users(users, config):
    settings = config["calibrator"]
    ratios = np.asarray(settings["user_split"], dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios <= 0):
        raise ValueError("calibrator.user_split must contain three positive ratios.")
    ratios /= ratios.sum()
    rng = np.random.default_rng(int(config["seed"]) + 2027)
    order = rng.permutation(len(users))
    train_end = int(round(len(users) * ratios[0]))
    valid_end = train_end + int(round(len(users) * ratios[1]))
    train_end = min(max(train_end, 1), len(users) - 2)
    valid_end = min(max(valid_end, train_end + 1), len(users) - 1)
    codes = np.empty(len(users), dtype=np.int8)
    codes[order[:train_end]] = 0
    codes[order[train_end:valid_end]] = 1
    codes[order[valid_end:]] = 2
    return codes


class PositiveLinearCalibrator(nn.Module):
    def __init__(self, feature_count=2):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(feature_count))
        self.bias = nn.Parameter(torch.zeros(()))

    @property
    def weights(self):
        return F.softplus(self.raw_weights)

    def forward(self, features):
        return torch.matmul(features, self.weights) + self.bias


def feature_matrix(evidence, row_mask=None):
    semantic = evidence["semantic_evidence"]
    overlap = evidence["overlap_evidence"]
    if row_mask is not None:
        semantic = semantic[row_mask]
        overlap = overlap[row_mask]
    return np.stack([semantic.reshape(-1), overlap.reshape(-1)], axis=1).astype(
        np.float32
    )


def train_calibrator(config, output_dir, evidence, labels, split_codes, force=False):
    path = os.path.join(output_dir, "static_reliability_calibrator.pt")
    if force and os.path.isfile(path):
        os.remove(path)
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location="cpu")
        LOGGER.info("Loaded static reliability calibrator from %s", path)
        return checkpoint

    settings = config["calibrator"]
    device = torch.device(config["device"])
    train_rows = split_codes == 0
    valid_rows = split_codes == 1
    train_x = feature_matrix(evidence, train_rows)
    valid_x = feature_matrix(evidence, valid_rows)
    train_y = labels[train_rows].reshape(-1).astype(np.float32)
    valid_y = labels[valid_rows].reshape(-1).astype(np.float32)
    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = train_x.std(axis=0).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-6).astype(np.float32)
    train_x = (train_x - feature_mean) / feature_std
    valid_x = (valid_x - feature_mean) / feature_std
    train_positive = float(train_y.sum())
    valid_positive = float(valid_y.sum())
    if train_positive <= 0 or valid_positive <= 0:
        raise ValueError("Calibration train/valid users must contain positive hits.")
    natural_ratio = (len(train_y) - train_positive) / train_positive
    positive_weight = min(
        float(settings["positive_weight_cap"]), max(1.0, natural_ratio)
    )
    LOGGER.info(
        "Calibration pairs: train=%d (positive=%d), valid=%d (positive=%d), "
        "positive_weight=%.4f",
        len(train_y),
        int(train_positive),
        len(valid_y),
        int(valid_positive),
        positive_weight,
    )

    model = PositiveLinearCalibrator(2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    generator = np.random.default_rng(int(config["seed"]) + 4049)
    positive_weight_t = torch.tensor(positive_weight, device=device)
    valid_x_t = torch.as_tensor(valid_x, dtype=torch.float32, device=device)
    valid_y_t = torch.as_tensor(valid_y, dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        order = generator.permutation(len(train_y))
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = torch.as_tensor(train_x[indices], dtype=torch.float32, device=device)
            y = torch.as_tensor(train_y[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=positive_weight_t
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            total_count += len(indices)
        model.eval()
        with torch.no_grad():
            valid_loss = float(
                F.binary_cross_entropy_with_logits(
                    model(valid_x_t), valid_y_t, pos_weight=positive_weight_t
                ).cpu()
            )
        if valid_loss < best_loss - float(settings["minimum_delta"]):
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info(
                "Calibrator epoch %d: train=%.6f valid=%.6f best_epoch=%d",
                epoch + 1,
                total_loss / max(total_count, 1),
                valid_loss,
                best_epoch,
            )
        if stale >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Reliability calibrator did not produce a checkpoint.")
    model.load_state_dict(best_state)
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "state_dict": best_state,
        "feature_names": ["semantic_evidence", "overlap_evidence"],
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "positive_weights": True,
        "class_positive_weight": float(positive_weight),
        "logit_prior_correction": float(math.log(positive_weight)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "weights": model.weights.detach().cpu().tolist(),
        "bias": float(model.bias.detach().cpu()),
        "seed": int(config["seed"]),
    }
    atomic_torch_save(path, checkpoint)
    LOGGER.info(
        "Saved calibrator: weights=%s bias=%.6f prior_correction=%.6f",
        checkpoint["weights"],
        checkpoint["bias"],
        checkpoint["logit_prior_correction"],
    )
    return checkpoint


def predict_confidence(checkpoint, evidence, config):
    device = torch.device(config["device"])
    model = PositiveLinearCalibrator(len(checkpoint["feature_names"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = feature_matrix(evidence)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = (features - feature_mean) / feature_std
    batch_size = int(config["calibrator"]["inference_batch_size"])
    output = np.empty(len(features), dtype=np.float32)
    correction = float(checkpoint["logit_prior_correction"])
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            values = torch.as_tensor(
                features[start:stop], dtype=torch.float32, device=device
            )
            output[start:stop] = torch.sigmoid(
                model(values) - correction
            ).cpu().numpy()
    return output.reshape(evidence["semantic_evidence"].shape)


def average_precision(scores, labels):
    labels = labels.astype(np.float64)
    positives = float(labels.sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def binary_diagnostics(scores, labels, bins):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    order = np.argsort(scores, kind="stable")
    chunks = np.array_split(order, int(bins))
    reliability = []
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        reliability.append(
            {
                "bin": index + 1,
                "count": int(len(chunk)),
                "mean_score": float(scores[chunk].mean()),
                "hit_rate": float(labels[chunk].mean()),
            }
        )
    score_means = np.asarray([row["mean_score"] for row in reliability])
    hit_rates = np.asarray([row["hit_rate"] for row in reliability])
    correlation = 0.0
    if len(reliability) >= 2 and np.std(score_means) > 0 and np.std(hit_rates) > 0:
        correlation = float(np.corrcoef(score_means, hit_rates)[0, 1])
    expected_error = sum(
        row["count"] / len(scores) * abs(row["mean_score"] - row["hit_rate"])
        for row in reliability
    )
    top_count = max(1, int(math.ceil(len(scores) * 0.1)))
    top = np.argpartition(scores, len(scores) - top_count)[-top_count:]
    prevalence = float(labels.mean())
    top_rate = float(labels[top].mean())
    return {
        "pairs": int(len(scores)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "average_precision": average_precision(scores, labels),
        "brier": float(np.mean((scores - labels) ** 2)),
        "ece": float(expected_error),
        "top_decile_hit_rate": top_rate,
        "top_decile_lift": float(top_rate / prevalence) if prevalence > 0 else 0.0,
        "bin_hit_correlation": correlation,
        "reliability_bins": reliability,
    }


def stable_rank(items, scores):
    ranked = np.empty_like(items)
    for row in range(len(items)):
        order = np.lexsort((items[row], -scores[row]))
        ranked[row] = items[row, order]
    return ranked


def ranking_metrics(items, scores, labels, cutoffs):
    ranked_items = stable_rank(items, scores)
    label_lookup = [
        set(map(int, row_items[row_labels > 0]))
        for row_items, row_labels in zip(items, labels)
    ]
    result = {}
    for cutoff in cutoffs:
        cutoff = min(int(cutoff), items.shape[1])
        hits = np.asarray(
            [
                sum(int(item) in positives for item in row[:cutoff])
                for row, positives in zip(ranked_items, label_lookup)
            ],
            dtype=np.float64,
        )
        relevant = np.asarray([len(values) for values in label_lookup], dtype=np.float64)
        valid = relevant > 0
        result[str(cutoff)] = {
            "precision": float((hits / cutoff).mean()),
            "recall_within_candidate_pool": float(
                np.mean(hits[valid] / relevant[valid]) if np.any(valid) else 0.0
            ),
            "hit_rate": float((hits > 0).mean()),
            "mean_hits": float(hits.mean()),
        }
    return result


def evaluate_scores(evidence, labels, split_codes, confidence, config):
    split_names = {0: "train", 1: "valid", 2: "test"}
    score_sources = {
        "semantic_only": evidence["semantic_evidence"],
        "overlap_only": evidence["overlap_evidence"],
        "combined_confidence": confidence,
    }
    diagnostics = {}
    for code, name in split_names.items():
        mask = split_codes == code
        diagnostics[name] = {}
        for source, scores in score_sources.items():
            diagnostics[name][source] = {
                "binary": binary_diagnostics(
                    scores[mask],
                    labels[mask],
                    int(config["diagnostics"]["confidence_bins"]),
                ),
                "ranking": ranking_metrics(
                    evidence["item_ids"][mask],
                    scores[mask],
                    labels[mask],
                    config["diagnostics"]["cutoffs"],
                ),
            }
    return diagnostics


def save_confidence_outputs(
    config,
    output_dir,
    overlap_evidence,
    cold_evidence,
    labels,
    split_codes,
    confidence_overlap,
    confidence_cold,
    checkpoint,
    prerequisites,
    source_checkpoint,
):
    overlap_path = os.path.join(output_dir, "overlap_static_confidence.npz")
    cold_path = os.path.join(output_dir, "cold_static_confidence.npz")
    diagnostics_path = os.path.join(output_dir, "diagnostics.json")
    manifest_path = os.path.join(output_dir, "manifest.json")
    atomic_npz(
        overlap_path,
        user_ids=overlap_evidence["user_ids"],
        item_ids=overlap_evidence["item_ids"],
        semantic_evidence=overlap_evidence["semantic_evidence"],
        overlap_evidence=overlap_evidence["overlap_evidence"],
        confidence=confidence_overlap,
        labels=labels,
        split_codes=split_codes,
    )
    ranked_order = np.argsort(-confidence_cold, axis=1, kind="stable")
    ranked_items = np.take_along_axis(
        cold_evidence["item_ids"], ranked_order, axis=1
    )
    ranked_confidence = np.take_along_axis(confidence_cold, ranked_order, axis=1)
    atomic_npz(
        cold_path,
        user_ids=cold_evidence["user_ids"],
        item_ids=cold_evidence["item_ids"],
        semantic_evidence=cold_evidence["semantic_evidence"],
        overlap_evidence=cold_evidence["overlap_evidence"],
        confidence=confidence_cold,
        confidence_ranked_item_ids=ranked_items,
        confidence_ranked_scores=ranked_confidence,
    )
    diagnostics = evaluate_scores(
        overlap_evidence,
        labels,
        split_codes,
        confidence_overlap,
        config,
    )
    atomic_json(diagnostics_path, diagnostics)
    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": config["dataset"],
        "domains": list(config["domains"]),
        "mapping_sha256": prerequisites["manifest"]["mapping_sha256"],
        "target_pattern_manifest_sha256": file_sha256(
            prerequisites["manifest_path"]
        ),
        "source_graph_sha256": source_checkpoint["source_graph_sha256"],
        "candidate_width": int(overlap_evidence["item_ids"].shape[1]),
        "num_overlap_users": int(len(overlap_evidence["user_ids"])),
        "num_cold_users": int(len(cold_evidence["user_ids"])),
        "neighbor_count": int(overlap_evidence["neighbor_ids"].shape[1]),
        "evidence": ["semantic_evidence", "overlap_evidence"],
        "calibrator": {
            "weights": checkpoint["weights"],
            "bias": checkpoint["bias"],
            "feature_mean": checkpoint["feature_mean"],
            "feature_std": checkpoint["feature_std"],
            "class_positive_weight": checkpoint["class_positive_weight"],
            "logit_prior_correction": checkpoint["logit_prior_correction"],
            "best_epoch": checkpoint["best_epoch"],
        },
        "split_codes": {"train": 0, "valid": 1, "test": 2},
        "artifacts": {
            "overlap_static_confidence": os.path.basename(overlap_path),
            "cold_static_confidence": os.path.basename(cold_path),
            "calibrator": "static_reliability_calibrator.pt",
            "diagnostics": os.path.basename(diagnostics_path),
        },
        "seed": int(config["seed"]),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(manifest_path, manifest)
    LOGGER.info("Saved grounded static-confidence outputs to %s", output_dir)
    return diagnostics


def log_test_summary(diagnostics):
    test = diagnostics["test"]
    for source in ("semantic_only", "overlap_only", "combined_confidence"):
        binary = test[source]["binary"]
        rank = test[source]["ranking"]
        top_key = max(rank, key=lambda value: int(value))
        LOGGER.info(
            "[TEST] %-20s AP=%.6f lift@10%%=%.3f bin_corr=%.3f "
            "candidate_hit@%s=%.6f",
            source,
            binary["average_precision"],
            binary["top_decile_lift"],
            binary["bin_hit_correlation"],
            top_key,
            rank[top_key]["hit_rate"],
        )


def phase_index(phase):
    return len(PHASES) - 1 if phase == "all" else PHASES.index(phase)


def main():
    args = parse_args()
    config = load_config(args.config, args.dataset, args.domains, args.device)
    set_seed(int(config["seed"]))
    split_dir, all_users, id_mapping, train_src, train_tgt = load_split(config)
    output_dir = os.path.join(split_dir, config["output_subdir"])
    configure_logging(output_dir)
    LOGGER.info(
        "Dataset=%s domains=%s phase=%s device=%s",
        config["dataset"],
        "+".join(config["domains"]),
        args.phase,
        config["device"],
    )
    prerequisites = load_prerequisites(config, split_dir, all_users, id_mapping)
    last_phase = phase_index(args.phase)

    source_users, source_checkpoint = create_source_lightgcn(
        config,
        output_dir,
        train_src,
        id_mapping,
        force=args.force_phase and args.phase in ("all", "source-encoder"),
    )
    if last_phase == 0:
        return

    num_overlap = len(all_users["overlap_users"])
    history_matrix, _history_lengths, histories = build_overlap_target_histories(
        train_tgt, id_mapping, num_overlap, config
    )
    overlap_evidence = build_scope_evidence(
        config,
        output_dir,
        "overlap",
        prerequisites["overlap"],
        source_users,
        prerequisites["target_items"],
        history_matrix,
        force=args.force_phase and args.phase in ("all", "evidence"),
    )
    cold_evidence = build_scope_evidence(
        config,
        output_dir,
        "cold",
        prerequisites["cold"],
        source_users,
        prerequisites["target_items"],
        history_matrix,
        force=args.force_phase and args.phase in ("all", "evidence"),
    )
    if last_phase == 1:
        return

    labels = create_labels(
        overlap_evidence["user_ids"], overlap_evidence["item_ids"], histories
    )
    split_codes = split_users(overlap_evidence["user_ids"], config)
    checkpoint = train_calibrator(
        config,
        output_dir,
        overlap_evidence,
        labels,
        split_codes,
        force=args.force_phase and args.phase in ("all", "calibrate"),
    )
    confidence_overlap = predict_confidence(checkpoint, overlap_evidence, config)
    confidence_cold = predict_confidence(checkpoint, cold_evidence, config)
    diagnostics = save_confidence_outputs(
        config,
        output_dir,
        overlap_evidence,
        cold_evidence,
        labels,
        split_codes,
        confidence_overlap,
        confidence_cold,
        checkpoint,
        prerequisites,
        source_checkpoint,
    )
    log_test_summary(diagnostics)


if __name__ == "__main__":
    main()
