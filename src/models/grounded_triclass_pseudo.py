"""Static Hit/Near/Unsupported supervision, shared by all CDR backbones.

Run the offline builder with ``python -m models.grounded_triclass_pseudo``.
No API calls are made. Original semantic artifacts and model files are read-only.
"""

import argparse
import fcntl
import hashlib
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.direct_semantic_pseudo import aggregate_user_queries, load_user_interests
from models.grounded_static_confidence import (
    atomic_npy, atomic_npz, configure_logging, create_source_lightgcn, load_config,
)
from models.target_pattern_pseudo import (
    apply_model_domain_profile, atomic_json, atomic_torch_save, config_scalar,
    create_bridge, create_lightgcn, dataset_split_directory, file_sha256,
    load_split, mapping_fingerprint, set_seed,
)


LOGGER = logging.getLogger("grounded-triclass")
VERSION = 1
CLASS_NAMES = ["hit", "near", "unsupported"]
FEATURE_NAMES = ["semantic_cosine", "bridge_cosine", "overlap_vote", "log_popularity"]
DEFAULT_SUBDIR = "grounded_triclass"
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def histories(frame):
    result = defaultdict(set)
    for user, item in frame[["user", "item"]].itertuples(index=False, name=None):
        result[int(user)].add(int(item))
    return result


def split_overlap_users(num_overlap, ratios, seed):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("user_split must contain three positive ratios summing to one.")
    if num_overlap < 6:
        raise ValueError("At least six overlap users are needed for user-disjoint splits.")
    order = np.random.default_rng(seed).permutation(num_overlap)
    counts = [int(num_overlap * ratios[0]), int(num_overlap * ratios[1])]
    if min(counts) < 2 or sum(counts) >= num_overlap:
        raise ValueError("Each split needs users; training needs at least two.")
    codes = np.full(num_overlap, 2, dtype=np.int8)
    codes[order[:counts[0]]] = 0
    codes[order[counts[0]:sum(counts)]] = 1
    return codes


def isolate_teacher_edges(train_tgt, num_overlap):
    result = train_tgt.loc[train_tgt["user"] > num_overlap].copy()
    if result.empty:
        raise ValueError("No target-only training users remain for the frozen teachers.")
    return result


def exact_overlap_votes(query_ids, candidates, source_users, reference_ids,
                        target_histories, neighbor_count, temperature):
    if neighbor_count <= 0 or temperature <= 0 or len(reference_ids) < 2:
        raise ValueError("Invalid overlap neighbor settings or reference population.")
    reference_ids = np.asarray(reference_ids, dtype=np.int64)
    reference = np.asarray(source_users[reference_ids], dtype=np.float32)
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    query = np.asarray(source_users[query_ids], dtype=np.float32)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarities = query @ reference.T
    result = np.zeros(candidates.shape, dtype=np.float32)
    for row, uid in enumerate(query_ids):
        allowed = reference_ids != uid
        positions = np.flatnonzero(allowed)
        order = np.lexsort((reference_ids[positions], -similarities[row, positions]))
        positions = positions[order[:neighbor_count]]
        logits = similarities[row, positions] / temperature
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        for position, weight in zip(positions, weights):
            observed = target_histories.get(int(reference_ids[position]), set())
            result[row] += weight * np.asarray(
                [int(item) in observed and item > 0 for item in candidates[row]],
                dtype=np.float32,
            )
    return result


def support_labels(user_ids, candidates, target_histories, item_neighbors):
    labels = np.full(candidates.shape, -1, dtype=np.int8)
    for row, uid in enumerate(user_ids):
        truth = target_histories.get(int(uid), set())
        near = set()
        for item in truth:
            near.update(int(x) for x in item_neighbors[item] if x > 0)
        for col, item in enumerate(candidates[row]):
            if item > 0:
                labels[row, col] = 0 if item in truth else (1 if item in near else 2)
    return labels


class TriClassCalibrator(nn.Module):
    """A 4-to-3 linear classifier; normalizers use training users only."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("scale", torch.ones(4))
        self.linear = nn.Linear(4, 3)

    def forward(self, evidence):
        return self.linear((evidence - self.mean) / self.scale)


def fit_calibrator(features, labels, split_codes, settings, seed, device):
    train_mask = (split_codes[:, None] == 0) & (labels >= 0)
    valid_mask = (split_codes[:, None] == 1) & (labels >= 0)
    train_x = torch.as_tensor(features[train_mask], dtype=torch.float32, device=device)
    train_y = torch.as_tensor(labels[train_mask], dtype=torch.long, device=device)
    valid_x = torch.as_tensor(features[valid_mask], dtype=torch.float32, device=device)
    valid_y = torch.as_tensor(labels[valid_mask], dtype=torch.long, device=device)
    counts = torch.bincount(train_y, minlength=3)
    if torch.any(counts == 0) or not len(valid_y):
        raise ValueError("Training must contain all three support classes and validation must be nonempty.")
    set_seed(seed)
    model = TriClassCalibrator().to(device)
    with torch.no_grad():
        model.mean.copy_(train_x.mean(0))
        model.scale.copy_(train_x.std(0, unbiased=False).clamp_min(1e-6))
        model.linear.weight.zero_()
        model.linear.bias.copy_((counts.float() / counts.sum()).log())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]),
                                 weight_decay=float(settings["weight_decay"]))
    rng = np.random.default_rng(seed)
    best_loss, stale, best_state, best_epoch = float("inf"), 0, None, 0
    batch_size = int(settings["batch_size"])
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        for start in range(0, len(train_y), batch_size):
            if start == 0:
                order = rng.permutation(len(train_y))
            indices = torch.as_tensor(order[start:start + batch_size], device=device)
            optimizer.zero_grad()
            # Natural-prior CE: no class weighting that silently changes probabilities.
            loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(model(valid_x), valid_y).cpu())
        if value < best_loss - float(settings["minimum_delta"]):
            best_loss, stale, best_epoch = value, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info("Calibrator epoch=%d validation_ce=%.6f best=%d", epoch + 1, value, best_epoch)
        if stale >= int(settings["patience"]):
            break
    model.load_state_dict(best_state)
    return model, {"state_dict": best_state, "best_epoch": best_epoch,
                   "validation_ce": best_loss, "training_class_counts": counts.cpu().tolist(),
                   "class_names": CLASS_NAMES, "feature_names": FEATURE_NAMES}


def predict_probabilities(model, features, valid_mask, device, batch_size=65536):
    flat = features.reshape(-1, 4)
    output = np.zeros((len(flat), 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            x = torch.as_tensor(flat[start:start + batch_size], device=device)
            output[start:start + batch_size] = model(x).softmax(-1).cpu().numpy()
    output = output.reshape(features.shape[:-1] + (3,))
    output[~valid_mask] = 0
    return output


def select_support(items, probabilities, top_k, eta, threshold):
    if not 0 <= eta < 1 or not 0 <= threshold <= 1 or top_k <= 0:
        raise ValueError("Require 0 <= eta < 1, 0 <= threshold <= 1 and K > 0.")
    selected = np.zeros((len(items), top_k), dtype=np.int64)
    weights = np.zeros(selected.shape, dtype=np.float32)
    scores = probabilities[..., 0] + eta * probabilities[..., 1]
    for row in range(len(items)):
        indices = np.flatnonzero((items[row] > 0) & (scores[row] >= threshold) & (scores[row] > 0))
        order = np.lexsort((items[row, indices], -scores[row, indices]))
        indices = indices[order[:top_k]]
        selected[row, :len(indices)] = items[row, indices]
        weights[row, :len(indices)] = scores[row, indices]
    return selected, weights


def support_report(items, probabilities, labels, top_k, eta, threshold):
    selected, weights = select_support(items, probabilities, top_k, eta, threshold)
    hits = near = slots = active = 0
    for row in range(len(items)):
        lookup = {int(i): int(y) for i, y in zip(items[row], labels[row]) if i > 0}
        present = [int(i) for i in selected[row] if i > 0]
        active += bool(present)
        slots += len(present)
        hits += sum(lookup[i] == 0 for i in present)
        near += sum(lookup[i] == 1 for i in present)
    return {"users": len(items), "pairs": slots,
            "user_coverage": active / max(1, len(items)),
            "hit_precision": hits / max(1, slots),
            "near_fraction": near / max(1, slots),
            "supported_precision": (hits + near) / max(1, slots),
            "mean_slot_weight": float(weights.mean()) if weights.size else 0.0}


def choose_thresholds(items, probabilities, labels, split_codes, config):
    settings = config["selection"]
    mask = split_codes == 1
    result = {}
    for eta in settings["eta_values"]:
        trials = []
        for threshold in settings["thresholds"]:
            report = support_report(items[mask], probabilities[mask], labels[mask],
                                    settings["top_k"], eta, threshold)
            report["threshold"] = float(threshold)
            trials.append(report)
        eligible = [r for r in trials if r["user_coverage"] >= settings["minimum_user_coverage"]]
        if not eligible:
            raise ValueError("No threshold satisfies validation coverage; inspect diagnostics before proceeding.")
        # This is a proxy selection rule, not a measured recommendation gain.
        best = max(eligible, key=lambda r: (r["hit_precision"] + eta * r["near_fraction"],
                                            r["user_coverage"], -r["threshold"]))
        result[str(float(eta))] = {"threshold": best["threshold"], "validation": best,
                                 "trials": trials}
    return result


def load_semantics(directory, num_users, num_overlap, num_items):
    prefix = os.path.join(directory, "target_item_semantic_embeddings")
    meta = read_json(prefix + "_metadata.json")
    target = np.load(prefix + ".npy", mmap_mode="r")
    complete = np.load(prefix + "_complete.npy", allow_pickle=False)
    if meta["keys"] != [str(i) for i in range(1, num_items + 1)] or len(target) != num_items:
        raise ValueError("Target semantic IDs do not match the active split.")
    if len(complete) != len(target) or not complete.all() or not np.isfinite(target).all():
        raise ValueError("Target semantic embeddings are incomplete or invalid.")
    queries = np.zeros((num_users + 1, target.shape[1]), dtype=np.float32)
    for scope, users in (("overlap", list(range(1, num_overlap + 1))),
                         ("source", list(range(num_overlap + 1, num_users + 1)))):
        records = load_user_interests(os.path.join(directory, scope + "_user_interests.jsonl"))
        prefix = os.path.join(directory, scope + "_interest_semantic_embeddings")
        embedded = np.load(prefix + ".npy", mmap_mode="r")
        complete = np.load(prefix + "_complete.npy", allow_pickle=False)
        metadata = read_json(prefix + "_metadata.json")
        keys = metadata["keys"]
        if sorted(records) != users or len(keys) != len(embedded) or len(set(keys)) != len(keys):
            raise ValueError("Interest records/embedding rows are not aligned.")
        if not complete.all() or len(complete) != len(embedded) or not np.isfinite(embedded).all():
            raise ValueError("Interest semantic embeddings are incomplete or invalid.")
        if metadata["model"] != meta["model"] or embedded.shape[1] != target.shape[1]:
            raise ValueError("User and item semantic spaces differ.")
        key_rows = {key: i for i, key in enumerate(keys)}
        for start in range(0, len(users), 256):
            batch = users[start:start + 256]
            queries[batch] = aggregate_user_queries(batch, records, embedded, key_rows)
    return queries, target


def rank_top(scores, k):
    # Stable sorting resolves score ties by ascending item ID, including on GPU.
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]


def build_neighbors(collaborative_items, counts, top_k, device, batch_size):
    items = F.normalize(torch.as_tensor(collaborative_items, device=device), dim=-1)
    valid = torch.as_tensor(counts > 0, device=device)
    valid[0] = False
    width = min(int(top_k), int(valid.sum().item()) - 1)
    if width <= 0:
        raise ValueError("Too few target-training items for preference neighbors.")
    result = np.zeros((len(items), width), dtype=np.int64)
    eligible = np.flatnonzero(counts > 0)
    for start in range(0, len(eligible), batch_size):
        ids = torch.as_tensor(eligible[start:start + batch_size], device=device)
        scores = items[ids] @ items.T
        scores[:, ~valid] = -torch.inf
        scores[torch.arange(len(ids), device=device), ids] = -torch.inf
        result[ids.cpu().numpy()] = rank_top(scores, width).cpu().numpy()
    return result


def build_candidates(query_ids, semantic_queries, semantic_items, bridge, collab_items,
                     source_users, reference_ids, target_histories, counts, config):
    settings = config["candidates"]
    device = config["device"]
    n_items = len(counts) - 1
    width = min(int(settings["per_source_k"]), n_items)
    pool_width = min(3 * width, n_items)
    items = np.zeros((len(query_ids), pool_width), dtype=np.int64)
    features = np.zeros((len(query_ids), pool_width, 4), dtype=np.float32)
    sources = np.zeros(items.shape, dtype=np.uint8)
    semantic = F.normalize(torch.tensor(np.asarray(semantic_items), dtype=torch.float32, device=device), dim=-1)
    collaborative = F.normalize(torch.as_tensor(collab_items[1:], dtype=torch.float32, device=device), dim=-1)
    trained = torch.as_tensor(counts[1:] > 0, device=device)
    available = np.flatnonzero(counts[1:] > 0) + 1
    pop_ids = available[np.lexsort((available, -counts[available]))[:width]]
    bridge.eval()
    with torch.no_grad():
        for start in range(0, len(query_ids), int(settings["batch_size"])):
            users = query_ids[start:start + int(settings["batch_size"])]
            query = torch.as_tensor(semantic_queries[users], device=device)
            sem_scores = query @ semantic.T
            g_scores = bridge(query) @ collaborative.T
            sem_top = rank_top(sem_scores, width).cpu().numpy() + 1
            g_rank = g_scores.masked_fill(~trained[None, :], -torch.inf)
            g_top = rank_top(g_rank, min(width, len(available))).cpu().numpy() + 1
            # Only gather candidate scores; do not copy full item-score matrices to CPU.
            batch_items = np.zeros((len(users), pool_width), dtype=np.int64)
            for row in range(len(users)):
                origin = {}
                for bit, values in ((1, sem_top[row]), (2, g_top[row]), (4, pop_ids)):
                    for item in values:
                        origin[int(item)] = origin.get(int(item), 0) | bit
                merged = sorted(origin)
                batch_items[row, :len(merged)] = merged
                sources[start + row, :len(merged)] = [origin[i] for i in merged]
            gather = torch.as_tensor(np.maximum(batch_items - 1, 0), device=device)
            batch_features = np.zeros(batch_items.shape + (4,), dtype=np.float32)
            batch_features[..., 0] = sem_scores.gather(1, gather).cpu().numpy()
            batch_features[..., 1] = g_scores.gather(1, gather).cpu().numpy()
            batch_features[..., 2] = exact_overlap_votes(
                users, batch_items, source_users, reference_ids, target_histories,
                int(settings["neighbor_count"]), float(settings["neighbor_temperature"]))
            batch_features[..., 3] = np.log1p(counts[batch_items])
            batch_features[batch_items == 0] = 0
            items[start:start + len(users)] = batch_items
            features[start:start + len(users)] = batch_features
            if start == 0 or (start // settings["batch_size"]) % 100 == 0:
                LOGGER.info("Candidates %d/%d", start + len(users), len(query_ids))
    return items, features, sources


def prepare(config, output_dir):
    split_dir, all_users, mapping, train_src, train_tgt = load_split(config)
    semantic_dir = os.path.join(split_dir, config["source_subdir"])
    base_manifest = read_json(os.path.join(semantic_dir, "manifest.json"))
    expected = {"dataset": config["dataset"], "domains": config["domains"],
                "mapping_sha256": mapping_fingerprint(mapping)}
    if any(base_manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Semantic manifest belongs to a different split.")
    files = ["source_user_interests.jsonl", "overlap_user_interests.jsonl", "manifest.json"]
    for name in ("source_interest", "overlap_interest", "target_item"):
        files += [name + "_semantic_embeddings" + suffix
                  for suffix in (".npy", "_metadata.json", "_complete.npy")]
    provenance = {name: file_sha256(os.path.join(semantic_dir, name)) for name in files}
    clean_config = {k: v for k, v in config.items() if k not in ("device", "data_path")}
    code_files = ("grounded_triclass_pseudo.py", "grounded_static_confidence.py",
                  "target_pattern_pseudo.py", "direct_semantic_pseudo.py")
    source_cache_dir = os.path.join(split_dir, "grounded_static_confidence")
    source_files = ("source_lightgcn.pt", "source_lightgcn_user_embeddings.npy")
    reused_source = None
    if all(os.path.isfile(os.path.join(source_cache_dir, name)) for name in source_files):
        reused_source = {name: file_sha256(os.path.join(source_cache_dir, name)) for name in source_files}
    identity = {"format_version": VERSION, "config": clean_config,
                "mapping_sha256": mapping_fingerprint(mapping), "semantic_files": provenance,
                "reused_source_teacher_sha256": reused_source,
                "code_sha256": {name: file_sha256(os.path.join(os.path.dirname(__file__), name))
                                for name in code_files},
                "train_src_sha256": file_sha256(os.path.join(split_dir, "train_src.pkl")),
                "train_tgt_sha256": file_sha256(os.path.join(split_dir, "train_tgt.pkl"))}
    identity["build_sha256"] = digest_json(identity)
    path = os.path.join(output_dir, "build_spec.json")
    if os.path.exists(path) and read_json(path) != identity:
        raise ValueError("Build inputs changed. Use a new output_subdir; old caches are immutable.")
    atomic_json(path, identity)
    n_overlap = len(all_users["overlap_users"])
    if mapping["src"]["id2user"][:n_overlap + 1] != mapping["tgt"]["id2user"][:n_overlap + 1]:
        raise ValueError("Overlap user IDs are not shared across source and target.")
    codes = split_overlap_users(n_overlap, config["calibrator"]["user_split"], config["seed"])
    atomic_npz(os.path.join(output_dir, "overlap_user_split.npz"),
               user_ids=np.arange(1, n_overlap + 1), split_codes=codes)
    LOGGER.info("Prepared %s overlap_splits=%s target_only_edges=%d; no cold evaluation labels read.",
                config["domains"], np.bincount(codes).tolist(),
                len(isolate_teacher_edges(train_tgt, n_overlap)))
    return split_dir, all_users, mapping, train_src, train_tgt, semantic_dir, codes, identity


def build(config, until="all"):
    output_dir = os.path.join(dataset_split_directory(config), config["output_subdir"])
    os.makedirs(output_dir, exist_ok=True)
    configure_logging(output_dir)
    with open(os.path.join(output_dir, ".build.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        split_dir, all_users, mapping, src, tgt, semantic_dir, codes, identity = prepare(config, output_dir)
        if until == "prepare":
            return output_dir
        manifest_path = os.path.join(output_dir, "manifest.json")
        if os.path.exists(manifest_path):
            manifest = read_json(manifest_path)
            for name, digest in manifest["artifacts"].items():
                if file_sha256(os.path.join(output_dir, name)) != digest:
                    raise ValueError("Completed artifact was modified: " + name)
            LOGGER.info("Completed immutable cache already exists: %s", output_dir)
            return output_dir
        n_overlap = len(all_users["overlap_users"])
        n_users = len(mapping["src"]["id2user"]) - 1
        n_items = len(mapping["tgt"]["id2item"]) - 1
        queries, sem_items = load_semantics(semantic_dir, n_users, n_overlap, n_items)
        teacher_edges = isolate_teacher_edges(tgt, n_overlap)
        teacher_dir = os.path.join(output_dir, "target_only_teachers")
        os.makedirs(teacher_dir, exist_ok=True)
        target_users, target_items, _ = create_lightgcn(config, teacher_dir, teacher_edges, mapping)
        # The bridge helper uses zero-based user/item arrays, unlike cached rankings.
        bridge, _ = create_bridge(config, teacher_dir, sem_items, target_users[1:],
                                  target_items[1:], teacher_edges)
        existing_source = os.path.join(split_dir, "grounded_static_confidence")
        if not all(os.path.isfile(os.path.join(existing_source, f)) for f in
                   ("source_lightgcn.pt", "source_lightgcn_user_embeddings.npy")):
            existing_source = os.path.join(output_dir, "source_teacher")
            os.makedirs(existing_source, exist_ok=True)
        source_users, _ = create_source_lightgcn(config, existing_source, src, mapping)
        if until == "teachers":
            return output_dir
        counts = np.bincount(teacher_edges["item"].to_numpy(), minlength=n_items + 1)
        target_histories = histories(tgt.loc[tgt["user"] <= n_overlap])
        references = np.flatnonzero(codes == 0) + 1
        overlap_ids = np.arange(1, n_overlap + 1, dtype=np.int64)
        cold_ids = np.arange(n_overlap + 1, n_users + 1, dtype=np.int64)
        for scope, users in (("overlap", overlap_ids), ("cold", cold_ids)):
            path = os.path.join(output_dir, scope + "_candidates.npz")
            if not os.path.exists(path):
                items, features, sources = build_candidates(
                    users, queries, sem_items, bridge, target_items, source_users,
                    references, target_histories, counts, config)
                atomic_npz(path, user_ids=users, item_ids=items, features=features, sources=sources)
        if until == "candidates":
            return output_dir
        neighbor_path = os.path.join(output_dir, "target_item_neighbors.npy")
        if not os.path.exists(neighbor_path):
            atomic_npy(neighbor_path, build_neighbors(target_items, counts,
                       config["labels"]["neighbor_k"], config["device"], config["candidates"]["batch_size"]))
        with np.load(os.path.join(output_dir, "overlap_candidates.npz")) as data:
            items, features = data["item_ids"], data["features"]
        labels = support_labels(overlap_ids, items, target_histories,
                                np.load(neighbor_path, mmap_mode="r"))
        model, checkpoint = fit_calibrator(features, labels, codes, config["calibrator"],
                                           config["seed"], config["device"])
        atomic_torch_save(os.path.join(output_dir, "calibrator.pt"), checkpoint)
        probabilities = predict_probabilities(model, features, items > 0, config["device"])
        selection = choose_thresholds(items, probabilities, labels, codes, config)
        diagnostics = {"class_names": CLASS_NAMES, "validation_ce": checkpoint["validation_ce"],
                       "selection": selection, "splits": {}}
        for code, name in enumerate(("train", "validation", "test")):
            mask = codes == code
            y = labels[mask]; p = probabilities[mask]; valid = y >= 0
            counts_y = np.bincount(y[valid], minlength=3)
            confusion = np.zeros((3, 3), dtype=np.int64)
            np.add.at(confusion, (y[valid], p[valid].argmax(-1)), 1)
            reports = {}
            for eta, selected in selection.items():
                reports[eta] = support_report(items[mask], p, y, config["selection"]["top_k"],
                                               float(eta), selected["threshold"])
            diagnostics["splits"][name] = {"class_counts": counts_y.tolist(),
                "confusion_matrix": confusion.tolist(), "selection": reports,
                "multiclass_brier": float(np.square(p[valid] - np.eye(3)[y[valid]]).sum(-1).mean()),
                "candidate_hit_precision": float((y[valid] == 0).mean()),
                "candidate_near_fraction": float((y[valid] == 1).mean())}
        atomic_npz(os.path.join(output_dir, "overlap_support.npz"), user_ids=overlap_ids,
                   item_ids=items, probabilities=probabilities, labels=labels, split_codes=codes)
        with np.load(os.path.join(output_dir, "cold_candidates.npz")) as data:
            cold_items, cold_features = data["item_ids"], data["features"]
        cold_p = predict_probabilities(model, cold_features, cold_items > 0, config["device"])
        atomic_npz(os.path.join(output_dir, "cold_support.npz"), user_ids=cold_ids,
                   item_ids=cold_items, probabilities=cold_p)
        atomic_json(os.path.join(output_dir, "diagnostics.json"), diagnostics)
        names = ("cold_support.npz", "overlap_support.npz", "calibrator.pt", "diagnostics.json",
                 "overlap_candidates.npz", "cold_candidates.npz", "target_item_neighbors.npy")
        names += tuple("target_only_teachers/" + name for name in
                       ("target_lightgcn.pt", "target_lightgcn_user_embeddings.npy",
                        "target_lightgcn_item_embeddings.npy", "user_semantic_to_collaborative_bridge.pt"))
        if existing_source == os.path.join(output_dir, "source_teacher"):
            names += ("source_teacher/source_lightgcn.pt", "source_teacher/source_lightgcn_user_embeddings.npy")
        manifest = {"format_version": VERSION, "dataset": config["dataset"], "domains": config["domains"],
                    "mapping_sha256": mapping_fingerprint(mapping), "num_users_overlap": n_overlap,
                    "num_users_src": n_users, "num_items_tgt": n_items,
                    "train_src_sha256": identity["train_src_sha256"], "train_tgt_sha256": identity["train_tgt_sha256"],
                    "build_sha256": identity["build_sha256"], "class_names": CLASS_NAMES,
                    "feature_names": FEATURE_NAMES, "top_k": config["selection"]["top_k"],
                    "selection": selection, "teacher_target_users": "target_only",
                    "overlap_vote_users": "calibrator_train_excluding_query",
                    "cold_evaluation_labels_used": False,
                    "artifacts": {name: file_sha256(os.path.join(output_dir, name)) for name in names}}
        atomic_json(manifest_path, manifest)
        LOGGER.info("Completed shared tri-class cache: %s; test=%s", output_dir,
                    diagnostics["splits"]["test"])
        return output_dir


class TriClassPseudoSupervisor:
    """Uniform user sampling with K fixed slots, including zero-weight empty slots."""

    def __init__(self, config, dataloader):
        self.device = config["device"]
        self.rng = np.random.default_rng(int(config_scalar(config.get("pseudo_sampler_seed"), 999)))
        dataset = dataloader.dataset
        self.num_items_tgt = dataset.num_items_tgt
        directory = config.get("triclass_cache_path", "auto")
        split_dir = dataset_split_directory(config)
        if directory in (None, "", "auto"):
            directory = os.path.join(split_dir, DEFAULT_SUBDIR)
        manifest = read_json(os.path.join(directory, "manifest.json"))
        expected = {"format_version": VERSION, "dataset": config["dataset"], "domains": list(config["domains"]),
                    "mapping_sha256": mapping_fingerprint(dataset.id_mapping),
                    "num_users_src": dataset.num_users_src, "num_users_overlap": dataset.num_users_overlap,
                    "num_items_tgt": dataset.num_items_tgt, "class_names": CLASS_NAMES,
                    "cold_evaluation_labels_used": False}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError("Tri-class cache mismatch: " + key)
        for name in ("train_src.pkl", "train_tgt.pkl"):
            key = name.replace(".pkl", "_sha256")
            if file_sha256(os.path.join(split_dir, name)) != manifest[key]:
                raise ValueError("Tri-class cache uses different training interactions.")
        cache_path = os.path.join(directory, "cold_support.npz")
        if file_sha256(cache_path) != manifest["artifacts"]["cold_support.npz"]:
            raise ValueError("Cold support cache checksum mismatch.")
        with np.load(cache_path, allow_pickle=False) as data:
            self.user_ids = data["user_ids"]
            self.candidates = data["item_ids"]
            probabilities = data["probabilities"]
        expected_users = np.arange(dataset.num_users_overlap + 1, dataset.num_users_src + 1)
        if not np.array_equal(self.user_ids, expected_users):
            raise ValueError("Cold support users do not match the source-cold population.")
        if self.candidates.ndim != 2 or len(self.candidates) != len(self.user_ids):
            raise ValueError("Invalid candidate shape.")
        if probabilities.shape != self.candidates.shape + (3,) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid tri-class probability shape or values.")
        valid = self.candidates > 0
        if np.any(self.candidates < 0) or np.any(self.candidates > self.num_items_tgt):
            raise ValueError("Invalid candidate IDs.")
        if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities[valid].sum(-1), 1, atol=1e-5):
            raise ValueError("Tri-class probabilities are not normalized.")
        for row in self.candidates:
            nonzero = row[row > 0]
            if len(nonzero) != len(np.unique(nonzero)):
                raise ValueError("Candidate cache contains duplicate items.")
        eta = float(config_scalar(config.get("near_support_weight"), 0.25))
        key = str(eta)
        if key not in manifest["selection"]:
            raise ValueError("No validation-selected threshold exists for eta=" + key)
        self.top_k = int(manifest["top_k"])
        self.threshold = manifest["selection"][key]["threshold"]
        self.items, self.weights = select_support(self.candidates, probabilities, self.top_k, eta, self.threshold)
        target_history = histories(dataset.df["train_tgt"])
        target_mapping = dataset.id_mapping["tgt"]["user2id"]
        # Domain-local numeric IDs can coincide for different non-overlap users.
        self.target_positive = {
            int(uid): target_history.get(target_mapping.get(dataset.id_mapping["src"]["id2user"][int(uid)]), set())
            for uid in self.user_ids
        }
        self.forbidden = [set(map(int, row[row > 0])) | self.target_positive.get(int(uid), set())
                          for uid, row in zip(self.user_ids, self.candidates)]
        for row, blocked in enumerate(self.forbidden):
            if len(blocked) >= self.num_items_tgt and np.any(self.weights[row] > 0):
                raise ValueError("No target negative remains for a selected cold user.")
        LOGGER.info("Tri-class loaded users=%d K=%d eta=%.3g threshold=%.4g coverage=%.4f mean_slot_weight=%.6f",
                    len(self.user_ids), self.top_k, eta, self.threshold,
                    float((self.weights.sum(1) > 0).mean()), float(self.weights.mean()))

    def sample(self, batch_size):
        if int(batch_size) <= 0 or not len(self.user_ids):
            raise ValueError("Require a positive user batch size and nonempty cold population.")
        rows = self.rng.integers(0, len(self.user_ids), size=int(batch_size))
        users = np.repeat(self.user_ids[rows], self.top_k)
        positives = self.items[rows].reshape(-1).copy()
        weights = self.weights[rows].reshape(-1).copy()
        negatives = np.ones(len(users), dtype=np.int64)
        for slot, row in enumerate(np.repeat(rows, self.top_k)):
            if weights[slot] == 0:
                positives[slot] = 1
                continue
            blocked = self.forbidden[row]
            for _ in range(100):
                candidate = int(self.rng.integers(1, self.num_items_tgt + 1))
                if candidate not in blocked:
                    negatives[slot] = candidate
                    break
            else:
                available = np.setdiff1d(np.arange(1, self.num_items_tgt + 1), list(blocked))
                negatives[slot] = self.rng.choice(available)
        return tuple(torch.as_tensor(x, device=self.device, dtype=dtype) for x, dtype in
                     ((users, torch.long), (positives, torch.long), (negatives, torch.long), (weights, torch.float32)))

    @staticmethod
    def weighted_bpr_loss(users, positives, negatives, weights):
        margin = (users * (positives - negatives)).sum(-1)
        return (weights * F.softplus(-margin)).mean()


class TriClassPseudoMixin:
    """Keep each backbone's training path and scoring; add only a pseudo loss."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.config = config
        self.pseudo_supervisor = None
        if float(config_scalar(config.get("pseudo_loss_weight"), 0)) > 0:
            self.pseudo_supervisor = TriClassPseudoSupervisor(config, dataloader)
        self._triclass_source_graph = self._triclass_target_graph = None
        if self.triclass_backbone == "cdrib" and self.pseudo_supervisor is not None:
            self._triclass_hooks = [self.source_GNN.register_forward_hook(self._capture_triclass_source),
                                   self.target_GNN.register_forward_hook(self._capture_triclass_target)]

    def _capture_triclass_source(self, module, inputs, output):
        self._triclass_source_graph = output

    def _capture_triclass_target(self, module, inputs, output):
        self._triclass_target_graph = output

    def _triclass_embeddings(self, users, pos, neg):
        if self.triclass_backbone == "emcdr":
            return (self.mapping_mlp(self.source_user_mlp(self.source_user_embedding(users))),
                    self.target_item_mlp(self.target_item_embedding(pos)),
                    self.target_item_mlp(self.target_item_embedding(neg)))
        if self.triclass_backbone == "cd_cdr":
            return self.cold_user_condition(users), self.item_emb_tgt(pos), self.item_emb_tgt(neg)
        if self.triclass_backbone == "cdrib":
            if self._triclass_source_graph is None or self._triclass_target_graph is None:
                raise RuntimeError("Missing CDRIB graph outputs from the backbone loss.")
            return (self._triclass_source_graph[0][users - 1],
                    self._triclass_target_graph[1][pos - 1], self._triclass_target_graph[1][neg - 1])
        if self.triclass_backbone == "unicdr":
            items = torch.zeros((len(users), 2, self.max_len), dtype=torch.long, device=self.device)
            scores = torch.full(items.shape, -100.0, device=self.device)
            items[:, 0] = self.history_items_src[users]
            scores[:, 0] = self.history_scores_src[users]
            embedded = torch.cat((self.emb_item_src(items[:, 0]), self.emb_item_tgt(items[:, 1])), dim=1)
            query = self.agg_list[-1](self.emb_user_shared(users), embedded, scores)
            return tuple(F.dropout(x, self.dropout, training=self.training)
                         for x in (query, self.emb_item_tgt(pos), self.emb_item_tgt(neg)))
        raise ValueError("Unsupported tri-class backbone.")

    def calculate_loss(self, interaction, epoch_idx):
        self._triclass_source_graph = self._triclass_target_graph = None
        loss = super().calculate_loss(interaction, epoch_idx)
        try:
            weight = float(config_scalar(self.config.get("pseudo_loss_weight"), 0))
            active = (weight > 0 and epoch_idx >= int(self.config["pseudo_start_epoch"]))
            if self.triclass_backbone == "emcdr":
                active = active and self.stage_id == 2
            if not active:
                return loss
            users, pos, neg, weights = self.pseudo_supervisor.sample(self.config["pseudo_num_per_batch"])
            if not torch.any(weights > 0):
                return loss
            embeddings = self._triclass_embeddings(users, pos, neg)
            return loss + weight * self.pseudo_supervisor.weighted_bpr_loss(*embeddings, weights)
        finally:
            self._triclass_source_graph = self._triclass_target_graph = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["Amazon2014", "Douban"])
    parser.add_argument("--domains", nargs=2, required=True)
    parser.add_argument("--config", default=os.path.join(SRC_ROOT, "configs/model/GroundedTriClass.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--until", choices=["prepare", "teachers", "candidates", "all"], default="all")
    args = parser.parse_args()
    config = load_config(args.config, args.dataset, args.domains, args.device)
    torch.set_num_threads(int(config.get("num_threads", 4)))
    build(config, args.until)


if __name__ == "__main__":
    main()
