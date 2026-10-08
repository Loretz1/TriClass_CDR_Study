#!/usr/bin/env python3
"""Build and load fixed pseudo interactions from direct semantic retrieval."""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from models.target_pattern_pseudo import (
    CACHE_FORMAT_VERSION,
    TargetPatternPseudoSupervisor,
    atomic_json,
    atomic_npz,
    dataset_split_directory,
    file_sha256,
)


LOGGER = logging.getLogger("direct-semantic-pseudo")
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class DirectSemanticPseudoSupervisor(TargetPatternPseudoSupervisor):
    """Load the immutable direct-semantic cache shared by all backbones."""

    def __init__(self, config, dataloader):
        configured_path = config.get("pseudo_cache_path", "auto")
        if configured_path and configured_path != "auto":
            super().__init__(config, dataloader)
            return

        direct_path = os.path.join(
            dataset_split_directory(config),
            "direct_semantic_pseudo",
            "pseudo_rankings.npz",
        )
        original_path = configured_path
        config["pseudo_cache_path"] = direct_path
        try:
            super().__init__(config, dataloader)
        finally:
            config["pseudo_cache_path"] = original_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve target items directly with weighted user-interest semantic "
            "embeddings, without LightGCN or a semantic-to-collaborative bridge."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["Amazon2014"])
    parser.add_argument("--domains", nargs=2, required=True, metavar=("SRC", "TGT"))
    parser.add_argument("--data-path", default="../data")
    parser.add_argument("--source-subdir", default="target_pattern_pseudo")
    parser.add_argument("--output-subdir", default="direct_semantic_pseudo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--user-batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_config(args):
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(SRC_ROOT, data_path))
    return {
        "data_path": data_path,
        "dataset": args.dataset,
        "domains": list(args.domains),
        "only_overlap_users": False,
        "k_cores": 3,
        "shuffle_user_sequence": True,
        "warm_valid_ratio": 0,
        "warm_test_ratio": 0,
        "t_cold_valid": 0.1,
        "t_cold_test": 0.1,
    }


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "build.log"), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_user_interests(path):
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            user = int(record["user_id"])
            if user in records:
                raise ValueError(
                    "Duplicate user {} in {} at line {}.".format(
                        user, path, line_number
                    )
                )
            records[user] = record
    return records


def validate_source_artifacts(source_dir, base_manifest):
    required = [
        "source_user_interests.jsonl",
        "source_interest_semantic_embeddings.npy",
        "source_interest_semantic_embeddings_metadata.json",
        "target_item_semantic_embeddings.npy",
        "target_item_semantic_embeddings_metadata.json",
    ]
    missing = [name for name in required if not os.path.isfile(os.path.join(source_dir, name))]
    if missing:
        raise FileNotFoundError(
            "Direct semantic retrieval is missing source artifacts: {}".format(
                ", ".join(missing)
            )
        )
    expected_users = int(base_manifest["num_users_src"]) - int(
        base_manifest["num_users_overlap"]
    )
    if expected_users <= 0:
        raise ValueError("The source split contains no cold users.")


def aggregate_user_queries(
    batch_users,
    user_interests,
    interest_embeddings,
    interest_row,
):
    dimensions = int(interest_embeddings.shape[1])
    queries = np.zeros((len(batch_users), dimensions), dtype=np.float32)
    for batch_row, user in enumerate(batch_users):
        record = user_interests[user]
        interests = record.get("interests") or []
        if not interests:
            raise ValueError("User {} has no target-domain interests.".format(user))
        total_weight = 0.0
        for interest_index, interest in enumerate(interests):
            key = "{}:{}".format(user, interest_index)
            row = interest_row.get(key)
            if row is None:
                raise KeyError("Missing semantic embedding for interest {}.".format(key))
            weight = float(interest.get("weight", 0.0))
            if weight < 0:
                raise ValueError("Interest {} has a negative weight.".format(key))
            queries[batch_row] += weight * np.asarray(
                interest_embeddings[row], dtype=np.float32
            )
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("User {} has no positive interest weight.".format(user))
        queries[batch_row] /= total_weight
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("A user semantic query has zero norm.")
    return queries / norms


def build_direct_semantic_cache(args):
    config = split_config(args)
    split_dir = dataset_split_directory(config)
    source_dir = os.path.join(split_dir, args.source_subdir)
    output_dir = os.path.join(split_dir, args.output_subdir)
    configure_logging(output_dir)

    source_manifest_path = os.path.join(source_dir, "manifest.json")
    if not os.path.isfile(source_manifest_path):
        raise FileNotFoundError(
            "The source target-pattern manifest does not exist: {}".format(
                source_manifest_path
            )
        )
    base_manifest = load_json(source_manifest_path)
    validate_source_artifacts(source_dir, base_manifest)
    if base_manifest.get("dataset") != args.dataset or base_manifest.get(
        "domains"
    ) != list(args.domains):
        raise ValueError("The source semantic cache belongs to another dataset direction.")

    ranking_path = os.path.join(output_dir, "pseudo_rankings.npz")
    manifest_path = os.path.join(output_dir, "manifest.json")
    if not args.force and os.path.isfile(ranking_path) and os.path.isfile(manifest_path):
        LOGGER.info("Direct semantic cache already exists: %s", ranking_path)
        return ranking_path, manifest_path

    user_interests_path = os.path.join(source_dir, "source_user_interests.jsonl")
    interest_embedding_path = os.path.join(
        source_dir, "source_interest_semantic_embeddings.npy"
    )
    interest_metadata_path = os.path.join(
        source_dir, "source_interest_semantic_embeddings_metadata.json"
    )
    target_embedding_path = os.path.join(
        source_dir, "target_item_semantic_embeddings.npy"
    )
    target_metadata_path = os.path.join(
        source_dir, "target_item_semantic_embeddings_metadata.json"
    )

    user_interests = load_user_interests(user_interests_path)
    users = sorted(user_interests)
    expected_users = list(
        range(
            int(base_manifest["num_users_overlap"]) + 1,
            int(base_manifest["num_users_src"]) + 1,
        )
    )
    if users != expected_users:
        raise ValueError(
            "User interest records do not exactly cover source-side cold users."
        )

    interest_embeddings = np.load(interest_embedding_path, mmap_mode="r")
    target_embeddings = np.load(target_embedding_path, mmap_mode="r")
    interest_metadata = load_json(interest_metadata_path)
    target_metadata = load_json(target_metadata_path)
    interest_keys = list(interest_metadata["keys"])
    target_keys = list(target_metadata["keys"])
    if len(interest_keys) != len(interest_embeddings):
        raise ValueError("Interest metadata keys are not aligned to embeddings.")
    expected_target_keys = [
        str(index) for index in range(1, int(base_manifest["num_items_tgt"]) + 1)
    ]
    if target_keys != expected_target_keys or len(target_embeddings) != len(
        expected_target_keys
    ):
        raise ValueError("Target semantic embeddings are not aligned to target IDs.")
    if interest_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("User-interest and target-item semantic dimensions differ.")

    interest_row = {key: row for row, key in enumerate(interest_keys)}
    official_k = int(base_manifest["official_top_k"])
    cached_k = min(
        int(base_manifest["cached_top_k"]), int(base_manifest["num_items_tgt"])
    )
    if official_k <= 0 or official_k > cached_k:
        raise ValueError("Invalid official/cached Top-K values in source manifest.")
    if args.user_batch_size <= 0:
        raise ValueError("--user-batch-size must be positive.")

    device = resolve_device(args.device)
    LOGGER.info(
        "Building direct semantic cache for %s %s on %s: %d users, %d items.",
        args.dataset,
        "+".join(args.domains),
        device,
        len(users),
        len(target_embeddings),
    )
    target_tensor = torch.tensor(
        np.asarray(target_embeddings), dtype=torch.float32, device=device
    )
    target_tensor = F.normalize(target_tensor, dim=1)
    ranked_items = np.empty((len(users), cached_k), dtype=np.int64)
    ranked_scores = np.empty((len(users), cached_k), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(users), args.user_batch_size):
            batch_users = users[start : start + args.user_batch_size]
            queries = aggregate_user_queries(
                batch_users,
                user_interests,
                interest_embeddings,
                interest_row,
            )
            query_tensor = torch.as_tensor(
                queries, dtype=torch.float32, device=device
            )
            values, selected = torch.topk(
                torch.matmul(query_tensor, target_tensor.t()),
                k=cached_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            stop = start + len(batch_users)
            ranked_items[start:stop] = selected.cpu().numpy().astype(np.int64) + 1
            ranked_scores[start:stop] = values.cpu().numpy().astype(np.float32)
            if stop % 1000 < len(batch_users) or stop == len(users):
                LOGGER.info("Direct semantic retrieval: %d/%d users.", stop, len(users))

    atomic_npz(
        ranking_path,
        user_ids=np.asarray(users, dtype=np.int64),
        item_ids=ranked_items,
        scores=ranked_scores,
    )
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset": base_manifest["dataset"],
        "domains": base_manifest["domains"],
        "num_users_overlap": int(base_manifest["num_users_overlap"]),
        "num_users_src": int(base_manifest["num_users_src"]),
        "num_items_tgt": int(base_manifest["num_items_tgt"]),
        "mapping_sha256": base_manifest["mapping_sha256"],
        "official_top_k": official_k,
        "cached_top_k": cached_k,
        "cold_source_users": len(users),
        "pseudo_pairs": len(users) * official_k,
        "retrieval_type": "direct_semantic_cosine",
        "query_aggregation": "weighted_target_interest_mean",
        "semantic_source_subdir": args.source_subdir,
        "embedding_model": interest_metadata.get("model"),
        "embedding_dimensions": int(interest_embeddings.shape[1]),
        "seed": int(base_manifest.get("seed", 999)),
        "created_at": datetime.now().astimezone().isoformat(),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "user_interests_sha256": file_sha256(user_interests_path),
        "interest_embeddings_sha256": file_sha256(interest_embedding_path),
        "target_embeddings_sha256": file_sha256(target_embedding_path),
        "rankings_sha256": file_sha256(ranking_path),
    }
    atomic_json(manifest_path, manifest)
    LOGGER.info(
        "Saved direct semantic cache: %d users x Top-%d (%s).",
        len(users),
        cached_k,
        ranking_path,
    )
    return ranking_path, manifest_path


def main():
    args = parse_args()
    build_direct_semantic_cache(args)


if __name__ == "__main__":
    main()
