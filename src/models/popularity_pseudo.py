"""Fixed pseudo interactions built from target-domain item popularity."""

import fcntl
import logging
import os
from datetime import datetime

import numpy as np

from models.target_pattern_pseudo import (
    CACHE_FORMAT_VERSION,
    TargetPatternPseudoSupervisor,
    atomic_json,
    atomic_npz,
    config_scalar,
    dataset_split_directory,
    mapping_fingerprint,
)


LOGGER = logging.getLogger("popularity-pseudo")


def _default_cache_path(config):
    return os.path.join(
        dataset_split_directory(config),
        "popularity_pseudo",
        "pseudo_rankings.npz",
    )


def _build_cache(config, dataloader, cache_path):
    dataset = dataloader.dataset
    output_dir = os.path.dirname(cache_path)
    manifest_path = os.path.join(output_dir, "manifest.json")
    os.makedirs(output_dir, exist_ok=True)

    lock_path = os.path.join(output_dir, ".build.lock")
    with open(lock_path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if os.path.isfile(cache_path) and os.path.isfile(manifest_path):
            return

        official_top_k = int(
            config_scalar(config.get("popularity_official_top_k"), 5)
        )
        cached_top_k = int(
            config_scalar(config.get("popularity_cached_top_k"), 100)
        )
        cached_top_k = min(cached_top_k, dataset.num_items_tgt)
        if official_top_k <= 0 or official_top_k > cached_top_k:
            raise ValueError(
                "popularity_official_top_k must lie in [1, cached_top_k]."
            )

        candidates = np.arange(1, dataset.num_items_tgt + 1, dtype=np.int64)
        target_items = np.asarray(dataset.train_tgt["item"], dtype=np.int64)
        counts = np.bincount(
            target_items, minlength=dataset.num_items_tgt + 1
        )[1:]
        order = np.lexsort((candidates, -counts))[:cached_top_k]
        popular_items = candidates[order]
        maximum = max(1, int(counts.max()))
        popular_scores = counts[order].astype(np.float32) / float(maximum)

        user_ids = np.arange(
            dataset.num_users_overlap + 1,
            dataset.num_users_src + 1,
            dtype=np.int64,
        )
        ranked_items = np.tile(popular_items, (len(user_ids), 1))
        ranked_scores = np.tile(popular_scores, (len(user_ids), 1))

        atomic_npz(
            cache_path,
            user_ids=user_ids,
            item_ids=ranked_items,
            scores=ranked_scores,
        )
        atomic_json(
            manifest_path,
            {
                "format_version": CACHE_FORMAT_VERSION,
                "dataset": config["dataset"],
                "domains": list(config["domains"]),
                "num_users_overlap": dataset.num_users_overlap,
                "num_users_src": dataset.num_users_src,
                "num_items_tgt": dataset.num_items_tgt,
                "mapping_sha256": mapping_fingerprint(dataset.id_mapping),
                "official_top_k": official_top_k,
                "cached_top_k": cached_top_k,
                "cold_source_users": len(user_ids),
                "pseudo_pairs": len(user_ids) * official_top_k,
                "method": "global_target_train_popularity",
                "target_train_interactions": int(len(dataset.train_tgt)),
                "created_at": datetime.now().astimezone().isoformat(),
            },
        )
        LOGGER.info(
            "Built popularity pseudo cache: %d users x Top-%d from %d target "
            "interactions (%s).",
            len(user_ids),
            cached_top_k,
            len(dataset.train_tgt),
            cache_path,
        )


class PopularityPseudoSupervisor(TargetPatternPseudoSupervisor):
    """Give every cold user the same target-domain popularity ranking."""

    def __init__(self, config, dataloader):
        cache_path = config.get("pseudo_cache_path", "auto")
        if not cache_path or cache_path == "auto":
            cache_path = _default_cache_path(config)
            config["pseudo_cache_path"] = cache_path
        cache_path = os.path.abspath(cache_path)
        _build_cache(config, dataloader, cache_path)
        super().__init__(config, dataloader)
        self.logger.info(
            "[POPULARITY PSEUDO] Shared target Top-%d assigned to %d cold users.",
            self.official_top_k,
            len(self.user_ids),
        )
