import hashlib
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.target_pattern_pseudo import (
    TargetPatternPseudoSupervisor,
    config_scalar,
    dataset_split_directory,
    mapping_fingerprint,
)


CONFIDENCE_CACHE_FORMAT_VERSION = 1


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GroundedConfidencePseudoSupervisor(TargetPatternPseudoSupervisor):
    """Apply calibrated static confidence to the fixed Top-K pseudo pairs."""

    SOURCES = {
        "semantic": "semantic_evidence",
        "overlap": "overlap_evidence",
        "combined": "confidence",
    }

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.logger = logging.getLogger()
        self.confidence_source = str(
            config_scalar(config.get("confidence_source"), "combined")
        ).lower()
        if self.confidence_source not in self.SOURCES:
            raise ValueError(
                "confidence_source must be one of: {}.".format(
                    ", ".join(sorted(self.SOURCES))
                )
            )

        self.confidence_power = float(
            config_scalar(config.get("confidence_power"), 1.0)
        )
        self.confidence_epsilon = float(
            config_scalar(config.get("confidence_epsilon"), 1.0e-8)
        )
        if self.confidence_power < 0:
            raise ValueError("confidence_power must be non-negative.")
        if self.confidence_epsilon <= 0:
            raise ValueError("confidence_epsilon must be positive.")

        cache_path = config.get("grounded_confidence_cache_path", "auto")
        if not cache_path or cache_path == "auto":
            cache_path = os.path.join(
                dataset_split_directory(config),
                "grounded_static_confidence",
                "cold_static_confidence.npz",
            )
        self.confidence_cache_path = os.path.abspath(cache_path)
        self.confidence_manifest_path = os.path.join(
            os.path.dirname(self.confidence_cache_path), "manifest.json"
        )
        self._load_confidence(config, dataloader)

    def _load_confidence(self, config, dataloader):
        if not os.path.isfile(self.confidence_cache_path):
            raise FileNotFoundError(
                "Grounded confidence cache does not exist: {}.".format(
                    self.confidence_cache_path
                )
            )
        if not os.path.isfile(self.confidence_manifest_path):
            raise FileNotFoundError(
                "Grounded confidence manifest does not exist: {}.".format(
                    self.confidence_manifest_path
                )
            )

        with open(self.confidence_manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        expected = {
            "format_version": CONFIDENCE_CACHE_FORMAT_VERSION,
            "dataset": dataloader.dataset.config["dataset"],
            "domains": list(dataloader.dataset.config["domains"]),
            "num_cold_users": len(self.user_ids),
            "mapping_sha256": mapping_fingerprint(self.id_mapping),
            "target_pattern_manifest_sha256": _file_sha256(self.manifest_path),
        }
        mismatches = {
            key: (manifest.get(key), wanted)
            for key, wanted in expected.items()
            if manifest.get(key) != wanted
        }
        if mismatches:
            detail = ", ".join(
                "{}: cache={!r}, expected={!r}".format(key, actual, wanted)
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(
                "Grounded confidence cache is incompatible with the active "
                "pseudo cache ({}).".format(detail)
            )

        with np.load(self.confidence_cache_path, allow_pickle=False) as cache:
            required = {
                "user_ids",
                "item_ids",
                self.SOURCES[self.confidence_source],
            }
            missing = sorted(required.difference(cache.files))
            if missing:
                raise ValueError(
                    "Grounded confidence cache is missing: {}.".format(
                        ", ".join(missing)
                    )
                )
            user_ids = np.asarray(cache["user_ids"], dtype=np.int64)
            item_ids = np.asarray(cache["item_ids"], dtype=np.int64)
            evidence = np.asarray(
                cache[self.SOURCES[self.confidence_source]], dtype=np.float32
            )

        if not np.array_equal(user_ids, self.user_ids):
            raise ValueError(
                "Grounded confidence users are not aligned with the pseudo cache."
            )
        if item_ids.ndim != 2 or evidence.shape != item_ids.shape:
            raise ValueError(
                "Grounded confidence item_ids and evidence must be equally shaped."
            )
        if item_ids.shape[1] < self.official_top_k:
            raise ValueError(
                "Grounded confidence width is smaller than the official Top-K."
            )
        if not np.array_equal(
            item_ids[:, : self.official_top_k], self.item_ids
        ):
            raise ValueError(
                "Grounded confidence items do not preserve the original Top-K."
            )

        fixed_evidence = evidence[:, : self.official_top_k]
        if not np.all(np.isfinite(fixed_evidence)):
            raise ValueError("Grounded confidence contains a non-finite value.")
        if np.any(fixed_evidence < 0) or np.any(fixed_evidence > 1):
            raise ValueError("Grounded confidence values must lie in [0, 1].")

        powered = np.power(
            np.maximum(fixed_evidence, self.confidence_epsilon),
            self.confidence_power,
        ).astype(np.float32, copy=False)
        self.pair_confidence = fixed_evidence.reshape(-1)
        self.pair_weights = powered.reshape(-1)
        self.logger.info(
            "[GROUNDED CONFIDENCE] source=%s power=%.4g users=%d Top-%d "
            "raw=[%.6g, %.6g] weight=[%.6g, %.6g] cache=%s",
            self.confidence_source,
            self.confidence_power,
            len(self.user_ids),
            self.official_top_k,
            float(fixed_evidence.min()),
            float(fixed_evidence.max()),
            float(powered.min()),
            float(powered.max()),
            self.confidence_cache_path,
        )

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("pseudo_num_per_batch must be positive.")

        pair_index = self.rng.integers(
            0, len(self.pair_users), size=batch_size
        )
        users = self.pair_users[pair_index]
        positives = self.pair_items[pair_index]
        weights = self.pair_weights[pair_index]
        negatives = np.empty(batch_size, dtype=np.int64)

        for row, user in enumerate(users.tolist()):
            forbidden = self.pseudo_positive[int(user)] | self.target_positive[int(user)]
            if len(forbidden) >= self.num_items_tgt:
                raise RuntimeError(
                    "No target negative remains for source user {}.".format(user)
                )
            while True:
                candidate = int(self.rng.integers(1, self.num_items_tgt + 1))
                if candidate not in forbidden:
                    negatives[row] = candidate
                    break

        return (
            torch.as_tensor(users, dtype=torch.long, device=self.device),
            torch.as_tensor(positives, dtype=torch.long, device=self.device),
            torch.as_tensor(negatives, dtype=torch.long, device=self.device),
            torch.as_tensor(weights, dtype=torch.float, device=self.device),
        )

    @staticmethod
    def weighted_bpr_loss(
        user_embedding,
        positive_embedding,
        negative_embedding,
        weights,
    ):
        positive_score = (user_embedding * positive_embedding).sum(dim=-1)
        negative_score = (user_embedding * negative_embedding).sum(dim=-1)
        pair_loss = -F.logsigmoid(positive_score - negative_score)
        weights = weights.to(device=pair_loss.device, dtype=pair_loss.dtype)
        denominator = weights.sum().clamp_min(torch.finfo(pair_loss.dtype).eps)
        return (weights * pair_loss).sum() / denominator
