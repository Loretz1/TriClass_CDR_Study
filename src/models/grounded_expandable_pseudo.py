"""Stage-v3 pseudo supervision built on the adaptive stage-v2.1 model.

The semantic Top-K set remains an immutable anchor. An EMA copy of the
adaptive model proposes full-catalog candidates, while a positive linear
calibrator estimates confidence from semantic, overlap, and model evidence.
Expansion is deliberately conservative and updates both the direct pseudo
loss and the adaptive residual branch.
"""

import fcntl
import json
import logging
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from models.grounded_confidence_pseudo import GroundedConfidencePseudoSupervisor
from models.grounded_static_confidence import PositiveLinearCalibrator, average_precision
from models.target_pattern_pseudo import (
    TargetPatternBridge,
    atomic_npz,
    config_scalar,
    dataset_split_directory,
)


V3_SETTINGS = {
    "strict": {
        "minimum_user_gate": 0.75,
        "maximum_acceptance_fraction": 0.0025,
        "proposal_quantile": 0.75,
        "anchor_confidence_ratio": 1.0,
        "anchor_reweight_blend": 0.0,
    },
    "balanced": {
        "minimum_user_gate": 0.50,
        "maximum_acceptance_fraction": 0.005,
        "proposal_quantile": 0.50,
        "anchor_confidence_ratio": 0.75,
        "anchor_reweight_blend": 0.15,
    },
    "broad": {
        "minimum_user_gate": 0.25,
        "maximum_acceptance_fraction": 0.01,
        "proposal_quantile": 0.0,
        "anchor_confidence_ratio": 0.50,
        "anchor_reweight_blend": 0.30,
    },
}


class ExpandableGroundedPseudoSupervisor(GroundedConfidencePseudoSupervisor):
    """Maintain an anchored support set and conservatively admit EMA proposals."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.config = config
        self.logger = logging.getLogger()
        self.seed = int(config_scalar(config.get("pseudo_sampler_seed"), 999))
        setting_name = str(config_scalar(config.get("v3_setting"), "balanced"))
        if setting_name not in V3_SETTINGS:
            raise ValueError(
                "Unknown v3_setting {!r}. Available settings: {}.".format(
                    setting_name, ", ".join(V3_SETTINGS)
                )
            )
        self.setting_name = setting_name
        self.setting = V3_SETTINGS[setting_name]
        self.user_batch_size = int(config_scalar(config.get("dynamic_user_batch_size"), 64))
        self.calibration_width = int(
            config_scalar(config.get("dynamic_calibrator_candidate_width"), 20)
        )
        self.expansion_top_k = int(config_scalar(config.get("expansion_top_k"), 1))
        self.maximum_items = int(
            config_scalar(config.get("max_pseudo_items_per_user"), 8)
        )
        self.calibration_quantile = float(
            config_scalar(config.get("dynamic_calibration_quantile"), 0.90)
        )
        self.minimum_validation_precision = float(
            config_scalar(config.get("dynamic_min_validation_precision"), 0.05)
        )
        self.minimum_confidence = float(
            config_scalar(config.get("dynamic_min_confidence"), 0.0)
        )
        self.anchor_quantile = float(
            config_scalar(config.get("dynamic_anchor_quantile"), 0.0)
        )
        if self.user_batch_size <= 0:
            raise ValueError("dynamic_user_batch_size must be positive.")
        if self.calibration_width < 2:
            raise ValueError("dynamic_calibrator_candidate_width must be at least 2.")
        if self.expansion_top_k <= 0:
            raise ValueError("expansion_top_k must be positive.")
        if self.maximum_items < self.official_top_k:
            raise ValueError(
                "max_pseudo_items_per_user cannot be smaller than official Top-K."
            )
        if not 0.0 <= self.calibration_quantile < 1.0:
            raise ValueError("dynamic_calibration_quantile must lie in [0, 1).")
        if not 0.0 <= self.minimum_validation_precision <= 1.0:
            raise ValueError("dynamic_min_validation_precision must lie in [0, 1].")
        if not 0.0 <= self.anchor_quantile <= 1.0:
            raise ValueError("dynamic_anchor_quantile must lie in [0, 1].")

        self.split_dir = dataset_split_directory(config)
        self.target_pattern_dir = os.path.dirname(self.cache_path)
        self.grounded_dir = os.path.dirname(self.confidence_cache_path)
        self._load_dynamic_artifacts(dataloader)
        self._load_residual_base_candidates(config)
        self.current_items = [list(map(int, row)) for row in self.item_ids]
        self.current_confidences = [
            list(map(float, row))
            for row in self.pair_confidence.reshape(
                len(self.user_ids), self.official_top_k
            )
        ]
        self.residual_dynamic_confidence = self.residual_base_confidence.copy()
        self.user_gate = np.zeros(self.num_users_src + 1, dtype=np.float32)
        self.refresh_count = 0
        self._target_collaborative_items_device = None
        self._target_history_matrix_device = None
        self.logger.info(
            "[V3 PSEUDO] setting=%s min_gate=%.3f max_acceptance=%.4f "
            "proposal_quantile=%.3f anchor_ratio=%.3f reweight_blend=%.3f",
            self.setting_name,
            self.setting["minimum_user_gate"],
            self.setting["maximum_acceptance_fraction"],
            self.setting["proposal_quantile"],
            self.setting["anchor_confidence_ratio"],
            self.setting["anchor_reweight_blend"],
        )

    @staticmethod
    def _read_jsonl(path, key):
        records = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records[int(row[key])] = row
        return records

    def _load_dynamic_artifacts(self, dataloader):
        cold_path = os.path.join(self.grounded_dir, "cold_static_evidence.npz")
        overlap_path = os.path.join(self.grounded_dir, "overlap_static_evidence.npz")
        with np.load(cold_path, allow_pickle=False) as cache:
            self.cold_evidence = {key: np.asarray(cache[key]) for key in cache.files}
        with np.load(overlap_path, allow_pickle=False) as cache:
            self.overlap_evidence = {
                key: np.asarray(cache[key]) for key in cache.files
            }
        if not np.array_equal(self.cold_evidence["user_ids"], self.user_ids):
            raise ValueError("Cold dynamic evidence is not aligned with pseudo users.")

        target_path = os.path.join(
            self.target_pattern_dir, "target_lightgcn_item_embeddings.npy"
        )
        target = np.asarray(np.load(target_path), dtype=np.float32)
        target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
        target[0] = 0.0
        self.target_collaborative_items = target
        self.cold_semantic_queries = self._load_or_build_semantic_queries(
            "cold", self.cold_evidence["user_ids"]
        )
        self.overlap_semantic_queries = self._load_or_build_semantic_queries(
            "overlap", self.overlap_evidence["user_ids"]
        )
        self._build_target_histories(dataloader)
        self.overlap_split_codes = self._split_overlap_users(
            len(self.overlap_evidence["user_ids"])
        )

    def _load_residual_base_candidates(self, config):
        with np.load(self.confidence_cache_path, allow_pickle=False) as cache:
            all_item_ids = np.asarray(cache["item_ids"], dtype=np.int64)
            evidence = np.asarray(
                cache[self.SOURCES[self.confidence_source]], dtype=np.float32
            )
            if (
                self.confidence_source == "combined"
                and "confidence_ranked_item_ids" in cache.files
                and "confidence_ranked_scores" in cache.files
            ):
                ranked_items = np.asarray(
                    cache["confidence_ranked_item_ids"], dtype=np.int64
                )
                ranked_confidence = np.asarray(
                    cache["confidence_ranked_scores"], dtype=np.float32
                )
            else:
                order = np.argsort(-evidence, axis=1, kind="stable")
                ranked_items = np.take_along_axis(all_item_ids, order, axis=1)
                ranked_confidence = np.take_along_axis(evidence, order, axis=1)

        pool_size = min(
            int(config_scalar(config.get("residual_candidate_pool_size"), 100)),
            ranked_items.shape[1],
        )
        top_k = int(config_scalar(config.get("residual_top_k"), 1))
        if top_k > pool_size:
            raise ValueError("residual_top_k exceeds the available confidence pool.")
        self.residual_base_items = ranked_items[:, :pool_size][:, :top_k]
        self.residual_base_confidence = ranked_confidence[:, :pool_size][:, :top_k]

    def _load_or_build_semantic_queries(self, scope, expected_users):
        cache_path = os.path.join(
            self.grounded_dir,
            "{}_semantic_collaborative_queries.npz".format(scope),
        )
        lock_path = cache_path + ".lock"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if os.path.isfile(cache_path):
                with np.load(cache_path, allow_pickle=False) as cache:
                    users = np.asarray(cache["user_ids"], dtype=np.int64)
                    queries = np.asarray(cache["queries"], dtype=np.float32)
                if np.array_equal(users, expected_users):
                    return queries
                raise ValueError(
                    "Cached {} semantic queries are not aligned.".format(scope)
                )

            prefix = "source" if scope == "cold" else "overlap"
            interest_path = os.path.join(
                self.target_pattern_dir, "{}_user_interests.jsonl".format(prefix)
            )
            embedding_path = os.path.join(
                self.target_pattern_dir,
                "{}_interest_semantic_embeddings.npy".format(prefix),
            )
            metadata_path = os.path.join(
                self.target_pattern_dir,
                "{}_interest_semantic_embeddings_metadata.json".format(prefix),
            )
            bridge_path = os.path.join(
                self.target_pattern_dir, "user_semantic_to_collaborative_bridge.pt"
            )
            records = self._read_jsonl(interest_path, "user_id")
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            key_to_row = {
                str(key): index for index, key in enumerate(metadata["keys"])
            }
            embeddings = np.load(embedding_path, mmap_mode="r")
            checkpoint = torch.load(bridge_path, map_location="cpu")
            bridge = TargetPatternBridge(
                checkpoint["input_dim"],
                checkpoint["hidden_dim"],
                checkpoint["output_dim"],
                checkpoint["dropout"],
            )
            bridge.load_state_dict(checkpoint["state_dict"])
            bridge.eval()
            queries = np.empty(
                (len(expected_users), checkpoint["output_dim"]), dtype=np.float32
            )
            with torch.no_grad():
                for start in range(0, len(expected_users), 256):
                    stop = min(start + 256, len(expected_users))
                    semantic = np.zeros(
                        (stop - start, checkpoint["input_dim"]), dtype=np.float32
                    )
                    for row, user in enumerate(expected_users[start:stop]):
                        record = records.get(int(user))
                        if record is None:
                            raise KeyError(
                                "Missing {} interest record for user {}.".format(
                                    scope, int(user)
                                )
                            )
                        for interest_index, interest in enumerate(record["interests"]):
                            key = "{}:{}".format(int(user), interest_index)
                            semantic[row] += float(interest["weight"]) * np.asarray(
                                embeddings[key_to_row[key]], dtype=np.float32
                            )
                    semantic /= np.maximum(
                        np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12
                    )
                    queries[start:stop] = bridge(
                        torch.as_tensor(semantic, dtype=torch.float32)
                    ).cpu().numpy()
            atomic_npz(
                cache_path,
                user_ids=np.asarray(expected_users, dtype=np.int64),
                queries=queries,
            )
            self.logger.info(
                "[V3 PSEUDO] Cached %s semantic queries at %s", scope, cache_path
            )
            return queries

    def _build_target_histories(self, dataloader):
        source_id2user = self.id_mapping["src"]["id2user"]
        target_user2id = self.id_mapping["tgt"]["user2id"]
        target_sets = dataloader.dataset.positive_items_tgt
        self.all_target_positive = {}
        overlap_histories = defaultdict(set)
        for source_user in range(1, self.num_users_src + 1):
            raw_user = source_id2user[source_user]
            target_user = target_user2id.get(raw_user)
            values = (
                set(map(int, target_sets.get(target_user, set())))
                if target_user is not None
                else set()
            )
            self.all_target_positive[source_user] = values
            if source_user <= self.num_users_overlap:
                overlap_histories[source_user] = values

        maximum = int(
            config_scalar(self.config.get("dynamic_max_neighbor_target_items"), 64)
        )
        self.target_history_matrix = np.zeros(
            (self.num_users_overlap + 1, maximum), dtype=np.int64
        )
        for user in range(1, self.num_users_overlap + 1):
            items = np.asarray(sorted(overlap_histories[user]), dtype=np.int64)
            if len(items) > maximum:
                rng = np.random.default_rng(self.seed + user * 104729)
                items = np.sort(rng.choice(items, size=maximum, replace=False))
            self.target_history_matrix[user, : len(items)] = items

    def _split_overlap_users(self, count):
        if count < 3:
            raise ValueError("At least three overlap users are required for calibration.")
        ratios = np.asarray(
            self.config.get("dynamic_calibrator_user_split", [0.6, 0.2, 0.2]),
            dtype=np.float64,
        )
        ratios /= ratios.sum()
        order = np.random.default_rng(self.seed + 2027).permutation(count)
        train_end = min(max(int(round(count * ratios[0])), 1), count - 2)
        valid_end = min(
            max(train_end + int(round(count * ratios[1])), train_end + 1),
            count - 1,
        )
        codes = np.empty(count, dtype=np.int8)
        codes[order[:train_end]] = 0
        codes[order[train_end:valid_end]] = 1
        codes[order[valid_end:]] = 2
        return codes

    def set_user_gate(self, gate):
        gate = np.asarray(gate, dtype=np.float32)
        if gate.shape != (self.num_users_src + 1,):
            raise ValueError("V3 user gate has an incompatible shape.")
        self.user_gate = gate.copy()

    def _labels_for_candidates(self, users, items):
        labels = np.zeros(items.shape, dtype=np.float32)
        for row, user in enumerate(users.tolist()):
            positives = self.all_target_positive[int(user)]
            labels[row] = np.fromiter(
                (1.0 if int(item) in positives else 0.0 for item in items[row]),
                dtype=np.float32,
                count=items.shape[1],
            )
        return labels

    @staticmethod
    def _model_evidence(scores, item_tensor):
        catalog = scores[:, 1:]
        mean = catalog.mean(dim=1, keepdim=True)
        std = catalog.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
        raw = scores.gather(1, item_tensor)
        return torch.sigmoid(((raw - mean) / std).clamp(-12.0, 12.0))

    def _static_device_tensors(self, device):
        if (
            self._target_collaborative_items_device is None
            or self._target_collaborative_items_device.device != device
        ):
            self._target_collaborative_items_device = torch.as_tensor(
                self.target_collaborative_items,
                dtype=torch.float32,
                device=device,
            )
            self._target_history_matrix_device = torch.as_tensor(
                self.target_history_matrix,
                dtype=torch.long,
                device=device,
            )
        return (
            self._target_collaborative_items_device,
            self._target_history_matrix_device,
        )

    def _candidate_static_evidence(
        self, evidence_store, semantic_queries, rows, candidates, valid_mask, device
    ):
        static_items = evidence_store["item_ids"][rows]
        matches = candidates[:, :, None] == static_items[:, None, :]
        known = matches.any(axis=-1)
        positions = matches.argmax(axis=-1)
        semantic = np.take_along_axis(
            evidence_store["semantic_evidence"][rows], positions, axis=1
        )
        overlap = np.take_along_axis(
            evidence_store["overlap_evidence"][rows], positions, axis=1
        )

        raw = np.einsum(
            "bkd,bd->bk",
            self.target_collaborative_items[candidates],
            semantic_queries[rows],
        )
        raw_min = evidence_store["semantic_raw"][rows].min(axis=1)
        raw_max = evidence_store["semantic_raw"][rows].max(axis=1)
        arbitrary_semantic = np.clip(
            (raw - raw_min[:, None])
            / np.maximum(raw_max - raw_min, 1e-8)[:, None],
            0.0,
            1.0,
        )
        semantic = np.where(known, semantic, arbitrary_semantic).astype(np.float32)

        unknown = valid_mask & ~known
        if unknown.any():
            packed_width = int(unknown.sum(axis=1).max())
            packed = np.ones((len(rows), packed_width), dtype=np.int64)
            packed_positions = []
            for row in range(len(rows)):
                columns = np.flatnonzero(unknown[row])
                packed_positions.append(columns)
                packed[row, : len(columns)] = candidates[row, columns]
            target, histories = self._static_device_tensors(device)
            neighbor_ids = torch.as_tensor(
                evidence_store["neighbor_ids"][rows],
                dtype=torch.long,
                device=device,
            )
            neighbor_weights = torch.as_tensor(
                evidence_store["neighbor_weights"][rows],
                dtype=torch.float32,
                device=device,
            )
            neighbor_histories = histories[neighbor_ids]
            history_mask = neighbor_histories > 0
            history_embedding = target[neighbor_histories]
            candidate_embedding = target[
                torch.as_tensor(packed, dtype=torch.long, device=device)
            ]
            pairwise = torch.einsum(
                "bkd,bnhd->bknh", candidate_embedding, history_embedding
            )
            pairwise = pairwise.masked_fill(
                ~history_mask[:, None, :, :], -torch.inf
            )
            per_neighbor = pairwise.max(dim=-1).values
            per_neighbor = per_neighbor.masked_fill(
                ~history_mask.any(dim=-1)[:, None, :], 0.0
            )
            support = (
                (per_neighbor * neighbor_weights[:, None, :]).sum(dim=-1) + 1.0
            ) * 0.5
            support = support.clamp(0.0, 1.0).cpu().numpy()
            for row, columns in enumerate(packed_positions):
                if len(columns):
                    overlap[row, columns] = support[row, : len(columns)]
        semantic[~valid_mask] = 0.0
        overlap[~valid_mask] = 0.0
        return semantic, overlap.astype(np.float32)

    def _score_overlap_candidates(self, score_callback):
        static_width = min(
            max(1, self.calibration_width // 2),
            self.overlap_evidence["item_ids"].shape[1],
        )
        model_width = self.calibration_width - static_width
        users = self.overlap_evidence["user_ids"]
        output_items = np.empty((len(users), self.calibration_width), dtype=np.int64)
        output_features = np.empty(
            (len(users), self.calibration_width, 3), dtype=np.float32
        )
        for start in range(0, len(users), self.user_batch_size):
            stop = min(start + self.user_batch_size, len(users))
            user_tensor = torch.as_tensor(
                users[start:stop], dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                scores = score_callback(user_tensor)
                model_items = torch.topk(
                    scores[:, 1:], k=model_width, dim=1, largest=True, sorted=True
                ).indices + 1
            static_items = self.overlap_evidence["item_ids"][
                start:stop, :static_width
            ]
            items = np.concatenate(
                [static_items, model_items.cpu().numpy().astype(np.int64)], axis=1
            )
            valid_mask = np.ones(items.shape, dtype=bool)
            semantic, overlap = self._candidate_static_evidence(
                self.overlap_evidence,
                self.overlap_semantic_queries,
                np.arange(start, stop),
                items,
                valid_mask,
                scores.device,
            )
            item_tensor = torch.as_tensor(items, dtype=torch.long, device=scores.device)
            with torch.no_grad():
                model = self._model_evidence(scores, item_tensor).cpu().numpy()
            output_items[start:stop] = items
            output_features[start:stop] = np.stack(
                [semantic, overlap, model], axis=-1
            )
        labels = self._labels_for_candidates(users, output_items)
        return output_features, labels

    @staticmethod
    def _precision_threshold(scores, labels, minimum_precision, minimum_hits):
        order = np.argsort(-scores, kind="stable")
        ranked_labels = labels[order]
        cumulative_hits = np.cumsum(ranked_labels)
        counts = np.arange(1, len(order) + 1)
        precision = cumulative_hits / counts
        eligible = np.flatnonzero(
            (cumulative_hits >= minimum_hits) & (precision >= minimum_precision)
        )
        if len(eligible) == 0:
            return None
        return float(scores[order[int(eligible[-1])]])

    def _fit_dynamic_calibrator(self, features, labels, epoch_idx):
        train_rows = self.overlap_split_codes == 0
        valid_rows = self.overlap_split_codes == 1
        test_rows = self.overlap_split_codes == 2
        train_x = features[train_rows].reshape(-1, 3)
        valid_x = features[valid_rows].reshape(-1, 3)
        train_y = labels[train_rows].reshape(-1)
        valid_y = labels[valid_rows].reshape(-1)
        mean = train_x.mean(axis=0).astype(np.float32)
        std = np.maximum(train_x.std(axis=0), 1e-6).astype(np.float32)
        train_x = (train_x - mean) / std
        valid_x = (valid_x - mean) / std
        train_positive = float(train_y.sum())
        valid_positive = float(valid_y.sum())
        if train_positive <= 0 or valid_positive <= 0:
            raise RuntimeError("Dynamic calibration split contains no positive hit.")
        positive_weight = min(
            float(config_scalar(self.config.get("dynamic_positive_weight_cap"), 100.0)),
            max(1.0, (len(train_y) - train_positive) / train_positive),
        )
        correction = float(math.log(positive_weight))
        torch.manual_seed(self.seed + 4049 + int(epoch_idx))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + 4049 + int(epoch_idx))
        model = PositiveLinearCalibrator(3).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(
                config_scalar(self.config.get("dynamic_calibrator_learning_rate"), 0.01)
            ),
            weight_decay=float(
                config_scalar(self.config.get("dynamic_calibrator_weight_decay"), 0.0001)
            ),
        )
        batch_size = int(
            config_scalar(self.config.get("dynamic_calibrator_batch_size"), 65536)
        )
        generator = np.random.default_rng(self.seed + 5051 + int(epoch_idx))
        positive_weight_t = torch.tensor(positive_weight, device=self.device)
        valid_x_t = torch.as_tensor(valid_x, dtype=torch.float32, device=self.device)
        valid_y_t = torch.as_tensor(valid_y, dtype=torch.float32, device=self.device)
        best_loss = float("inf")
        best_state = None
        stale = 0
        patience = int(
            config_scalar(self.config.get("dynamic_calibrator_patience"), 8)
        )
        minimum_delta = float(
            config_scalar(self.config.get("dynamic_calibrator_minimum_delta"), 1e-6)
        )
        maximum_epochs = int(
            config_scalar(self.config.get("dynamic_calibrator_max_epochs"), 60)
        )
        for _calibration_epoch in range(maximum_epochs):
            model.train()
            order = generator.permutation(len(train_y))
            for start in range(0, len(train_y), batch_size):
                indices = order[start : start + batch_size]
                x = torch.as_tensor(
                    train_x[indices], dtype=torch.float32, device=self.device
                )
                y = torch.as_tensor(
                    train_y[indices], dtype=torch.float32, device=self.device
                )
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    model(x), y, pos_weight=positive_weight_t
                )
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(
                    F.binary_cross_entropy_with_logits(
                        model(valid_x_t), valid_y_t, pos_weight=positive_weight_t
                    ).cpu()
                )
            if valid_loss < best_loss - minimum_delta:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Dynamic calibrator did not produce a checkpoint.")
        model.load_state_dict(best_state)
        checkpoint = {
            "state_dict": best_state,
            "mean": mean,
            "std": std,
            "correction": correction,
            "weights": model.weights.detach().cpu().numpy(),
            "bias": float(model.bias.detach().cpu()),
        }
        confidence = self._predict_confidence(checkpoint, features)
        valid_scores = confidence[valid_rows].reshape(-1)
        valid_labels = labels[valid_rows].reshape(-1)
        quantile_threshold = float(
            np.quantile(valid_scores, self.calibration_quantile)
        )
        precision_threshold = self._precision_threshold(
            valid_scores,
            valid_labels,
            self.minimum_validation_precision,
            int(config_scalar(self.config.get("dynamic_min_validation_hits"), 3)),
        )
        threshold = max(self.minimum_confidence, quantile_threshold)
        if precision_threshold is not None:
            threshold = max(threshold, precision_threshold)
        valid_ap = average_precision(valid_scores, valid_labels)
        test_ap = average_precision(
            confidence[test_rows].reshape(-1), labels[test_rows].reshape(-1)
        )
        valid_brier = float(np.mean((valid_scores - valid_labels) ** 2))
        self.logger.info(
            "[V3 CALIBRATION] epoch=%d weights=%s bias=%.5f valid_AP=%.6f "
            "test_AP=%.6f valid_Brier=%.6f threshold=%.6f "
            "(quantile=%.6f precision=%s)",
            epoch_idx,
            np.round(checkpoint["weights"], 4).tolist(),
            checkpoint["bias"],
            valid_ap,
            test_ap,
            valid_brier,
            threshold,
            quantile_threshold,
            "none" if precision_threshold is None else "{:.6f}".format(precision_threshold),
        )
        return checkpoint, threshold

    def _predict_confidence(self, checkpoint, features):
        shape = features.shape[:-1]
        values = (features.reshape(-1, 3) - checkpoint["mean"]) / checkpoint["std"]
        model = PositiveLinearCalibrator(3).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        output = np.empty(len(values), dtype=np.float32)
        batch_size = int(
            config_scalar(
                self.config.get("dynamic_calibrator_inference_batch_size"), 262144
            )
        )
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                tensor = torch.as_tensor(
                    values[start:stop], dtype=torch.float32, device=self.device
                )
                output[start:stop] = torch.sigmoid(
                    model(tensor) - checkpoint["correction"]
                ).cpu().numpy()
        return output.reshape(shape)

    def refresh(self, score_callback, epoch_idx):
        overlap_features, overlap_labels = self._score_overlap_candidates(score_callback)
        checkpoint, calibration_threshold = self._fit_dynamic_calibrator(
            overlap_features, overlap_labels, epoch_idx
        )
        updated_confidences = [None] * len(self.user_ids)
        residual_confidence = np.empty_like(
            self.residual_base_confidence, dtype=np.float32
        )
        proposals = []
        proposed_total = 0

        for start in range(0, len(self.user_ids), self.user_batch_size):
            stop = min(start + self.user_batch_size, len(self.user_ids))
            users = self.user_ids[start:stop]
            user_tensor = torch.as_tensor(users, dtype=torch.long, device=self.device)
            with torch.no_grad():
                full_scores = score_callback(user_tensor)
            proposal_scores = full_scores.clone()
            proposal_scores[:, 0] = -torch.inf
            for local_row, user in enumerate(users.tolist()):
                forbidden = set(self.current_items[start + local_row])
                forbidden.update(self.all_target_positive[int(user)])
                if forbidden:
                    indices = torch.as_tensor(
                        sorted(forbidden), dtype=torch.long, device=full_scores.device
                    )
                    proposal_scores[local_row, indices] = -torch.inf
            proposal_items = torch.topk(
                proposal_scores[:, 1:],
                k=self.expansion_top_k,
                dim=1,
                largest=True,
                sorted=True,
            ).indices + 1
            proposal_items = proposal_items.cpu().numpy().astype(np.int64)
            proposed_total += proposal_items.size

            candidate_lists = []
            proposal_positions = []
            residual_positions = []
            for local_row in range(stop - start):
                values = list(self.current_items[start + local_row])
                proposal_positions.append(
                    np.arange(len(values), len(values) + self.expansion_top_k)
                )
                values.extend(proposal_items[local_row].tolist())
                residual_positions.append(
                    np.arange(
                        len(values),
                        len(values) + self.residual_base_items.shape[1],
                    )
                )
                values.extend(self.residual_base_items[start + local_row].tolist())
                candidate_lists.append(values)

            width = max(map(len, candidate_lists))
            candidates = np.ones((stop - start, width), dtype=np.int64)
            valid_mask = np.zeros((stop - start, width), dtype=bool)
            for row, values in enumerate(candidate_lists):
                candidates[row, : len(values)] = values
                valid_mask[row, : len(values)] = True
            candidate_tensor = torch.as_tensor(
                candidates, dtype=torch.long, device=full_scores.device
            )
            with torch.no_grad():
                model_evidence = self._model_evidence(
                    full_scores, candidate_tensor
                ).cpu().numpy()
            semantic, overlap = self._candidate_static_evidence(
                self.cold_evidence,
                self.cold_semantic_queries,
                np.arange(start, stop),
                candidates,
                valid_mask,
                full_scores.device,
            )
            confidence = self._predict_confidence(
                checkpoint, np.stack([semantic, overlap, model_evidence], axis=-1)
            )

            for local_row in range(stop - start):
                global_row = start + local_row
                existing_count = len(self.current_items[global_row])
                dynamic_current = confidence[local_row, :existing_count]
                previous_current = np.asarray(
                    self.current_confidences[global_row], dtype=np.float32
                )
                dynamic_anchor_median = float(
                    np.median(dynamic_current[: self.official_top_k])
                )
                previous_anchor_median = float(
                    np.median(previous_current[: self.official_top_k])
                )
                confidence_scale = previous_anchor_median / max(
                    dynamic_anchor_median, self.confidence_epsilon
                )
                mapped_current = np.clip(
                    dynamic_current * confidence_scale,
                    self.confidence_epsilon,
                    1.0,
                )
                blend = self.setting["anchor_reweight_blend"]
                current_confidence = (
                    (1.0 - blend) * previous_current + blend * mapped_current
                ).astype(np.float32)
                updated_confidences[global_row] = current_confidence.tolist()

                dynamic_residual = confidence[
                    local_row, residual_positions[local_row]
                ]
                mapped_residual = np.clip(
                    dynamic_residual * confidence_scale,
                    self.confidence_epsilon,
                    1.0,
                )
                residual_confidence[global_row] = (
                    (1.0 - blend) * self.residual_dynamic_confidence[global_row]
                    + blend * mapped_residual
                )
                anchor_floor = float(
                    np.quantile(
                        dynamic_current[: self.official_top_k], self.anchor_quantile
                    )
                )
                minimum_anchor = (
                    self.setting["anchor_confidence_ratio"] * anchor_floor
                )
                for offset, item in enumerate(proposal_items[local_row].tolist()):
                    item_confidence = float(
                        confidence[local_row, proposal_positions[local_row][offset]]
                    )
                    mapped_item_confidence = float(
                        np.clip(
                            item_confidence * confidence_scale,
                            self.confidence_epsilon,
                            1.0,
                        )
                    )
                    if (
                        item_confidence >= calibration_threshold
                        and item_confidence >= minimum_anchor
                        and self.user_gate[int(users[local_row])]
                        >= self.setting["minimum_user_gate"]
                        and item not in self.current_items[global_row]
                    ):
                        proposals.append(
                            (
                                item_confidence,
                                mapped_item_confidence,
                                global_row,
                                int(item),
                            )
                        )

        selected = []
        if proposals:
            proposal_confidence = np.asarray(
                [entry[0] for entry in proposals], dtype=np.float32
            )
            proposal_threshold = float(
                np.quantile(proposal_confidence, self.setting["proposal_quantile"])
            )
            qualified = [entry for entry in proposals if entry[0] >= proposal_threshold]
            qualified.sort(key=lambda entry: (-entry[0], entry[2], entry[3]))
            quota = max(
                1,
                int(
                    math.ceil(
                        len(self.user_ids)
                        * self.setting["maximum_acceptance_fraction"]
                    )
                ),
            )
            selected = qualified[:quota]

        accepted_total = 0
        accepted_users = set()
        for _raw_confidence, item_confidence, global_row, item in selected:
            if len(self.current_items[global_row]) >= self.maximum_items:
                continue
            if item in self.current_items[global_row]:
                continue
            self.current_items[global_row].append(item)
            updated_confidences[global_row].append(item_confidence)
            accepted_total += 1
            accepted_users.add(global_row)

        self.current_confidences = updated_confidences
        self.residual_dynamic_confidence = residual_confidence
        self._rebuild_sampling_arrays()
        self.refresh_count += 1
        support_size = sum(map(len, self.current_items))
        self.logger.info(
            "[V3 EXPANSION] epoch=%d refresh=%d proposed=%d eligible=%d "
            "accepted=%d users=%d accept_rate=%.6f support_pairs=%d mean_items=%.3f",
            epoch_idx,
            self.refresh_count,
            proposed_total,
            len(proposals),
            accepted_total,
            len(accepted_users),
            accepted_total / max(proposed_total, 1),
            support_size,
            support_size / max(len(self.current_items), 1),
        )
        return {
            "epoch": int(epoch_idx),
            "proposed": int(proposed_total),
            "eligible": int(len(proposals)),
            "accepted": int(accepted_total),
            "support_pairs": int(support_size),
            "threshold": float(calibration_threshold),
        }

    def _rebuild_sampling_arrays(self):
        pair_users = []
        pair_items = []
        pair_confidence = []
        self.pseudo_positive = {}
        for user, items, confidence in zip(
            self.user_ids.tolist(), self.current_items, self.current_confidences
        ):
            pair_users.extend([int(user)] * len(items))
            pair_items.extend(items)
            pair_confidence.extend(confidence)
            self.pseudo_positive[int(user)] = set(items)
        self.pair_users = np.asarray(pair_users, dtype=np.int64)
        self.pair_items = np.asarray(pair_items, dtype=np.int64)
        self.pair_confidence = np.asarray(pair_confidence, dtype=np.float32)
        self.pair_weights = np.power(
            np.maximum(self.pair_confidence, self.confidence_epsilon),
            self.confidence_power,
        ).astype(np.float32)

    def sync_adaptive_residual(self, supervisor):
        top_k = supervisor.top_k
        selected_items = np.empty((len(self.user_ids), top_k), dtype=np.int64)
        selected_confidence = np.empty(
            (len(self.user_ids), top_k), dtype=np.float32
        )
        for row in range(len(self.user_ids)):
            item_to_confidence = {
                int(item): float(confidence)
                for item, confidence in zip(
                    self.residual_base_items[row],
                    self.residual_dynamic_confidence[row],
                )
            }
            for item, confidence in zip(
                self.current_items[row][self.official_top_k :],
                self.current_confidences[row][self.official_top_k :],
            ):
                item_to_confidence[int(item)] = max(
                    float(confidence), item_to_confidence.get(int(item), 0.0)
                )
            ranked = sorted(
                item_to_confidence.items(), key=lambda pair: (-pair[1], pair[0])
            )[:top_k]
            if len(ranked) < top_k:
                raise RuntimeError("V3 residual support is smaller than residual_top_k.")
            selected_items[row] = [item for item, _confidence in ranked]
            selected_confidence[row] = [confidence for _item, confidence in ranked]

        quality = selected_confidence[:, 0]
        retained_count = max(
            1, int(np.ceil(len(self.user_ids) * supervisor.retained_user_fraction))
        )
        user_order = np.lexsort((self.user_ids, -quality))
        retained_rows = user_order[:retained_count]
        retained_users = self.user_ids[retained_rows]
        retained_items = selected_items[retained_rows]
        retained_confidence = selected_confidence[retained_rows]
        supervisor.user_ids = retained_users
        supervisor.item_ids = retained_items
        supervisor.pair_users = np.repeat(retained_users, top_k)
        supervisor.pair_items = retained_items.reshape(-1)
        supervisor.pair_confidence = retained_confidence.reshape(-1)
        supervisor.pair_weights = (
            supervisor.pair_confidence * supervisor.loss_confidence_scale
        ).astype(np.float32, copy=False)
        sampling_weights = np.power(
            np.maximum(supervisor.pair_confidence, supervisor.confidence_epsilon),
            supervisor.sampling_power,
        ).astype(np.float64, copy=False)
        supervisor.pair_probabilities = sampling_weights / sampling_weights.sum()
        supervisor.pseudo_positive = {
            int(user): set(map(int, items))
            for user, items in zip(retained_users.tolist(), retained_items.tolist())
        }
        supervisor.user_gate = np.zeros(
            supervisor.num_users_src + 1, dtype=np.float32
        )
        maximum = float(retained_confidence[:, 0].max())
        if maximum > 0:
            supervisor.user_gate[retained_users] = retained_confidence[:, 0] / maximum

    def adaptive_gate(self, temperature, floor):
        quality = np.maximum(
            self.residual_dynamic_confidence[:, 0],
            np.asarray(
                [max(values) for values in self.current_confidences],
                dtype=np.float32,
            ),
        )
        order = np.lexsort((self.user_ids, quality))
        percentiles = np.empty(len(self.user_ids), dtype=np.float32)
        if len(self.user_ids) == 1:
            percentiles[0] = 1.0
        else:
            percentiles[order] = np.linspace(
                0.0, 1.0, len(self.user_ids), dtype=np.float32
            )
        retained_fraction = float(
            config_scalar(self.config.get("residual_retained_user_fraction"), 0.05)
        )
        threshold = 1.0 - retained_fraction
        soft_gate = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    -(percentiles - threshold) / float(temperature), -60.0, 60.0
                )
            )
        )
        if floor > 0:
            soft_gate = floor + (1.0 - floor) * soft_gate
        gate = np.zeros(self.num_users_src + 1, dtype=np.float32)
        gate[self.user_ids] = soft_gate.astype(np.float32, copy=False)
        return gate


class GroundedExpandablePseudoMixin:
    """Add EMA-calibrated expansion to an adaptive stage-v2.1 model."""

    def _init_grounded_expandable(self, config, dataloader):
        self.config = config
        self.pseudo_supervisor = ExpandableGroundedPseudoSupervisor(
            config, dataloader
        )
        self.pseudo_supervisor.set_user_gate(
            self.adaptive_user_gate.detach().cpu().numpy()
        )
        object.__setattr__(self, "_dynamic_dataloader", dataloader)
        object.__setattr__(self, "_ema_teacher", None)
        self._ema_decay = float(
            config_scalar(config.get("ema_teacher_decay"), 0.995)
        )
        self._expansion_start_epoch = int(
            config_scalar(config.get("expansion_start_epoch"), 10)
        )
        self._expansion_interval = int(
            config_scalar(config.get("expansion_interval"), 10)
        )
        self._expansion_max_rounds = int(
            config_scalar(config.get("expansion_max_rounds"), 3)
        )
        if not 0.0 <= self._ema_decay < 1.0:
            raise ValueError("ema_teacher_decay must lie in [0, 1).")
        if self._expansion_interval <= 0:
            raise ValueError("expansion_interval must be positive.")
        if self._expansion_max_rounds <= 0:
            raise ValueError("expansion_max_rounds must be positive.")

    def _dynamic_stage_active(self):
        return self.stage_id == self.adaptive_stage_id

    def _new_dynamic_teacher(self):
        raise NotImplementedError

    @staticmethod
    def _save_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _ensure_ema_teacher(self):
        if self._ema_teacher is not None:
            return
        state = self._save_rng_state()
        try:
            teacher = self._new_dynamic_teacher()
            device = next(self.parameters()).device
            teacher = teacher.to(device)
            teacher.load_state_dict(self.state_dict(), strict=True)
            teacher.set_train_stage(self.stage_id)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False
            object.__setattr__(self, "_ema_teacher", teacher)
        finally:
            self._restore_rng_state(state)
        logging.getLogger().info(
            "[V3 EMA] initialized from adaptive stage-v2.1 decay=%.6f",
            self._ema_decay,
        )

    def _update_ema_teacher(self):
        teacher_parameters = dict(self._ema_teacher.named_parameters())
        teacher_buffers = dict(self._ema_teacher.named_buffers())
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                teacher_parameters[name].mul_(self._ema_decay).add_(
                    parameter.detach(), alpha=1.0 - self._ema_decay
                )
            for name, buffer in self.named_buffers():
                if name in teacher_buffers:
                    teacher_buffers[name].copy_(buffer.detach())
        self._ema_teacher.eval()

    def _prepare_dynamic_teacher(self):
        pass

    def _clear_dynamic_teacher(self):
        pass

    def _dynamic_teacher_scores(self, users):
        scores = self._ema_teacher.full_sort_predict((users,), False)
        scores[:, 0] = -torch.inf
        return scores

    def pre_epoch_processing(self, epoch_idx):
        parent = super().pre_epoch_processing(epoch_idx)
        if not self._dynamic_stage_active():
            return parent
        self._ensure_ema_teacher()
        if self.pseudo_supervisor.refresh_count >= self._expansion_max_rounds:
            return parent
        if epoch_idx < self._expansion_start_epoch:
            return parent
        if (epoch_idx - self._expansion_start_epoch) % self._expansion_interval:
            return parent

        state = self._save_rng_state()
        self._ema_teacher.eval()
        self._prepare_dynamic_teacher()
        try:
            self.pseudo_supervisor.refresh(self._dynamic_teacher_scores, epoch_idx)
            self.pseudo_supervisor.sync_adaptive_residual(
                self.adaptive_residual_supervisor
            )
            gate = self.pseudo_supervisor.adaptive_gate(
                self.adaptive_gate_temperature, self.adaptive_gate_floor
            )
            self.adaptive_user_gate.copy_(
                torch.as_tensor(
                    gate,
                    dtype=self.adaptive_user_gate.dtype,
                    device=self.adaptive_user_gate.device,
                )
            )
            self.pseudo_supervisor.set_user_gate(gate)
        finally:
            self._clear_dynamic_teacher()
            self._restore_rng_state(state)
        return parent

    def post_batch_processing(self, epoch_idx, batch_idx):
        parent = super().post_batch_processing(epoch_idx, batch_idx)
        if self._dynamic_stage_active():
            self._ensure_ema_teacher()
            self._update_ema_teacher()
        return parent
