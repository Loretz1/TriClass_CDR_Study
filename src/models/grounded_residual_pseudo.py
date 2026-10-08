import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grounded_confidence_pseudo import (
    GroundedConfidencePseudoSupervisor,
)
from models.target_pattern_pseudo import config_scalar


class GroundedResidualPseudoSupervisor(GroundedConfidencePseudoSupervisor):
    """Select and sample a small, high-confidence pseudo-interaction set."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.logger = logging.getLogger()
        self.candidate_pool_size = int(
            config_scalar(config.get("residual_candidate_pool_size"), 100)
        )
        self.retained_user_fraction = float(
            config_scalar(config.get("residual_retained_user_fraction"), 0.05)
        )
        self.top_k = int(config_scalar(config.get("residual_top_k"), 1))
        self.sampling_power = float(
            config_scalar(config.get("residual_sampling_power"), 1.0)
        )
        self.loss_confidence_scale = float(
            config_scalar(config.get("residual_loss_confidence_scale"), 1.0)
        )

        if self.candidate_pool_size <= 0:
            raise ValueError("residual_candidate_pool_size must be positive.")
        if not 0 < self.retained_user_fraction <= 1:
            raise ValueError(
                "residual_retained_user_fraction must lie in (0, 1]."
            )
        if self.top_k <= 0:
            raise ValueError("residual_top_k must be positive.")
        if self.sampling_power < 0:
            raise ValueError("residual_sampling_power must be non-negative.")
        if self.loss_confidence_scale <= 0:
            raise ValueError("residual_loss_confidence_scale must be positive.")

        self._build_residual_candidates()

    def _build_residual_candidates(self):
        with np.load(self.confidence_cache_path, allow_pickle=False) as cache:
            user_ids = np.asarray(cache["user_ids"], dtype=np.int64)
            all_item_ids = np.asarray(cache["item_ids"], dtype=np.int64)
            evidence = np.asarray(
                cache[self.SOURCES[self.confidence_source]], dtype=np.float32
            )

            if (
                self.confidence_source == "combined"
                and "confidence_ranked_item_ids" in cache.files
                and "confidence_ranked_scores" in cache.files
            ):
                ranked_item_ids = np.asarray(
                    cache["confidence_ranked_item_ids"], dtype=np.int64
                )
                ranked_confidence = np.asarray(
                    cache["confidence_ranked_scores"], dtype=np.float32
                )
            else:
                order = np.argsort(-evidence, axis=1, kind="stable")
                ranked_item_ids = np.take_along_axis(
                    all_item_ids, order, axis=1
                )
                ranked_confidence = np.take_along_axis(
                    evidence, order, axis=1
                )

        if not np.array_equal(user_ids, self.user_ids):
            raise ValueError(
                "Residual confidence users are not aligned with the pseudo cache."
            )
        if ranked_item_ids.shape != ranked_confidence.shape:
            raise ValueError(
                "Residual ranked items and confidence must have the same shape."
            )

        pool_size = min(self.candidate_pool_size, ranked_item_ids.shape[1])
        if self.top_k > pool_size:
            raise ValueError(
                "residual_top_k cannot exceed residual_candidate_pool_size."
            )
        ranked_item_ids = ranked_item_ids[:, :pool_size]
        ranked_confidence = ranked_confidence[:, :pool_size]
        if not np.all(np.isfinite(ranked_confidence)):
            raise ValueError("Residual confidence contains a non-finite value.")
        if np.any(ranked_confidence < 0) or np.any(ranked_confidence > 1):
            raise ValueError("Residual confidence values must lie in [0, 1].")

        user_confidence = ranked_confidence[:, 0]
        retained_count = max(
            1,
            int(np.ceil(len(user_ids) * self.retained_user_fraction)),
        )
        user_order = np.lexsort((user_ids, -user_confidence))
        retained_rows = user_order[:retained_count]

        retained_users = user_ids[retained_rows]
        retained_items = ranked_item_ids[retained_rows, : self.top_k]
        retained_confidence = ranked_confidence[
            retained_rows, : self.top_k
        ]
        if np.any(retained_items <= 0) or np.any(
            retained_items > self.num_items_tgt
        ):
            raise ValueError("Residual pseudo cache contains an invalid item id.")

        self.user_ids = retained_users
        self.item_ids = retained_items
        self.pair_users = np.repeat(retained_users, self.top_k)
        self.pair_items = retained_items.reshape(-1)
        self.pair_confidence = retained_confidence.reshape(-1)
        self.pair_weights = (
            self.pair_confidence * self.loss_confidence_scale
        ).astype(np.float32, copy=False)

        sampling_weights = np.power(
            np.maximum(self.pair_confidence, self.confidence_epsilon),
            self.sampling_power,
        ).astype(np.float64, copy=False)
        self.pair_probabilities = sampling_weights / sampling_weights.sum()
        self.pseudo_positive = {
            int(user): set(map(int, items))
            for user, items in zip(
                retained_users.tolist(), retained_items.tolist()
            )
        }

        self.user_gate = np.zeros(self.num_users_src + 1, dtype=np.float32)
        maximum = float(retained_confidence[:, 0].max())
        if maximum > 0:
            self.user_gate[retained_users] = retained_confidence[:, 0] / maximum

        self.logger.info(
            "[RESIDUAL PSEUDO] source=%s pool=%d retained=%d/%d (%.2f%%) "
            "Top-%d confidence=[%.6g, %.6g] mean=%.6g",
            self.confidence_source,
            pool_size,
            retained_count,
            len(user_ids),
            100.0 * retained_count / len(user_ids),
            self.top_k,
            float(retained_confidence.min()),
            float(retained_confidence.max()),
            float(retained_confidence.mean()),
        )

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("residual_num_per_batch must be positive.")

        pair_index = self.rng.choice(
            len(self.pair_users),
            size=batch_size,
            replace=True,
            p=self.pair_probabilities,
        )
        users = self.pair_users[pair_index]
        positives = self.pair_items[pair_index]
        weights = self.pair_weights[pair_index]
        negatives = np.empty(batch_size, dtype=np.int64)

        for row, user in enumerate(users.tolist()):
            forbidden = self.pseudo_positive[int(user)] | self.target_positive[
                int(user)
            ]
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


class GroundedResidualPseudoMixin:
    """Train a frozen-backbone residual ranker from grounded pseudo pairs."""

    def _init_grounded_residual_pseudo(self, config, dataloader, embedding_dim):
        self.config = config
        self.residual_supervisor = GroundedResidualPseudoSupervisor(
            config, dataloader
        )
        self.residual_embedding_dim = int(embedding_dim)
        self.residual_stage_id = int(
            config_scalar(
                config.get("residual_stage_id"),
                len(config["training_stages"]) - 1,
            )
        )
        self.residual_beta = float(
            config_scalar(config.get("residual_beta"), 0.1)
        )
        self.residual_l2 = float(
            config_scalar(config.get("residual_l2"), 1.0e-4)
        )
        self.residual_num_per_batch = int(
            config_scalar(config.get("residual_num_per_batch"), 512)
        )
        self.residual_score_scale_clip = float(
            config_scalar(config.get("residual_score_scale_clip"), 20.0)
        )
        if self.residual_beta < 0:
            raise ValueError("residual_beta must be non-negative.")
        if self.residual_l2 < 0:
            raise ValueError("residual_l2 must be non-negative.")
        if self.residual_num_per_batch <= 0:
            raise ValueError("residual_num_per_batch must be positive.")
        if self.residual_score_scale_clip <= 0:
            raise ValueError("residual_score_scale_clip must be positive.")

        initial_user_table = torch.zeros(
            self.num_users_src + 1, self.residual_embedding_dim
        )
        self.residual_user_embedding = nn.Embedding.from_pretrained(
            initial_user_table,
            freeze=False,
            padding_idx=0,
        )
        self.register_buffer(
            "residual_target_item_table",
            torch.zeros(
                self.num_items_tgt + 1, self.residual_embedding_dim
            ),
            persistent=False,
        )
        self.register_buffer(
            "residual_user_gate",
            torch.from_numpy(self.residual_supervisor.user_gate.copy()),
            persistent=False,
        )
        self._residual_active = False

    def _residual_target_item_embeddings(self):
        raise NotImplementedError

    def _refresh_residual_item_table(self):
        was_training = self.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        self.eval()
        with torch.no_grad():
            item_table = self._residual_target_item_embeddings().detach()
            if item_table.shape != self.residual_target_item_table.shape:
                raise ValueError(
                    "Residual target item table has shape {}, expected {}.".format(
                        tuple(item_table.shape),
                        tuple(self.residual_target_item_table.shape),
                    )
                )
            item_table = F.normalize(item_table, dim=-1)
            item_table[0].zero_()
            self.residual_target_item_table.copy_(item_table)

        self.train(was_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    def set_train_stage(self, stage_id):
        super().set_train_stage(stage_id)
        self._residual_active = stage_id == self.residual_stage_id
        if not self._residual_active:
            self.residual_user_embedding.weight.requires_grad = False
            return

        for parameter in self.parameters():
            parameter.requires_grad = False
        self.residual_user_embedding.weight.requires_grad = True
        self._refresh_residual_item_table()
        logging.getLogger().info(
            "[RESIDUAL PSEUDO] Activated stage=%d beta=%.4g l2=%.4g "
            "batch=%d trainable_users=%d",
            stage_id,
            self.residual_beta,
            self.residual_l2,
            self.residual_num_per_batch,
            len(self.residual_supervisor.user_ids),
        )

    def _residual_loss(self):
        users, positive_items, negative_items, confidence = (
            self.residual_supervisor.sample(self.residual_num_per_batch)
        )
        user_embedding = self.residual_user_embedding(users)
        positive_embedding = self.residual_target_item_table[positive_items]
        negative_embedding = self.residual_target_item_table[negative_items]
        positive_score = (user_embedding * positive_embedding).sum(dim=-1)
        negative_score = (user_embedding * negative_embedding).sum(dim=-1)
        pair_loss = -F.logsigmoid(positive_score - negative_score)
        confidence = confidence.to(pair_loss.dtype)

        loss = (confidence * pair_loss).mean()
        if self.residual_l2 > 0:
            loss = loss + self.residual_l2 * user_embedding.square().sum(
                dim=-1
            ).mean()
        return loss

    def calculate_loss(self, interaction, epoch_idx):
        if self._residual_active:
            return self._residual_loss()
        return super().calculate_loss(interaction, epoch_idx)

    def full_sort_predict(self, interaction, is_warm):
        base_scores = super().full_sort_predict(interaction, is_warm)
        if not self._residual_active or is_warm or self.residual_beta == 0:
            return base_scores

        users = interaction[0].long()
        user_embedding = self.residual_user_embedding(users)
        residual_scores = torch.matmul(
            user_embedding, self.residual_target_item_table.T
        )

        base_scale = base_scores[:, 1:].std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-8)
        residual_scale = residual_scores[:, 1:].std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-8)
        scale = (base_scale / residual_scale).clamp(
            max=self.residual_score_scale_clip
        )
        gate = self.residual_user_gate[users].unsqueeze(1)
        scores = base_scores + (
            self.residual_beta * gate * scale * residual_scores
        )
        scores[:, 0] = base_scores[:, 0]
        return scores
