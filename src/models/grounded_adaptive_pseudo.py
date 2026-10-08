import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grounded_residual_pseudo import GroundedResidualPseudoSupervisor
from models.target_pattern_pseudo import config_scalar


ADAPTIVE_SETTINGS = {
    "j1_b0": (1.0, 0.0),
    "j1_b0005": (1.0, 0.005),
    "j1_b001": (1.0, 0.01),
    "j1_b003": (1.0, 0.03),
    "j05_b0005": (0.5, 0.005),
    "j05_b001": (0.5, 0.01),
    "j05_b003": (0.5, 0.03),
    "j025_b001": (0.25, 0.01),
    "j025_b003": (0.25, 0.03),
    "j0_b001": (0.0, 0.01),
    "j0_b003": (0.0, 0.03),
    "j0_b005": (0.0, 0.05),
}


class GroundedAdaptivePseudoMixin:
    """Jointly tune direct pseudo training and a user-conditioned residual."""

    def _init_grounded_adaptive_pseudo(self, config, dataloader, embedding_dim):
        self.config = config
        setting = str(config_scalar(config.get("adaptive_setting"), "j1_b0"))
        if setting not in ADAPTIVE_SETTINGS:
            raise ValueError(
                "Unknown adaptive_setting {!r}. Available settings: {}.".format(
                    setting, ", ".join(ADAPTIVE_SETTINGS)
                )
            )
        self.adaptive_setting = setting
        (
            self.adaptive_joint_pseudo_scale,
            self.adaptive_residual_beta,
        ) = ADAPTIVE_SETTINGS[setting]

        self.adaptive_embedding_dim = int(embedding_dim)
        self.adaptive_stage_id = int(
            config_scalar(
                config.get("adaptive_stage_id"),
                len(config["training_stages"]) - 1,
            )
        )
        self.adaptive_residual_loss_weight = float(
            config_scalar(config.get("adaptive_residual_loss_weight"), 1.0)
        )
        self.adaptive_residual_l2 = float(
            config_scalar(config.get("adaptive_residual_l2"), 1.0e-4)
        )
        self.adaptive_residual_num_per_batch = int(
            config_scalar(config.get("residual_num_per_batch"), 512)
        )
        self.adaptive_gate_temperature = float(
            config_scalar(config.get("adaptive_gate_temperature"), 0.025)
        )
        self.adaptive_gate_floor = float(
            config_scalar(config.get("adaptive_gate_floor"), 0.0)
        )
        self.adaptive_score_scale_clip = float(
            config_scalar(config.get("residual_score_scale_clip"), 20.0)
        )
        if self.adaptive_residual_loss_weight < 0:
            raise ValueError("adaptive_residual_loss_weight must be non-negative.")
        if self.adaptive_residual_l2 < 0:
            raise ValueError("adaptive_residual_l2 must be non-negative.")
        if self.adaptive_residual_num_per_batch <= 0:
            raise ValueError("residual_num_per_batch must be positive.")
        if self.adaptive_gate_temperature <= 0:
            raise ValueError("adaptive_gate_temperature must be positive.")
        if not 0 <= self.adaptive_gate_floor <= 1:
            raise ValueError("adaptive_gate_floor must lie in [0, 1].")
        if self.adaptive_score_scale_clip <= 0:
            raise ValueError("residual_score_scale_clip must be positive.")

        self.adaptive_residual_supervisor = GroundedResidualPseudoSupervisor(
            config, dataloader
        )
        self.adaptive_residual_adapter = self._build_adaptive_adapter(config)
        self.register_buffer(
            "adaptive_user_gate",
            self._build_adaptive_user_gate(),
            persistent=False,
        )
        self._adaptive_residual_active = False

        logging.getLogger().info(
            "[ADAPTIVE PSEUDO] setting=%s joint_scale=%.4g beta=%.4g "
            "stage=%d residual_batch=%d gate_temperature=%.4g",
            self.adaptive_setting,
            self.adaptive_joint_pseudo_scale,
            self.adaptive_residual_beta,
            self.adaptive_stage_id,
            self.adaptive_residual_num_per_batch,
            self.adaptive_gate_temperature,
        )

    def _build_adaptive_adapter(self, config):
        hidden_dim = int(
            config_scalar(
                config.get("adaptive_hidden_dim"),
                max(16, self.adaptive_embedding_dim // 2),
            )
        )
        if hidden_dim <= 0:
            raise ValueError("adaptive_hidden_dim must be positive.")

        # Keep the backbone random stream identical to the direct-pseudo model.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()
        adapter = nn.Sequential(
            nn.LayerNorm(self.adaptive_embedding_dim),
            nn.Linear(self.adaptive_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.adaptive_embedding_dim),
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        return adapter

    def _build_adaptive_user_gate(self):
        supervisor = self.adaptive_residual_supervisor
        with np.load(supervisor.confidence_cache_path, allow_pickle=False) as cache:
            user_ids = np.asarray(cache["user_ids"], dtype=np.int64)
            evidence = np.asarray(
                cache[supervisor.SOURCES[supervisor.confidence_source]],
                dtype=np.float32,
            )
            if (
                supervisor.confidence_source == "combined"
                and "confidence_ranked_scores" in cache.files
            ):
                quality = np.asarray(
                    cache["confidence_ranked_scores"][:, 0], dtype=np.float32
                )
            else:
                pool_size = min(supervisor.candidate_pool_size, evidence.shape[1])
                quality = evidence[:, :pool_size].max(axis=1)

        if len(user_ids) == 0:
            raise ValueError("Cannot build an adaptive gate without cold users.")
        if not np.all(np.isfinite(quality)):
            raise ValueError("Adaptive user confidence contains a non-finite value.")

        order = np.lexsort((user_ids, quality))
        percentiles = np.empty(len(user_ids), dtype=np.float32)
        if len(user_ids) == 1:
            percentiles[0] = 1.0
        else:
            percentiles[order] = np.linspace(
                0.0, 1.0, len(user_ids), dtype=np.float32
            )
        threshold = 1.0 - supervisor.retained_user_fraction
        soft_gate = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    -(percentiles - threshold) / self.adaptive_gate_temperature,
                    -60.0,
                    60.0,
                )
            )
        )
        if self.adaptive_gate_floor > 0:
            soft_gate = self.adaptive_gate_floor + (
                1.0 - self.adaptive_gate_floor
            ) * soft_gate

        gate = np.zeros(supervisor.num_users_src + 1, dtype=np.float32)
        gate[user_ids] = soft_gate.astype(np.float32, copy=False)
        return torch.from_numpy(gate)

    def _pseudo_weight(self):
        return self.adaptive_joint_pseudo_scale * super()._pseudo_weight()

    def _adaptive_backbone_loss(self, interaction, epoch_idx):
        raise NotImplementedError

    def _adaptive_source_user_representation(self, users):
        raise NotImplementedError

    def _adaptive_target_item_embeddings(self, item_ids=None):
        raise NotImplementedError

    def set_train_stage(self, stage_id):
        super().set_train_stage(stage_id)
        self._adaptive_residual_active = (
            stage_id == self.adaptive_stage_id
            and self.adaptive_residual_beta > 0
        )
        for parameter in self.adaptive_residual_adapter.parameters():
            parameter.requires_grad = self._adaptive_residual_active

    def _adaptive_residual_loss(self):
        users, positive_items, negative_items, confidence = (
            self.adaptive_residual_supervisor.sample(
                self.adaptive_residual_num_per_batch
            )
        )
        user_embedding = self._adaptive_source_user_representation(users).detach()
        positive_embedding = self._adaptive_target_item_embeddings(
            positive_items
        ).detach()
        negative_embedding = self._adaptive_target_item_embeddings(
            negative_items
        ).detach()
        positive_embedding = F.normalize(positive_embedding, dim=-1)
        negative_embedding = F.normalize(negative_embedding, dim=-1)

        residual_user = self.adaptive_residual_adapter(user_embedding)
        positive_score = (residual_user * positive_embedding).sum(dim=-1)
        negative_score = (residual_user * negative_embedding).sum(dim=-1)
        pair_loss = -F.logsigmoid(positive_score - negative_score)
        confidence = confidence.to(pair_loss.dtype)
        denominator = confidence.sum().clamp_min(
            torch.finfo(pair_loss.dtype).eps
        )
        loss = (confidence * pair_loss).sum() / denominator
        if self.adaptive_residual_l2 > 0:
            loss = loss + self.adaptive_residual_l2 * residual_user.square().sum(
                dim=-1
            ).mean()
        return self.adaptive_residual_loss_weight * loss

    def calculate_loss(self, interaction, epoch_idx):
        if self.adaptive_joint_pseudo_scale == 0:
            loss = self._adaptive_backbone_loss(interaction, epoch_idx)
        else:
            loss = super().calculate_loss(interaction, epoch_idx)

        if not self._adaptive_residual_active:
            return loss
        if epoch_idx < int(config_scalar(self.config.get("pseudo_start_epoch"), 0)):
            return loss
        return loss + self._adaptive_residual_loss()

    def full_sort_predict(self, interaction, is_warm):
        base_scores = super().full_sort_predict(interaction, is_warm)
        if (
            not self._adaptive_residual_active
            or is_warm
            or self.adaptive_residual_beta == 0
        ):
            return base_scores

        users = interaction[0].long()
        user_embedding = self._adaptive_source_user_representation(users)
        item_embedding = self._adaptive_target_item_embeddings().detach()
        item_embedding = F.normalize(item_embedding, dim=-1)
        residual_user = self.adaptive_residual_adapter(user_embedding)
        residual_scores = torch.matmul(residual_user, item_embedding.T)

        base_scale = base_scores[:, 1:].std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-8)
        residual_scale = residual_scores[:, 1:].std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-8)
        score_scale = (base_scale / residual_scale).clamp(
            max=self.adaptive_score_scale_clip
        )
        gate = self.adaptive_user_gate[users].unsqueeze(1)
        scores = base_scores + (
            self.adaptive_residual_beta
            * gate
            * score_scale
            * residual_scores
        )
        scores[:, 0] = base_scores[:, 0]
        return scores
