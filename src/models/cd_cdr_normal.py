import numpy as np
import torch
from torch import nn

from models.cd_cdr_simple import CD_CDR_simple
from models.cd_cdr_simple_without_diffusion import CD_CDR_simple_without_diffusion
from models.target_pattern_pseudo import apply_model_domain_profile, config_scalar


class CD_CDR_Normal(CD_CDR_simple):
    """CD-CDR with active cross-domain fusion and stable ranking inference."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.config = config
        self.disable_diffusion = bool(config.get("disable_diffusion", False))
        self.normal_fusion_attention_weight = float(
            config_scalar(config.get("normal_fusion_attention_weight"), 0.5)
        )
        self.normal_direct_score_weight = float(
            config_scalar(config.get("normal_direct_score_weight"), 0.5)
        )
        self.normal_diffusion_scale_clip = float(
            config_scalar(config.get("normal_diffusion_scale_clip"), 20.0)
        )
        if self.normal_fusion_attention_weight < 0:
            raise ValueError("normal_fusion_attention_weight must be non-negative.")
        if not 0 <= self.normal_direct_score_weight <= 1:
            raise ValueError("normal_direct_score_weight must lie in [0, 1].")
        if self.normal_diffusion_scale_clip <= 0:
            raise ValueError("normal_diffusion_scale_clip must be positive.")

        self.normal_fusion_norm = nn.LayerNorm(self.embedding_size)
        seed = int(config_scalar(config.get("inference_noise_seed"), 2027))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        fixed_noise = torch.randn(
            self.total_num_users + 1,
            self.embedding_size,
            generator=generator,
        )
        fixed_noise[0].zero_()
        self.register_buffer(
            "normal_inference_noise", fixed_noise, persistent=False
        )

        if self.disable_diffusion:
            for parameter in self.diffu_mlp.parameters():
                parameter.requires_grad = False
            for parameter in self.step_mlp.parameters():
                parameter.requires_grad = False

    def _build_history(self, config, dataloader):
        """Build reproducible histories without consuming the global RNG stream."""
        max_len = int(config["history_len"])
        if max_len <= 0:
            raise ValueError("history_len must be positive.")
        seed = int(config_scalar(config.get("seed"), 999))
        device = config["device"]

        def build(positive_items, num_users, domain_offset):
            histories = torch.zeros(
                (num_users + 1, max_len), dtype=torch.long, device=device
            )
            lengths = torch.zeros(num_users + 1, dtype=torch.long, device=device)
            for user, values in positive_items.items():
                items = np.asarray(sorted(map(int, values)), dtype=np.int64)
                if items.size == 0:
                    continue
                if items.size > max_len:
                    rng = np.random.default_rng(
                        seed + domain_offset + int(user) * 104729
                    )
                    items = items[
                        np.sort(rng.choice(items.size, size=max_len, replace=False))
                    ]
                count = int(items.size)
                histories[int(user), :count] = torch.as_tensor(
                    items, dtype=torch.long, device=device
                )
                lengths[int(user)] = count
            return histories, lengths

        source_history, source_length = build(
            dataloader.dataset.positive_items_src,
            self.num_users_src,
            0,
        )
        target_history, target_length = build(
            dataloader.dataset.positive_items_tgt,
            self.num_users_tgt,
            1000003,
        )
        return source_history, source_length, target_history, target_length

    def domain_condition_generator(
        self, source_representation, target_representation, domain
    ):
        tokens = torch.stack(
            [source_representation, target_representation], dim=1
        )
        domain_ids = torch.full(
            (tokens.shape[0],),
            int(domain),
            dtype=torch.long,
            device=tokens.device,
        )
        query = self.domain_emb(domain_ids).unsqueeze(1)
        attended, _ = self.domain_attn(
            query, tokens, tokens, need_weights=False
        )

        gate_input = torch.cat(
            [source_representation, target_representation], dim=-1
        )
        gate = torch.sigmoid(self.gate_proj(gate_input))
        gated = (
            gate * source_representation
            + (1.0 - gate) * target_representation
        )
        return self.normal_fusion_norm(
            gated
            + self.normal_fusion_attention_weight * attended.squeeze(1)
            + query.squeeze(1)
        )

    def cold_user_condition(self, users):
        source_items = self.history_items_src[users]
        source_length = self.history_len_src[users]
        target_users = users.clone()
        target_users[target_users > self.num_users_overlap] = 0
        target_items = self.history_items_tgt[target_users]
        target_length = self.history_len_tgt[target_users]
        source_representation = self.get_user_representation(
            users,
            source_items,
            source_length,
            user_domain=0,
            item_domain=0,
        )
        target_representation = self.get_user_representation(
            users,
            target_items,
            target_length,
            user_domain=0,
            item_domain=1,
        )
        return self.domain_condition_generator(
            source_representation, target_representation, 1
        )

    def _evaluation_condition(self, users, is_warm):
        if not is_warm:
            return self.cold_user_condition(users)

        target_items = self.history_items_tgt[users]
        target_length = self.history_len_tgt[users]
        source_users = users.clone()
        source_users[source_users > self.num_users_overlap] = 0
        source_items = self.history_items_src[source_users]
        source_length = self.history_len_src[source_users]
        source_representation = self.get_user_representation(
            users,
            source_items,
            source_length,
            user_domain=1,
            item_domain=0,
        )
        target_representation = self.get_user_representation(
            users,
            target_items,
            target_length,
            user_domain=1,
            item_domain=1,
        )
        return self.domain_condition_generator(
            source_representation, target_representation, 1
        )

    @torch.no_grad()
    def sample_from_noise(
        self, model_forward, model_forward_uncon, condition, initial_noise=None
    ):
        if initial_noise is None:
            sample = torch.randn_like(condition)
        else:
            if initial_noise.shape != condition.shape:
                raise ValueError("initial_noise and condition must have equal shapes.")
            sample = initial_noise.to(
                device=condition.device, dtype=condition.dtype
            ).clone()

        for index in reversed(range(self.sub_timesteps)):
            step = torch.full(
                (condition.shape[0],),
                index * self.linespace,
                device=condition.device,
                dtype=torch.long,
            )
            sample = self.i_sample(
                model_forward,
                model_forward_uncon,
                sample,
                condition,
                step,
                index,
            )
        return sample

    def calculate_loss(self, interaction, epoch_idx):
        if self.disable_diffusion:
            return CD_CDR_simple_without_diffusion.calculate_loss(
                self, interaction, epoch_idx
            )
        return super().calculate_loss(interaction, epoch_idx)

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        condition = self._evaluation_condition(users, is_warm)
        item_embeddings = self.item_emb_tgt.weight
        direct_scores = torch.matmul(condition, item_embeddings.T)

        direct_weight = self.normal_direct_score_weight
        if self.disable_diffusion or direct_weight >= 1.0:
            scores = direct_scores
        else:
            initial_noise = self.normal_inference_noise[users]
            generated = self.sample_from_noise(
                self.denoise_step,
                self.denoise_uncon,
                condition,
                initial_noise=initial_noise,
            )
            diffusion_scores = torch.matmul(generated, item_embeddings.T)
            direct_scale = direct_scores[:, 1:].std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1.0e-8)
            diffusion_scale = diffusion_scores[:, 1:].std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1.0e-8)
            scale = (direct_scale / diffusion_scale).clamp(
                max=self.normal_diffusion_scale_clip
            )
            scores = (
                direct_weight * direct_scores
                + (1.0 - direct_weight) * scale * diffusion_scores
            )

        scores[:, 0] = -1.0e9
        return scores
