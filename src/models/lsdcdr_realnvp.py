import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
)


def _config_value(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim, mask, scale_limit):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_limit = scale_limit
        self.network = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2),
        )

    def _coupling_parameters(self, value, cond_src, cond_tgt):
        masked = value * self.mask
        log_scale, shift = self.network(
            torch.cat([masked, cond_src, cond_tgt], dim=-1)
        ).chunk(2, dim=-1)
        inverse_mask = 1.0 - self.mask
        log_scale = torch.tanh(log_scale) * self.scale_limit * inverse_mask
        shift = shift * inverse_mask
        return masked, log_scale, shift, inverse_mask

    def forward(self, value, cond_src, cond_tgt):
        masked, log_scale, shift, inverse_mask = self._coupling_parameters(value, cond_src, cond_tgt)
        output = masked + inverse_mask * (value * torch.exp(log_scale) + shift)
        return output, log_scale.sum(dim=-1)

    def inverse(self, value, cond_src, cond_tgt):
        masked, log_scale, shift, inverse_mask = self._coupling_parameters(value, cond_src, cond_tgt)
        return masked + inverse_mask * ((value - shift) * torch.exp(-log_scale))


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP with the same call surface as the diffusion block."""

    def __init__(self, config):
        super().__init__()
        self.timesteps = 1
        self.dim = int(_config_value(config, "embedding_dim", 64))
        hidden_dim = int(_config_value(config, "realnvp_hidden_dim", 128))
        num_layers = int(_config_value(config, "realnvp_num_layers", 4))
        scale_limit = float(_config_value(config, "realnvp_scale_limit", 1.5))
        self.reconstruction_weight = float(
            _config_value(config, "realnvp_reconstruction_weight", 1.0)
        )
        self.likelihood_weight = float(
            _config_value(config, "realnvp_likelihood_weight", 0.01)
        )
        self.layers = nn.ModuleList()
        coordinates = torch.arange(self.dim)
        for layer_index in range(num_layers):
            mask = ((coordinates + layer_index) % 2).float()
            self.layers.append(
                ConditionalAffineCoupling(self.dim, hidden_dim, mask, scale_limit)
            )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_flow(self, value, cond_src, cond_tgt):
        log_det = value.new_zeros(value.size(0))
        for layer in self.layers:
            value, layer_log_det = layer(value, cond_src, cond_tgt)
            log_det = log_det + layer_log_det
        return value, log_det

    def _inverse_flow(self, value, cond_src, cond_tgt):
        for layer in reversed(self.layers):
            value = layer.inverse(value, cond_src, cond_tgt)
        return value

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, loss_type="l2"):
        del t

        # Flow likelihood assumes fixed observations. Detaching this branch keeps
        # the jointly trained recommender embeddings from exploiting log-det scale.
        latent, log_det = self._forward_flow(
            x_start.detach(), cond_src.detach(), cond_tgt.detach()
        )
        log_base = -0.5 * (latent.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)
        raw_nll = -(log_base + log_det).mean() / self.dim
        likelihood_penalty = F.softplus(raw_nll)

        generated = self._inverse_flow(torch.zeros_like(x_start), cond_src, cond_tgt)
        target = x_start.detach()
        if loss_type == "l2":
            reconstruction = F.mse_loss(generated, target)
        elif loss_type == "l1":
            reconstruction = F.l1_loss(generated, target)
        elif loss_type == "huber":
            reconstruction = F.smooth_l1_loss(generated, target)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        loss = (
            self.reconstruction_weight * reconstruction
            + self.likelihood_weight * likelihood_penalty
        )
        return loss, generated

    @torch.no_grad()
    def sample(self, x_start, cond_src, cond_tgt):
        generated = self._inverse_flow(torch.zeros_like(x_start), cond_src, cond_tgt)
        return x_start, generated, generated, generated, generated


class LSDCDR_RealNVP(LLM_Diff_interaction_as_condition_mask_both_tag_EMA):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        rng_state = torch.get_rng_state()
        try:
            seed = int(config["seed"]) + int(_config_value(config, "generative_seed_offset", 30000))
            torch.default_generator.manual_seed(seed)
            self.diff_src = ConditionalRealNVP(config)
            self.diff_tgt = ConditionalRealNVP(config)
        finally:
            torch.set_rng_state(rng_state)
