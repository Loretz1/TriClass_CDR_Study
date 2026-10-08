import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban,
)


def _config_value(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


class ConditionalVAE(nn.Module):
    """Conditional VAE with the same call surface as the original diffusion block."""

    def __init__(self, config):
        super().__init__()
        self.timesteps = 1
        dim = int(_config_value(config, "embedding_dim", 64))
        hidden_dim = int(_config_value(config, "vae_hidden_dim", 128))
        latent_dim = int(_config_value(config, "vae_latent_dim", dim // 2))
        self.kl_weight = float(_config_value(config, "vae_kl_weight", 0.001))

        self.encoder = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode(self, x_start, cond_src, cond_tgt):
        hidden = self.encoder(torch.cat([x_start, cond_src, cond_tgt], dim=-1))
        return self.mu(hidden), self.logvar(hidden).clamp(min=-10.0, max=10.0)

    def _decode(self, latent, cond_src, cond_tgt):
        return self.decoder(torch.cat([latent, cond_src, cond_tgt], dim=-1))

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, loss_type="l2"):
        del t
        mu, logvar = self._encode(x_start, cond_src, cond_tgt)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(std)
        generated = self._decode(latent, cond_src, cond_tgt)

        if loss_type == "l2":
            reconstruction = F.mse_loss(generated, x_start)
        elif loss_type == "l1":
            reconstruction = F.l1_loss(generated, x_start)
        elif loss_type == "huber":
            reconstruction = F.smooth_l1_loss(generated, x_start)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction + self.kl_weight * kl, generated

    @torch.no_grad()
    def sample(self, x_start, cond_src, cond_tgt):
        mu, _ = self._encode(x_start, cond_src, cond_tgt)
        generated = self._decode(mu, cond_src, cond_tgt)
        return x_start, generated, generated, generated, generated


class LSDCDR_VAE_Douban(LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        rng_state = torch.get_rng_state()
        try:
            seed = int(config["seed"]) + int(_config_value(config, "generative_seed_offset", 10000))
            torch.default_generator.manual_seed(seed)
            self.diff_src = ConditionalVAE(config)
            self.diff_tgt = ConditionalVAE(config)
        finally:
            torch.set_rng_state(rng_state)
