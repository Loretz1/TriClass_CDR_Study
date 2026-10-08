import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
)


def _config_value(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


class ConditionalWGANGP(nn.Module):
    """Conditional WGAN-GP block compatible with the original diffusion API."""

    def __init__(self, config):
        super().__init__()
        self.timesteps = 1
        self.dim = int(_config_value(config, "embedding_dim", 64))
        hidden_dim = int(_config_value(config, "wgan_hidden_dim", 128))
        self.reconstruction_weight = float(_config_value(config, "wgan_reconstruction_weight", 1.0))
        self.adversarial_weight = float(_config_value(config, "wgan_adversarial_weight", 0.01))
        self.gp_weight = float(_config_value(config, "wgan_gp_weight", 10.0))
        self.critic_lr = float(_config_value(config, "wgan_critic_lr", 0.0001))
        self.critic_batch_size = int(_config_value(config, "wgan_critic_batch_size", 256))
        self.critic_steps = int(_config_value(config, "wgan_critic_steps", 1))
        self.critic_grad_clip = float(_config_value(config, "wgan_critic_grad_clip", 5.0))
        self.normalize_inputs = bool(_config_value(config, "wgan_normalize_inputs", True))
        self.center_scores = bool(_config_value(config, "wgan_center_scores", True))
        self.eval_noise_samples = int(_config_value(config, "wgan_eval_noise_samples", 4))
        if self.eval_noise_samples < 1:
            raise ValueError("wgan_eval_noise_samples must be at least 1")

        # Use fixed draws from the training latent distribution during evaluation.
        # A zero latent is far outside the typical set of a high-dimensional Gaussian.
        eval_noise_generator = torch.Generator(device="cpu")
        eval_noise_seed = (
            int(_config_value(config, "seed", 999))
            + int(_config_value(config, "generative_seed_offset", 20000))
            + int(_config_value(config, "wgan_eval_noise_seed_offset", 7919))
        )
        eval_noise_generator.manual_seed(eval_noise_seed)
        self.register_buffer(
            "eval_noise_bank",
            torch.randn(
                self.eval_noise_samples,
                self.dim,
                generator=eval_noise_generator,
            ),
            persistent=True,
        )

        self.generator = nn.Sequential(
            nn.Linear(self.dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.apply(self._init_weights)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        self._critic_optimizer = None
        self._critic_batches = []

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_critic_optimizer"] = None
        state["_critic_batches"] = []
        return state

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _generate(self, x_start, cond_src, cond_tgt, noise):
        inputs = torch.cat([x_start, noise, cond_src, cond_tgt], dim=-1)
        return self.generator(inputs)

    def _critic_score(self, value, cond_src, cond_tgt):
        if self.normalize_inputs:
            value = F.layer_norm(value, (value.size(-1),))
            cond_src = F.layer_norm(cond_src, (cond_src.size(-1),))
            cond_tgt = F.layer_norm(cond_tgt, (cond_tgt.size(-1),))
        return self.critic(torch.cat([value, cond_src, cond_tgt], dim=-1))

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, loss_type="l2"):
        del t
        noise = torch.randn_like(x_start)
        generated = self._generate(x_start, cond_src, cond_tgt, noise)
        if loss_type == "l2":
            reconstruction = F.mse_loss(generated, x_start)
        elif loss_type == "l1":
            reconstruction = F.l1_loss(generated, x_start)
        elif loss_type == "huber":
            reconstruction = F.smooth_l1_loss(generated, x_start)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        critic_cond_src = cond_src.detach()
        critic_cond_tgt = cond_tgt.detach()
        fake_score = self._critic_score(generated, critic_cond_src, critic_cond_tgt)
        if self.center_scores:
            with torch.no_grad():
                real_score = self._critic_score(
                    x_start.detach(), critic_cond_src, critic_cond_tgt
                )
            adversarial = -(fake_score - real_score).mean()
        else:
            adversarial = -fake_score.mean()

        if torch.is_grad_enabled() and self.training:
            count = min(x_start.size(0), self.critic_batch_size)
            self._critic_batches.append((
                x_start[:count].detach(),
                generated[:count].detach(),
                cond_src[:count].detach(),
                cond_tgt[:count].detach(),
            ))
        loss = self.reconstruction_weight * reconstruction + self.adversarial_weight * adversarial
        return loss, generated

    def _gradient_penalty(self, real, fake, cond_src, cond_tgt):
        alpha = torch.rand(real.size(0), 1, device=real.device, dtype=real.dtype)
        interpolated = alpha * real + (1.0 - alpha) * fake
        interpolated.requires_grad_(True)
        score = self._critic_score(interpolated, cond_src, cond_tgt)
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def critic_step(self):
        if not self._critic_batches:
            return
        if self._critic_optimizer is None:
            self._critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_lr, betas=(0.0, 0.9)
            )
        batches = self._critic_batches
        self._critic_batches = []
        critic_flags = [parameter.requires_grad for parameter in self.critic.parameters()]
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        try:
            for _ in range(self.critic_steps):
                self._critic_optimizer.zero_grad(set_to_none=True)
                losses = []
                for real, fake, cond_src, cond_tgt in batches:
                    critic_loss = (
                        self._critic_score(fake, cond_src, cond_tgt).mean()
                        - self._critic_score(real, cond_src, cond_tgt).mean()
                        + self.gp_weight * self._gradient_penalty(real, fake, cond_src, cond_tgt)
                    )
                    losses.append(critic_loss)
                torch.stack(losses).mean().backward()
                if self.critic_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.critic_grad_clip
                    )
                self._critic_optimizer.step()
        finally:
            for parameter, requires_grad in zip(self.critic.parameters(), critic_flags):
                parameter.requires_grad_(requires_grad)
                parameter.grad = None

    @torch.no_grad()
    def sample(self, x_start, cond_src, cond_tgt):
        generated = torch.zeros_like(x_start)
        for noise_template in self.eval_noise_bank:
            noise = noise_template.to(dtype=x_start.dtype).unsqueeze(0).expand_as(x_start)
            generated.add_(self._generate(x_start, cond_src, cond_tgt, noise))
        generated.div_(self.eval_noise_samples)
        return x_start, generated, generated, generated, generated


class LSDCDR_WGANGP(LLM_Diff_interaction_as_condition_mask_both_tag_EMA):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        rng_state = torch.get_rng_state()
        try:
            seed = int(config["seed"]) + int(_config_value(config, "generative_seed_offset", 20000))
            torch.default_generator.manual_seed(seed)
            self.diff_src = ConditionalWGANGP(config)
            self.diff_tgt = ConditionalWGANGP(config)
        finally:
            torch.set_rng_state(rng_state)

    def post_batch_processing(self, epoch_idx, batch_idx):
        self.diff_src.critic_step()
        self.diff_tgt.critic_step()
        super().post_batch_processing(epoch_idx, batch_idx)
