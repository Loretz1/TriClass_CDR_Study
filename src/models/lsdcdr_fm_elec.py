import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_diff_interaction_as_condition_mask_both_tag_ema_elec import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec,
)


def _config_value(config, key, default):
    value = config.get(key, default)
    return default if value is None else value


class ConditionalFlowMatching(nn.Module):
    """Conditional flow matching with the original diffusion call surface."""

    def __init__(self, config):
        super().__init__()
        self.timesteps = max(1, int(_config_value(config, "fm_timesteps", config["timesteps"])))
        self.sampling_steps = max(
            1, int(_config_value(config, "fm_sampling_steps", self.timesteps))
        )
        self.dim = int(_config_value(config, "feature_dim", 64))
        self.solver = str(_config_value(config, "fm_solver", "euler")).lower()
        self.noise_scale = float(_config_value(config, "fm_noise_scale", 1.0))
        self.eval_noise_samples = max(
            1, int(_config_value(config, "fm_eval_noise_samples", 1))
        )
        if self.solver not in {"euler", "heun"}:
            raise ValueError("fm_solver must be either 'euler' or 'heun'")
        if self.noise_scale <= 0:
            raise ValueError("fm_noise_scale must be positive")

        self.w_q = nn.Linear(self.dim, self.dim, bias=False)
        self.w_k = nn.Linear(self.dim, self.dim, bias=False)
        self.w_v = nn.Linear(self.dim, self.dim, bias=False)
        self.ln = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.apply(self._init_weights)

        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(
            int(_config_value(config, "seed", 999))
            + int(_config_value(config, "generative_seed_offset", 40000))
            + int(_config_value(config, "fm_eval_noise_seed_offset", 7919))
        )
        self.register_buffer(
            "eval_noise_bank",
            torch.randn(
                self.eval_noise_samples,
                self.dim,
                generator=noise_generator,
            ),
            persistent=True,
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _time_embedding(self, time):
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, dtype=time.dtype, device=time.device) * -scale
        )
        angles = time[:, None] * 1000.0 * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def _self_attention(self, features):
        mask = features.abs().sum(dim=-1) > 0
        features = self.ln(features)
        query = self.w_q(features)
        key = self.w_k(features)
        value = self.w_v(features)
        logits = (query * (self.dim ** -0.5)) @ key.transpose(-1, -2)
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)
        attention = logits.softmax(dim=-1)
        output = (attention @ value) * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return output.sum(dim=1) / denominator

    def _velocity(self, state, time, cond_src, cond_tgt):
        time_embedding = self._time_embedding(time)
        tokens = torch.stack([state, time_embedding, cond_src, cond_tgt], dim=1)
        return self._self_attention(tokens)

    def _base_noise(self, x_start):
        if torch.is_grad_enabled() and self.training:
            return torch.randn_like(x_start) * self.noise_scale
        template = self.eval_noise_bank[0].to(
            device=x_start.device, dtype=x_start.dtype
        )
        return template.unsqueeze(0).expand_as(x_start) * self.noise_scale

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, loss_type="l2"):
        time = (t.to(dtype=x_start.dtype) + 0.5) / float(self.timesteps)
        time = time.clamp(min=0.0, max=1.0)
        base = self._base_noise(x_start)
        time_column = time.unsqueeze(-1)
        state = (1.0 - time_column) * base + time_column * x_start
        target_velocity = x_start - base
        predicted_velocity = self._velocity(state, time, cond_src, cond_tgt)

        if loss_type == "l2":
            loss = F.mse_loss(predicted_velocity, target_velocity)
        elif loss_type == "l1":
            loss = F.l1_loss(predicted_velocity, target_velocity)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_velocity, target_velocity)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        predicted_x1 = state + (1.0 - time_column) * predicted_velocity
        return loss, predicted_x1

    def _integrate(self, initial_state, cond_src, cond_tgt):
        state = initial_state
        step_size = 1.0 / float(self.sampling_steps)
        quarter_step = max(1, round(self.sampling_steps * 0.25))
        half_step = max(1, round(self.sampling_steps * 0.50))
        three_quarter_step = max(1, round(self.sampling_steps * 0.75))
        quarter = state
        half = state
        three_quarter = state

        for step in range(self.sampling_steps):
            time_value = step / float(self.sampling_steps)
            time = torch.full(
                (state.size(0),),
                time_value,
                dtype=state.dtype,
                device=state.device,
            )
            velocity = self._velocity(state, time, cond_src, cond_tgt)
            if self.solver == "heun" and step + 1 < self.sampling_steps:
                proposal = state + step_size * velocity
                next_time = torch.full_like(time, (step + 1) / float(self.sampling_steps))
                next_velocity = self._velocity(
                    proposal, next_time, cond_src, cond_tgt
                )
                state = state + 0.5 * step_size * (velocity + next_velocity)
            else:
                state = state + step_size * velocity

            completed_step = step + 1
            if completed_step == quarter_step:
                quarter = state
            if completed_step == half_step:
                half = state
            if completed_step == three_quarter_step:
                three_quarter = state

        return state, quarter, half, three_quarter

    @torch.no_grad()
    def sample(self, x_start, cond_src, cond_tgt):
        generated = torch.zeros_like(x_start)
        quarter = torch.zeros_like(x_start)
        half = torch.zeros_like(x_start)
        three_quarter = torch.zeros_like(x_start)

        for noise_template in self.eval_noise_bank:
            initial_state = noise_template.to(
                device=x_start.device, dtype=x_start.dtype
            ).unsqueeze(0).expand_as(x_start)
            initial_state = initial_state * self.noise_scale
            output, output_quarter, output_half, output_three_quarter = self._integrate(
                initial_state, cond_src, cond_tgt
            )
            generated.add_(output)
            quarter.add_(output_quarter)
            half.add_(output_half)
            three_quarter.add_(output_three_quarter)

        generated.div_(self.eval_noise_samples)
        quarter.div_(self.eval_noise_samples)
        half.div_(self.eval_noise_samples)
        three_quarter.div_(self.eval_noise_samples)
        return x_start, generated, quarter, half, three_quarter


class LSDCDR_FM_Elec(LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        rng_state = torch.get_rng_state()
        try:
            seed = int(config["seed"]) + int(
                _config_value(config, "generative_seed_offset", 40000)
            )
            torch.default_generator.manual_seed(seed)
            self.diff_src = ConditionalFlowMatching(config)
            self.diff_tgt = ConditionalFlowMatching(config)
        finally:
            torch.set_rng_state(rng_state)
