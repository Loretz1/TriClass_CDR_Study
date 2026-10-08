import logging
import math
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss


def _lsdcdr_semantic_backend(config):
    if config["dataset"] == "Douban":
        return import_module(
            "models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban"
        )
    return import_module(
        "models.llm_diff_interaction_as_condition_mask_both_tag_ema"
    )


def extract_CrossDomain_semantics_both_tag_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    backend = _lsdcdr_semantic_backend(config)
    return backend.extract_CrossDomain_semantics_both_tag_modality_data(
        config,
        modality,
        interaction,
        id_mapping,
        raw_data_list,
    )


def generate_CrossDomain_semantics_both_tag_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    backend = _lsdcdr_semantic_backend(config)
    return backend.generate_CrossDomain_semantics_both_tag_embs(
        config,
        modality,
        interaction,
        id_mapping,
        modality_data,
    )


def generate_CrossDomain_semantics_both_tag_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    backend = _lsdcdr_semantic_backend(config)
    return backend.generate_CrossDomain_semantics_both_tag_final_embs(
        config,
        modality,
        interaction,
        id_mapping,
        modality_embs,
    )


class PermutationInvariantSetEncoder(nn.Module):
    """Attention pooling without positional encodings."""

    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.empty_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        nn.init.xavier_uniform_(self.seed)
        nn.init.xavier_uniform_(self.empty_token)

    def forward(self, item_features, item_mask):
        batch_size = item_features.size(0)
        empty = self.empty_token.expand(batch_size, -1, -1)
        tokens = torch.cat([empty, item_features], dim=1)
        valid = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=item_mask.device),
                item_mask,
            ],
            dim=1,
        )
        query = self.seed.expand(batch_size, -1, -1)
        pooled, _ = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=~valid,
            need_weights=False,
        )
        pooled = self.norm1(query + pooled)
        pooled = self.norm2(pooled + self.ffn(pooled))
        return pooled.squeeze(1)


class ConditionalDenoiser(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def _time_embedding(self, timesteps):
        half_dim = self.embedding_dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, noisy_target, timesteps, condition):
        query = noisy_target + self._time_embedding(timesteps)
        query = query.unsqueeze(1)
        condition = condition.unsqueeze(1)
        attended, _ = self.cross_attention(
            query,
            condition,
            condition,
            need_weights=False,
        )
        hidden = self.norm1(query + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return hidden.squeeze(1)


class ConditionalDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_steps = int(config["diffusion_steps"])
        beta_min = float(config["diffusion_beta_min"])
        beta_max = float(config["diffusion_beta_max"])
        beta_roots = torch.linspace(math.sqrt(beta_min), math.sqrt(beta_max), self.num_steps)
        betas = beta_roots.square().clamp(max=0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1.0 - alphas_cumprod).sqrt(),
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).clamp(min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )
        self.denoiser = ConditionalDenoiser(
            int(config["embedding_dim"]),
            int(config["diffusion_attention_heads"]),
            float(config["dropout"]),
        )

    @staticmethod
    def _extract(values, timesteps, target):
        selected = values.gather(0, timesteps)
        return selected.reshape(timesteps.size(0), *((1,) * (target.dim() - 1)))

    def q_sample(self, clean_target, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_target)
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, clean_target) * clean_target
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_target) * noise
        )

    def training_loss(self, clean_target, condition):
        timesteps = torch.randint(
            0,
            self.num_steps,
            (clean_target.size(0),),
            device=clean_target.device,
        )
        noisy_target = self.q_sample(clean_target, timesteps)
        predicted_clean = self.denoiser(noisy_target, timesteps, condition)
        return F.mse_loss(predicted_clean, clean_target), predicted_clean

    def reverse(self, initial_target, condition, start_ratio, deterministic=True):
        start_step = int(round(float(start_ratio) * (self.num_steps - 1)))
        start_step = max(0, min(self.num_steps - 1, start_step))
        timesteps = torch.full(
            (initial_target.size(0),),
            start_step,
            dtype=torch.long,
            device=initial_target.device,
        )
        noise = torch.zeros_like(initial_target) if deterministic else None
        current = self.q_sample(initial_target, timesteps, noise=noise)

        for step in range(start_step, -1, -1):
            timesteps.fill_(step)
            predicted_clean = self.denoiser(current, timesteps, condition)
            current = (
                self._extract(self.posterior_mean_coef1, timesteps, current) * predicted_clean
                + self._extract(self.posterior_mean_coef2, timesteps, current) * current
            )
            if step > 0 and not deterministic:
                variance = self._extract(self.posterior_variance, timesteps, current)
                current = current + variance.sqrt() * torch.randn_like(current)
        return current


class MoEFusion(nn.Module):
    def __init__(self, embedding_dim, num_experts, dropout):
        super().__init__()
        self.gate = nn.Linear(embedding_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, condition, generated_target):
        weights = F.softmax(self.gate(condition), dim=-1)
        expert_input = torch.cat([condition, generated_target], dim=-1)
        outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)


class LGCD_wit_LLM_init(GeneralRecommender):
    """Standalone LGCD using LSDCDR semantic initialization and pseudo sets."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.config = config
        self.embedding_dim = int(config["embedding_dim"])
        self.history_len = int(config["history_len"])
        self.pseudo_top_k = min(int(config["pseudo_top_k"]), self.num_items_tgt)
        self.bpr_loss = BPRLoss()
        self._joint_step = 0
        self._overlap_cursor = 0
        self._overlap_order = None

        semantic_embeddings = torch.from_numpy(
            dataloader.get_modality_embs()["CrossDomain_semantics_both_tag"]
        ).float()
        num_union_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        expected_rows = num_union_users + self.num_items_src + self.num_items_tgt
        if semantic_embeddings.shape[0] != expected_rows:
            raise ValueError(
                "CrossDomain_semantics_both_tag row mismatch: "
                f"expected {expected_rows}, got {semantic_embeddings.shape[0]}"
            )

        user_text = semantic_embeddings[:num_union_users]
        src_item_text = semantic_embeddings[
            num_union_users:num_union_users + self.num_items_src
        ]
        tgt_item_text = semantic_embeddings[num_union_users + self.num_items_src:]
        pseudo_items = self._retrieve_pseudo_items(user_text, tgt_item_text)

        text_dim = semantic_embeddings.shape[1]
        self.register_buffer(
            "src_item_text_emb",
            torch.cat([torch.zeros(1, text_dim), src_item_text], dim=0),
            persistent=False,
        )
        self.register_buffer(
            "tgt_item_text_emb",
            torch.cat([torch.zeros(1, text_dim), tgt_item_text], dim=0),
            persistent=False,
        )
        self.register_buffer("pseudo_target_items", pseudo_items, persistent=False)

        histories = self._build_histories(dataloader)
        self.register_buffer("history_src", histories[0], persistent=False)
        self.register_buffer("history_tgt", histories[1], persistent=False)
        self.register_buffer("history_src_lengths", histories[2], persistent=False)
        self.register_buffer("history_tgt_lengths", histories[3], persistent=False)
        self._target_positive_sets = {
            int(user): set(int(item) for item in items)
            for user, items in dataloader.dataset.positive_items_tgt.items()
        }

        overlap_users = [
            user
            for user in range(1, self.num_users_overlap + 1)
            if self.history_src_lengths[user] > 0 and self.history_tgt_lengths[user] > 0
        ]
        source_only_users = list(range(self.num_users_overlap + 1, self.num_users_src + 1))
        self.register_buffer(
            "overlap_users", torch.tensor(overlap_users, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "source_only_users",
            torch.tensor(source_only_users, dtype=torch.long),
            persistent=False,
        )

        self.src_item_id_embedding = nn.Embedding(
            self.num_items_src + 1,
            self.embedding_dim,
            padding_idx=0,
        )
        self.tgt_item_id_embedding = nn.Embedding(
            self.num_items_tgt + 1,
            self.embedding_dim,
            padding_idx=0,
        )
        self.src_text_adapter = nn.Linear(text_dim, self.embedding_dim, bias=False)
        self.tgt_text_adapter = nn.Linear(text_dim, self.embedding_dim, bias=False)

        num_heads = int(config["set_attention_heads"])
        dropout = float(config["dropout"])
        self.src_id_encoder = PermutationInvariantSetEncoder(
            self.embedding_dim, num_heads, dropout
        )
        self.src_text_encoder = PermutationInvariantSetEncoder(
            self.embedding_dim, num_heads, dropout
        )
        self.tgt_id_encoder = PermutationInvariantSetEncoder(
            self.embedding_dim, num_heads, dropout
        )
        self.tgt_text_encoder = PermutationInvariantSetEncoder(
            self.embedding_dim, num_heads, dropout
        )
        self.src_user_fusion = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.tgt_user_fusion = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.src_item_fusion = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=False)
        self.tgt_item_fusion = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=False)

        self.diffusion = ConditionalDiffusion(config)
        self.guesser = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.moe = MoEFusion(
            self.embedding_dim,
            int(config["num_experts"]),
            dropout,
        )

        self.apply(xavier_uniform_initialization)
        self.src_item_id_embedding.weight.data[0].zero_()
        self.tgt_item_id_embedding.weight.data[0].zero_()
        self._cached_target_item_features = None

    def _retrieve_pseudo_items(self, user_text, target_item_text):
        pseudo = torch.zeros(
            self.num_users_src + 1,
            self.pseudo_top_k,
            dtype=torch.long,
        )
        if self.pseudo_top_k == 0 or self.num_users_src == self.num_users_overlap:
            return pseudo

        source_only_text = user_text[self.num_users_overlap:self.num_users_src]
        requested_device = self.config.get("pseudo_retrieval_device", "auto")
        if requested_device == "auto":
            retrieval_device = self.config["device"]
            if retrieval_device.type == "cuda" and not torch.cuda.is_available():
                retrieval_device = torch.device("cpu")
        else:
            retrieval_device = torch.device(requested_device)

        user_batch_size = int(self.config.get("pseudo_retrieval_batch_size", 256))
        item_batch_size = int(
            self.config.get("pseudo_retrieval_item_batch_size", 8192)
        )
        logging.getLogger().info(
            "[LGCD] Exact pseudo retrieval: %d users x %d items on %s "
            "(user_batch=%d, item_batch=%d).",
            source_only_text.size(0),
            target_item_text.size(0),
            retrieval_device,
            user_batch_size,
            item_batch_size,
        )

        target_item_text = target_item_text.to(retrieval_device)
        with torch.inference_mode():
            for start in range(0, source_only_text.size(0), user_batch_size):
                end = min(start + user_batch_size, source_only_text.size(0))
                users = source_only_text[start:end].to(retrieval_device)
                best_scores = None
                best_indices = None
                for item_start in range(0, target_item_text.size(0), item_batch_size):
                    item_end = min(
                        item_start + item_batch_size,
                        target_item_text.size(0),
                    )
                    similarities = users @ target_item_text[item_start:item_end].t()
                    chunk_k = min(self.pseudo_top_k, item_end - item_start)
                    chunk_scores, chunk_indices = similarities.topk(chunk_k, dim=-1)
                    chunk_indices = chunk_indices + item_start + 1
                    if best_scores is None:
                        best_scores = chunk_scores
                        best_indices = chunk_indices
                    else:
                        candidate_scores = torch.cat([best_scores, chunk_scores], dim=1)
                        candidate_indices = torch.cat([best_indices, chunk_indices], dim=1)
                        keep_k = min(self.pseudo_top_k, candidate_scores.size(1))
                        best_scores, positions = candidate_scores.topk(keep_k, dim=-1)
                        best_indices = candidate_indices.gather(1, positions)

                first_user = self.num_users_overlap + 1 + start
                pseudo[first_user:first_user + best_indices.size(0)] = (
                    best_indices.cpu()
                )

        del target_item_text
        if retrieval_device.type == "cuda":
            torch.cuda.empty_cache()
        return pseudo

    def _build_histories(self, dataloader):
        src_history = torch.zeros(
            self.num_users_src + 1,
            self.history_len,
            dtype=torch.long,
        )
        tgt_history = torch.zeros(
            self.num_users_tgt + 1,
            self.history_len,
            dtype=torch.long,
        )
        src_lengths = torch.zeros(self.num_users_src + 1, dtype=torch.long)
        tgt_lengths = torch.zeros(self.num_users_tgt + 1, dtype=torch.long)

        for user, items in dataloader.dataset.positive_items_src.items():
            ordered = sorted(int(item) for item in items)[:self.history_len]
            if ordered:
                src_history[int(user), :len(ordered)] = torch.tensor(ordered)
                src_lengths[int(user)] = len(ordered)
        for user, items in dataloader.dataset.positive_items_tgt.items():
            ordered = sorted(int(item) for item in items)[:self.history_len]
            if ordered:
                tgt_history[int(user), :len(ordered)] = torch.tensor(ordered)
                tgt_lengths[int(user)] = len(ordered)
        return src_history, tgt_history, src_lengths, tgt_lengths

    @staticmethod
    def _without_positive(history, positive_items):
        history = history.clone()
        history[history == positive_items.unsqueeze(1)] = 0
        return history

    def _src_text_features(self, item_ids):
        return self.src_text_adapter(self.src_item_text_emb[item_ids])

    def _tgt_text_features(self, item_ids):
        return self.tgt_text_adapter(self.tgt_item_text_emb[item_ids])

    def _encode_source(self, users, positive_items=None):
        history = self.history_src[users]
        if positive_items is not None:
            history = self._without_positive(history, positive_items)
        mask = history.ne(0)
        id_representation = self.src_id_encoder(
            self.src_item_id_embedding(history), mask
        )
        text_representation = self.src_text_encoder(
            self._src_text_features(history), mask
        )
        fused = self.src_user_fusion(
            torch.cat([id_representation, text_representation], dim=-1)
        )
        return fused, id_representation, text_representation

    def _encode_target(self, users, positive_items=None):
        history = self.history_tgt[users]
        if positive_items is not None:
            history = self._without_positive(history, positive_items)
        mask = history.ne(0)
        id_representation = self.tgt_id_encoder(
            self.tgt_item_id_embedding(history), mask
        )
        text_representation = self.tgt_text_encoder(
            self._tgt_text_features(history), mask
        )
        fused = self.tgt_user_fusion(
            torch.cat([id_representation, text_representation], dim=-1)
        )
        return fused, text_representation

    def _encode_target_text_set(self, item_ids):
        return self.tgt_text_encoder(
            self._tgt_text_features(item_ids),
            item_ids.ne(0),
        )

    def _source_item_features(self, item_ids):
        return self.src_item_fusion(
            torch.cat(
                [
                    self.src_item_id_embedding(item_ids),
                    self._src_text_features(item_ids),
                ],
                dim=-1,
            )
        )

    def _target_item_features(self, item_ids):
        return self.tgt_item_fusion(
            torch.cat(
                [
                    self.tgt_item_id_embedding(item_ids),
                    self._tgt_text_features(item_ids),
                ],
                dim=-1,
            )
        )

    def _bpr(self, users, positive_items, negative_items):
        positive_scores = (users * positive_items).sum(dim=-1)
        negative_scores = (users * negative_items).sum(dim=-1)
        return self.bpr_loss(positive_scores, negative_scores)

    def _source_pretraining_loss(self, interaction):
        users, _, _ = self._encode_source(
            interaction["users"], interaction["pos_items"]
        )
        return self._bpr(
            users,
            self._source_item_features(interaction["pos_items"]),
            self._source_item_features(interaction["neg_items"]),
        )

    def _target_pretraining_loss(self, interaction):
        users, _ = self._encode_target(
            interaction["users"], interaction["pos_items"]
        )
        return self._bpr(
            users,
            self._target_item_features(interaction["pos_items"]),
            self._target_item_features(interaction["neg_items"]),
        )

    def pre_epoch_processing(self, epoch_idx):
        if self.stage_id == 2 and self.overlap_users.numel() > 0:
            permutation = torch.randperm(
                self.overlap_users.numel(), device=self.overlap_users.device
            )
            self._overlap_order = self.overlap_users[permutation]
            self._overlap_cursor = 0

    def _next_overlap_batch(self):
        batch_size = int(self.config["overlap_batch_size"])
        if self.overlap_users.numel() == 0 or batch_size <= 0:
            return self.overlap_users[:0]
        if self._overlap_order is None:
            self._overlap_order = self.overlap_users

        pieces = []
        remaining = batch_size
        while remaining > 0:
            available = self._overlap_order.numel() - self._overlap_cursor
            take = min(remaining, available)
            pieces.append(
                self._overlap_order[self._overlap_cursor:self._overlap_cursor + take]
            )
            self._overlap_cursor += take
            remaining -= take
            if self._overlap_cursor == self._overlap_order.numel():
                self._overlap_cursor = 0
        return torch.cat(pieces)

    def _sample_overlap_target_positive(self, users):
        history = self.history_tgt[users]
        lengths = self.history_tgt_lengths[users].clamp(min=1)
        offsets = (
            torch.arange(users.size(0), device=users.device) + self._joint_step
        ) % lengths
        return history.gather(1, offsets.unsqueeze(1)).squeeze(1)

    def _sample_target_negatives(
        self,
        users,
        extra_forbidden=None,
        include_target_history=True,
    ):
        negatives = []
        user_list = users.detach().cpu().tolist()
        extra_rows = None
        if extra_forbidden is not None:
            extra_rows = extra_forbidden.detach().cpu().tolist()

        for row_index, user in enumerate(user_list):
            forbidden = (
                set(self._target_positive_sets.get(int(user), set()))
                if include_target_history
                else set()
            )
            if extra_rows is not None:
                forbidden.update(int(item) for item in extra_rows[row_index] if int(item) > 0)
            while True:
                candidate = int(
                    torch.randint(
                        1,
                        self.num_items_tgt + 1,
                        (1,),
                        device=users.device,
                    ).item()
                )
                if candidate not in forbidden:
                    negatives.append(candidate)
                    break
        return torch.tensor(negatives, dtype=torch.long, device=users.device)

    def _real_overlap_loss(self):
        users = self._next_overlap_batch()
        if users.numel() == 0:
            return torch.zeros((), device=self.device)

        positive_items = self._sample_overlap_target_positive(users)
        negative_items = self._sample_target_negatives(users)
        condition, _, _ = self._encode_source(users)
        target_history = self._without_positive(self.history_tgt[users], positive_items)
        clean_target = self._encode_target_text_set(target_history)

        diffusion_loss, generated_target = self.diffusion.training_loss(
            clean_target, condition
        )
        guessed_target = self.guesser(condition)
        guesser_loss = F.mse_loss(guessed_target, clean_target)
        final_users = self.moe(condition, generated_target)
        recommendation_loss = self._bpr(
            final_users,
            self._target_item_features(positive_items),
            self._target_item_features(negative_items),
        )
        return (
            float(self.config["diffusion_loss_weight"]) * diffusion_loss
            + float(self.config["guesser_loss_weight"]) * guesser_loss
            + float(self.config["recommendation_loss_weight"]) * recommendation_loss
        )

    def _pseudo_overlap_loss(self):
        batch_size = int(self.config["pseudo_batch_size"])
        if self.source_only_users.numel() == 0 or self.pseudo_top_k == 0 or batch_size <= 0:
            return torch.zeros((), device=self.device)

        user_indices = torch.randint(
            0,
            self.source_only_users.numel(),
            (batch_size,),
            device=self.source_only_users.device,
        )
        users = self.source_only_users[user_indices]
        pseudo_sets = self.pseudo_target_items[users]
        positive_columns = torch.randint(
            0,
            self.pseudo_top_k,
            (batch_size,),
            device=users.device,
        )
        positive_items = pseudo_sets.gather(1, positive_columns.unsqueeze(1)).squeeze(1)
        target_context = pseudo_sets.clone()
        target_context.scatter_(1, positive_columns.unsqueeze(1), 0)
        negative_items = self._sample_target_negatives(
            users,
            pseudo_sets,
            include_target_history=False,
        )

        condition, _, _ = self._encode_source(users)
        clean_target = self._encode_target_text_set(target_context)
        diffusion_loss, generated_target = self.diffusion.training_loss(
            clean_target, condition
        )
        guesser_loss = F.mse_loss(self.guesser(condition), clean_target)
        alignment_loss = F.mse_loss(generated_target, condition)
        final_users = self.moe(condition, generated_target)
        recommendation_loss = self._bpr(
            final_users,
            self._target_item_features(positive_items),
            self._target_item_features(negative_items),
        )
        return (
            float(self.config["diffusion_loss_weight"]) * diffusion_loss
            + float(self.config["guesser_loss_weight"]) * guesser_loss
            + float(self.config["alignment_loss_weight"]) * alignment_loss
            + float(self.config["recommendation_loss_weight"]) * recommendation_loss
        )

    def _joint_loss(self, interaction):
        source_users, _, _ = self._encode_source(
            interaction["users_src"], interaction["pos_items_src"]
        )
        source_loss = self._bpr(
            source_users,
            self._source_item_features(interaction["pos_items_src"]),
            self._source_item_features(interaction["neg_items_src"]),
        )
        target_users, _ = self._encode_target(
            interaction["users_tgt"], interaction["pos_items_tgt"]
        )
        target_loss = self._bpr(
            target_users,
            self._target_item_features(interaction["pos_items_tgt"]),
            self._target_item_features(interaction["neg_items_tgt"]),
        )
        real_overlap_loss = self._real_overlap_loss()
        pseudo_overlap_loss = self._pseudo_overlap_loss()
        self._joint_step += 1
        return (
            float(self.config["domain_pretrain_loss_weight"]) * (source_loss + target_loss)
            + float(self.config["real_overlap_loss_weight"]) * real_overlap_loss
            + float(self.config["pseudo_overlap_loss_weight"]) * pseudo_overlap_loss
        )

    def calculate_loss(self, interaction, epoch_idx):
        if self.stage_id == 0:
            return self._source_pretraining_loss(interaction)
        if self.stage_id == 1:
            return self._target_pretraining_loss(interaction)
        if self.stage_id == 2:
            return self._joint_loss(interaction)
        raise ValueError(f"Unsupported LGCD training stage: {self.stage_id}")

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        if is_warm:
            user_representation, _ = self._encode_target(users)
        else:
            condition, _, _ = self._encode_source(users)
            initial_target = self.guesser(condition)
            generated_target = self.diffusion.reverse(
                initial_target,
                condition,
                self.config["reverse_start_ratio"],
                deterministic=bool(self.config.get("deterministic_inference", True)),
            )
            user_representation = self.moe(condition, generated_target)

        if self._cached_target_item_features is None:
            all_item_ids = torch.arange(
                self.num_items_tgt + 1,
                dtype=torch.long,
                device=users.device,
            )
            self._cached_target_item_features = self._target_item_features(all_item_ids)
        scores = user_representation @ self._cached_target_item_features.t()
        scores[:, 0] = -1e10
        return scores

    def train(self, mode=True):
        if mode:
            self._cached_target_item_features = None
        return super().train(mode)

    @staticmethod
    def _set_requires_grad(module, flag):
        for parameter in module.parameters():
            parameter.requires_grad = flag

    def set_train_stage(self, stage_id):
        super().set_train_stage(stage_id)
        trainable_modules = [
            self.src_item_id_embedding,
            self.tgt_item_id_embedding,
            self.src_text_adapter,
            self.tgt_text_adapter,
            self.src_id_encoder,
            self.src_text_encoder,
            self.tgt_id_encoder,
            self.tgt_text_encoder,
            self.src_user_fusion,
            self.tgt_user_fusion,
            self.src_item_fusion,
            self.tgt_item_fusion,
            self.diffusion,
            self.guesser,
            self.moe,
        ]
        for module in trainable_modules:
            self._set_requires_grad(module, False)

        if stage_id == 0:
            selected = [
                self.src_item_id_embedding,
                self.src_text_adapter,
                self.src_id_encoder,
                self.src_text_encoder,
                self.src_user_fusion,
                self.src_item_fusion,
            ]
        elif stage_id == 1:
            selected = [
                self.tgt_item_id_embedding,
                self.tgt_text_adapter,
                self.tgt_id_encoder,
                self.tgt_text_encoder,
                self.tgt_user_fusion,
                self.tgt_item_fusion,
            ]
        elif stage_id == 2:
            selected = trainable_modules
        else:
            raise ValueError(f"Unsupported LGCD training stage: {stage_id}")

        for module in selected:
            self._set_requires_grad(module, True)

