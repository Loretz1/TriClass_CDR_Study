import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    Diffusion,
    InteractionAggregator,
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
)


class LSDCDR_RandomPseudo(LLM_Diff_interaction_as_condition_mask_both_tag_EMA):
    """LSDCDR with random initial pseudo target interactions.

    This ablation keeps the original diffusion, EMA, self-evolving pseudo-label
    update, and recommendation objective unchanged. The only difference is that
    the initial src-only user pseudo positives are sampled uniformly at random
    instead of being retrieved by LLM semantic embeddings.
    """

    def __init__(self, config, dataloader):
        GeneralRecommender.__init__(self, config, dataloader)

        self.config = config
        self.embedding_dim = config["embedding_dim"]
        self.diff_weight = config["diff_weight"]
        self.bpr_loss = BPRLoss()

        self.emb_user = nn.Embedding(
            self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1,
            self.embedding_dim,
            padding_idx=0,
        )
        self.emb_item_src = nn.Embedding(self.num_items_src + 1, self.embedding_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1, self.embedding_dim, padding_idx=0)

        self.src_only_pseudo_pos = self._build_src_only_pseudo_positives()
        self.pseudo_ui_pairs = []
        for u, items in self.src_only_pseudo_pos.items():
            for i in items:
                self.pseudo_ui_pairs.append((u, i))

        self.diff_src = Diffusion(config)
        self.diff_tgt = Diffusion(config)

        self.diff_tgt_ema = None
        self.ema_initialized = False
        self.global_step = 0

        self.src_interaction_agg = InteractionAggregator(config)
        self.tgt_interaction_agg = InteractionAggregator(config)

        (
            self.history_src_user_src,
            self.history_src_user_tgt,
            self.history_tgt_user_src,
            self.history_tgt_user_tgt,
        ) = self._build_padded_history(dataloader)

        self.apply(xavier_uniform_initialization)
        self.emb_user.weight.data[0, :] = 0
        self.emb_item_src.weight.data[0, :] = 0
        self.emb_item_tgt.weight.data[0, :] = 0

    @torch.no_grad()
    def _build_src_only_pseudo_positives(self):
        top_k = int(self.config["top_k_pos"])
        if top_k > self.num_items_tgt:
            raise ValueError(
                f"top_k_pos={top_k} is larger than target item count={self.num_items_tgt}."
            )

        seed = int(self.config.get("random_pseudo_seed", self.config.get("seed", 999)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        pseudo_pos = {}
        u_start = self.num_users_overlap + 1
        u_end = self.num_users_src + 1

        for u_global in range(u_start, u_end):
            pseudo_items = (
                torch.randperm(self.num_items_tgt, generator=generator)[:top_k] + 1
            ).tolist()
            pseudo_pos[u_global] = set(pseudo_items)

        return pseudo_pos
