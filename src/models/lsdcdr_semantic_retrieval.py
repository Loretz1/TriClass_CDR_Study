import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


class LSDCDRSemanticRetrievalBase(GeneralRecommender):
    """Direct semantic embedding retrieval for LSDCDR-style modalities."""

    modality_name = None

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        modality_name = self.modality_name or config.get("semantic_retrieval_modality")
        if not modality_name:
            raise ValueError("semantic retrieval model requires modality_name.")

        semantic = dataloader.get_modality_embs()[modality_name]
        semantic = torch.from_numpy(semantic).float()

        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        expected_rows = num_users + self.num_items_src + self.num_items_tgt
        if semantic.size(0) < expected_rows:
            raise ValueError(
                f"{modality_name} has {semantic.size(0)} rows, "
                f"but at least {expected_rows} rows are required."
            )

        self.register_buffer("user_semantic_emb", self._with_padding(semantic[:num_users]))
        self.register_buffer("tgt_item_semantic_emb", self._with_padding(semantic[-self.num_items_tgt:]))

        # The trainer still builds an optimizer when req_training=False.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _with_padding(emb):
        pad = torch.zeros(1, emb.size(1), dtype=emb.dtype)
        return torch.cat([pad, emb], dim=0)

    def _to_global_user_id(self, users, is_warm):
        if not is_warm:
            return users

        offset = self.num_users_src - self.num_users_overlap
        return torch.where(users <= self.num_users_overlap, users, users + offset)

    def calculate_loss(self, interaction, epoch_idx):
        return self.dummy_param.sum() * 0.0

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        users_global = self._to_global_user_id(users, is_warm=is_warm)

        user_emb = self.user_semantic_emb[users_global]
        scores = torch.matmul(user_emb, self.tgt_item_semantic_emb.t())
        scores[:, 0] = 0.0
        return scores


class LSDCDR_Semantic_Retrieval(LSDCDRSemanticRetrievalBase):
    modality_name = "CrossDomain_semantics_both_tag"
