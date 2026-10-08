import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from models.llm4cdsr import (
    extract_CrossDomain_semantics_LLM4CDSR_modality_data,
    generate_CrossDomain_semantics_LLM4CDSR_embs,
    generate_CrossDomain_semantics_LLM4CDSR_final_embs,
)


class LLM4CDSR_Retrieval(GeneralRecommender):
    """LLM semantic retrieval-only variant of LLM4CDSR."""

    modality_name = "CrossDomain_semantics_LLM4CDSR"

    def __init__(self, config, dataloader):
        super(LLM4CDSR_Retrieval, self).__init__(config, dataloader)

        semantic = dataloader.get_modality_embs()[self.modality_name]
        semantic = torch.from_numpy(semantic).float()

        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        expected_rows = num_users + self.num_items_src + self.num_items_tgt
        if semantic.size(0) < expected_rows:
            raise ValueError(
                f"{self.modality_name} has {semantic.size(0)} rows, "
                f"but at least {expected_rows} rows are required."
            )

        user = semantic[:num_users]
        tgt_item = semantic[-self.num_items_tgt:]

        self.register_buffer("user_semantic_emb", self._with_padding(user))
        self.register_buffer("tgt_item_semantic_emb", self._with_padding(tgt_item))

        # The current Trainer always builds an optimizer, even with req_training=False.
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
