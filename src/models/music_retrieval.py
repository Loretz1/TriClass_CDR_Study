import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from models.music import (
    extract_CrossDomain_semantics_music_modality_data,
    generate_CrossDomain_semantics_music_embs,
    generate_CrossDomain_semantics_music_final_embs,
)


class MuSiC_Retrieval(GeneralRecommender):
    """LLM semantic retrieval-only variant of MuSiC.

    This model does not train recommender embeddings. It directly scores target
    items with the cached MuSiC LLM semantic embeddings.
    """

    modality_name = "CrossDomain_semantics_music"

    def __init__(self, config, dataloader):
        super(MuSiC_Retrieval, self).__init__(config, dataloader)

        semantic = dataloader.get_modality_embs()[self.modality_name]
        semantic = torch.from_numpy(semantic).float()

        min_rows = self.num_users_src + self.num_items_src + self.num_items_tgt
        if semantic.size(0) < min_rows:
            raise ValueError(
                f"{self.modality_name} has {semantic.size(0)} rows, "
                f"but at least {min_rows} rows are required."
            )

        src_user = semantic[:self.num_users_src]
        tgt_user_start = self.num_users_src
        tgt_user_end = tgt_user_start + self.num_users_tgt
        if semantic.size(0) >= tgt_user_end + self.num_items_src + self.num_items_tgt:
            tgt_user = semantic[tgt_user_start:tgt_user_end]
        else:
            tgt_user = torch.zeros(
                self.num_users_tgt, semantic.size(1), dtype=semantic.dtype
            )

        tgt_item = semantic[-self.num_items_tgt:]

        self.register_buffer("src_user_semantic_emb", self._with_padding(src_user))
        self.register_buffer("tgt_user_semantic_emb", self._with_padding(tgt_user))
        self.register_buffer("tgt_item_semantic_emb", self._with_padding(tgt_item))

        # The current Trainer always builds an optimizer, even with req_training=False.
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _with_padding(emb):
        pad = torch.zeros(1, emb.size(1), dtype=emb.dtype)
        return torch.cat([pad, emb], dim=0)

    def calculate_loss(self, interaction, epoch_idx):
        return self.dummy_param.sum() * 0.0

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()

        if is_warm:
            user_emb = self.tgt_user_semantic_emb[users]
        else:
            user_emb = self.src_user_semantic_emb[users]

        scores = torch.matmul(user_emb, self.tgt_item_semantic_emb.t())
        scores[:, 0] = 0.0
        return scores
