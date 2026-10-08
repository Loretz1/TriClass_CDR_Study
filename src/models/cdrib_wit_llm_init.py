import torch

from models.cdrib import CDRIB
from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


class CDRIB_wit_LLM_init(CDRIB):
    """CDRIB backbone with fixed LSDCDR semantic pseudo supervision."""

    modality_name = "CrossDomain_semantics_both_tag"

    def __init__(self, config, dataloader):
        super(CDRIB_wit_LLM_init, self).__init__(config, dataloader)
        if self._pseudo_weight() > 0:
            self.src_only_pseudo_pos = self._build_fixed_llm_pseudo_positives(dataloader)
            self.pseudo_ui_pairs = self._flatten_pseudo_pairs(self.src_only_pseudo_pos)
        else:
            self.src_only_pseudo_pos = {}
            self.pseudo_ui_pairs = []

    def _pseudo_weight(self):
        value = self.config.get("pseudo_rec_weight", 0.0)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else 0.0
        return float(value)

    @torch.no_grad()
    def _build_fixed_llm_pseudo_positives(self, dataloader):
        top_k = int(self.config["top_k_pos"])
        semantic = torch.from_numpy(dataloader.get_modality_embs()[self.modality_name]).float()

        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        expected_rows = num_users + self.num_items_src + self.num_items_tgt
        if semantic.size(0) != expected_rows:
            raise ValueError(
                f"{self.modality_name} has {semantic.size(0)} rows, expected {expected_rows}."
            )

        user_semantic_emb = semantic[:num_users]
        tgt_item_semantic_emb = semantic[num_users + self.num_items_src:]

        u_start = self.num_users_overlap + 1
        u_end = self.num_users_src + 1
        pseudo_pos = {}
        batch_size = int(self.config.get("pseudo_build_batch_size", 2048))

        for start in range(u_start, u_end, batch_size):
            end = min(start + batch_size, u_end)
            user_chunk = user_semantic_emb[start - 1:end - 1]
            sim = torch.matmul(user_chunk, tgt_item_semantic_emb.t())
            topk_idx = torch.topk(sim, k=top_k, dim=1).indices + 1
            for row, uid in enumerate(range(start, end)):
                pseudo_pos[uid] = set(topk_idx[row].tolist())

        return pseudo_pos

    @staticmethod
    def _flatten_pseudo_pairs(pseudo_pos):
        return [(u, i) for u, items in pseudo_pos.items() for i in items]

    def _sample_tgt_neg_excluding_pseudo(self, users, max_retry=5):
        neg_items = []
        for u in users.tolist():
            forbid = self.src_only_pseudo_pos.get(u, set())
            neg = None
            for _ in range(max_retry):
                cand = torch.randint(
                    low=1,
                    high=self.num_items_tgt + 1,
                    size=(1,),
                    device=self.device,
                ).item()
                if cand not in forbid:
                    neg = cand
                    break
            neg_items.append(cand if neg is None else neg)
        return torch.tensor(neg_items, dtype=torch.long, device=self.device)

    def _sample_pseudo_batch(self):
        num_pseudo = int(self.config["pseudo_num_per_batch"])
        idx = torch.randint(
            low=0,
            high=len(self.pseudo_ui_pairs),
            size=(num_pseudo,),
            device=self.device,
        )
        users, items = [], []
        for j in idx.tolist():
            u, i = self.pseudo_ui_pairs[j]
            users.append(u)
            items.append(i)
        users = torch.tensor(users, dtype=torch.long, device=self.device)
        items = torch.tensor(items, dtype=torch.long, device=self.device)
        neg_items = self._sample_tgt_neg_excluding_pseudo(users)
        return users, items, neg_items

    def _graph_encode_for_pseudo(self):
        source_user = self.source_user_embedding.weight[1:]
        target_user = self.target_user_embedding.weight[1:]
        source_item = self.source_item_embedding.weight[1:]
        target_item = self.target_item_embedding.weight[1:]
        source_learn_user, _ = self.source_GNN(
            source_user, source_item, self.source_UV, self.source_VU
        )
        _, target_learn_item = self.target_GNN(
            target_user, target_item, self.target_UV, self.target_VU
        )
        return source_learn_user, target_learn_item

    def _pseudo_cold_source_loss(self):
        if len(self.pseudo_ui_pairs) == 0:
            return torch.zeros((), device=self.device)

        pseudo_users, pseudo_items, neg_items = self._sample_pseudo_batch()
        source_learn_user, target_learn_item = self._graph_encode_for_pseudo()

        user_emb = source_learn_user[pseudo_users - 1]
        pos_emb = target_learn_item[pseudo_items - 1]
        neg_emb = target_learn_item[neg_items - 1]

        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

    def calculate_loss(self, interaction, epoch_idx):
        loss = CDRIB.calculate_loss(self, interaction, epoch_idx)
        pseudo_weight = self._pseudo_weight()
        if pseudo_weight <= 0:
            return loss
        if epoch_idx >= int(self.config["pseudo_start_epoch"]):
            loss = loss + pseudo_weight * self._pseudo_cold_source_loss()
        return loss
