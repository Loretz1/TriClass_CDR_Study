import torch
import torch.nn.functional as F

from models.unicdr_without_contrastive_loss import UniCDR_without_contrastive_loss
from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


class LLM_Pseudo_UniCDR(UniCDR_without_contrastive_loss):
    """UniCDR_without_contrastive_loss with fixed LLM semantic pseudo interactions."""

    modality_name = "CrossDomain_semantics_both_tag"

    def __init__(self, config, dataloader):
        super(LLM_Pseudo_UniCDR, self).__init__(config, dataloader)
        self.src_only_pseudo_pos = self._build_fixed_llm_pseudo_positives(dataloader)
        self.pseudo_ui_pairs = self._flatten_pseudo_pairs(self.src_only_pseudo_pos)

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

    def _encode_cold_src_users(self, users):
        batch_size = users.size(0)
        global_user_emb = self.emb_user_shared(users)

        global_item = torch.zeros(
            (batch_size, 2, self.max_len),
            dtype=torch.long,
            device=self.device,
        )
        global_score = torch.full(
            (batch_size, 2, self.max_len),
            -100.0,
            dtype=torch.float,
            device=self.device,
        )
        global_item[:, 0, :] = self.history_items_src[users]
        global_score[:, 0, :] = self.history_scores_src[users]

        global_item_emb = torch.cat(
            [
                self.emb_item_src(global_item[:, 0, :]),
                self.emb_item_tgt(global_item[:, 1, :]),
            ],
            dim=1,
        )
        user_emb = self.agg_list[-1](global_user_emb, global_item_emb, global_score)
        return F.dropout(user_emb, self.dropout, training=self.training)

    def _pseudo_shared_loss(self):
        if len(self.pseudo_ui_pairs) == 0:
            return torch.zeros((), device=self.device)

        pseudo_users, pseudo_items, neg_items = self._sample_pseudo_batch()

        user_emb = self._encode_cold_src_users(pseudo_users)
        pos_emb = self.emb_item_tgt(pseudo_items)
        pos_emb = F.dropout(pos_emb, self.dropout, training=self.training)
        neg_emb = self.emb_item_tgt(neg_items)
        neg_emb = F.dropout(neg_emb, self.dropout, training=self.training)

        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

    def calculate_loss(self, interaction, epoch_idx):
        loss = super(LLM_Pseudo_UniCDR, self).calculate_loss(interaction, epoch_idx)
        if epoch_idx >= int(self.config["pseudo_start_epoch"]):
            loss = loss + float(self.config["pseudo_rec_weight"]) * self._pseudo_shared_loss()
        return loss
