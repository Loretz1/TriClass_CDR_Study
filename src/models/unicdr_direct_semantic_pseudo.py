import torch
import torch.nn.functional as F

from models.direct_semantic_pseudo import DirectSemanticPseudoSupervisor
from models.target_pattern_pseudo import apply_model_domain_profile, config_scalar
from models.unicdr import UniCDR


class UniCDR_Direct_Semantic_Pseudo(UniCDR):
    """UniCDR trained with direct-semantic pseudo interactions."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.pseudo_supervisor = None
        if self._pseudo_weight() > 0:
            self.pseudo_supervisor = DirectSemanticPseudoSupervisor(config, dataloader)

    def _pseudo_weight(self):
        return float(config_scalar(self.config.get("pseudo_loss_weight"), 0.0))

    def _cold_source_user_embedding(self, users):
        batch_size = users.size(0)
        global_user_embedding = self.emb_user_shared(users)
        global_items = torch.zeros(
            (batch_size, 2, self.max_len), dtype=torch.long, device=self.device
        )
        global_scores = torch.full(
            (batch_size, 2, self.max_len),
            -100.0,
            dtype=torch.float,
            device=self.device,
        )
        global_items[:, 0, :] = self.history_items_src[users]
        global_scores[:, 0, :] = self.history_scores_src[users]
        global_item_embedding = torch.cat(
            [
                self.emb_item_src(global_items[:, 0, :]),
                self.emb_item_tgt(global_items[:, 1, :]),
            ],
            dim=1,
        )
        user_embedding = self.agg_list[-1](
            global_user_embedding, global_item_embedding, global_scores
        )
        return F.dropout(user_embedding, self.dropout, training=self.training)

    def _pseudo_loss(self):
        users, positive_items, negative_items = self.pseudo_supervisor.sample(
            self.config["pseudo_num_per_batch"]
        )
        user_embedding = self._cold_source_user_embedding(users)
        positive_embedding = F.dropout(
            self.emb_item_tgt(positive_items), self.dropout, training=self.training
        )
        negative_embedding = F.dropout(
            self.emb_item_tgt(negative_items), self.dropout, training=self.training
        )
        return self.pseudo_supervisor.bpr_loss(
            user_embedding, positive_embedding, negative_embedding
        )

    def calculate_loss(self, interaction, epoch_idx):
        loss = super().calculate_loss(interaction, epoch_idx)
        pseudo_weight = self._pseudo_weight()
        if pseudo_weight <= 0 or epoch_idx < int(self.config["pseudo_start_epoch"]):
            return loss
        return loss + pseudo_weight * self._pseudo_loss()
