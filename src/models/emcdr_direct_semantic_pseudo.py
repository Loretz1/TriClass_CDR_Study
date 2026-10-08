from models.direct_semantic_pseudo import DirectSemanticPseudoSupervisor
from models.target_pattern_pseudo import apply_model_domain_profile, config_scalar
from models.emcdr import EMCDR


class EMCDR_Direct_Semantic_Pseudo(EMCDR):
    """EMCDR trained with direct-semantic pseudo interactions."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.config = config
        self.pseudo_supervisor = None
        if self._pseudo_weight() > 0:
            self.pseudo_supervisor = DirectSemanticPseudoSupervisor(config, dataloader)

    def _pseudo_weight(self):
        return float(config_scalar(self.config.get("pseudo_loss_weight"), 0.0))

    def _pseudo_loss(self):
        users, positive_items, negative_items = self.pseudo_supervisor.sample(
            self.config["pseudo_num_per_batch"]
        )
        user_embedding = self.mapping_mlp(
            self.source_user_mlp(self.source_user_embedding(users))
        )
        positive_embedding = self.target_item_mlp(
            self.target_item_embedding(positive_items)
        )
        negative_embedding = self.target_item_mlp(
            self.target_item_embedding(negative_items)
        )
        return self.pseudo_supervisor.bpr_loss(
            user_embedding, positive_embedding, negative_embedding
        )

    def calculate_loss(self, interaction, epoch_idx):
        loss = super().calculate_loss(interaction, epoch_idx)
        pseudo_weight = self._pseudo_weight()
        if pseudo_weight <= 0 or self.stage_id != 2:
            return loss
        if epoch_idx < int(self.config["pseudo_start_epoch"]):
            return loss
        return loss + pseudo_weight * self._pseudo_loss()
