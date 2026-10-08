from models.emcdr import EMCDR
from models.grounded_confidence_pseudo import GroundedConfidencePseudoSupervisor
from models.target_pattern_pseudo import apply_model_domain_profile, config_scalar


class EMCDR_Grounded_Confidence_Pseudo(EMCDR):
    """EMCDR with confidence-weighted fixed pseudo interactions."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.config = config
        self.pseudo_supervisor = GroundedConfidencePseudoSupervisor(
            config, dataloader
        )

    def _pseudo_weight(self):
        return float(config_scalar(self.config.get("pseudo_loss_weight"), 0.0))

    def _pseudo_loss(self):
        users, positive_items, negative_items, confidence = (
            self.pseudo_supervisor.sample(self.config["pseudo_num_per_batch"])
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
        return self.pseudo_supervisor.weighted_bpr_loss(
            user_embedding,
            positive_embedding,
            negative_embedding,
            confidence,
        )

    def calculate_loss(self, interaction, epoch_idx):
        loss = super().calculate_loss(interaction, epoch_idx)
        if self.stage_id != 2:
            return loss
        if epoch_idx < int(self.config["pseudo_start_epoch"]):
            return loss
        return loss + self._pseudo_weight() * self._pseudo_loss()
