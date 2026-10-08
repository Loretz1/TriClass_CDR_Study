from models.grounded_adaptive_pseudo import GroundedAdaptivePseudoMixin
from models.unicdr import UniCDR
from models.unicdr_grounded_confidence_pseudo import (
    UniCDR_Grounded_Confidence_Pseudo,
)


class UniCDR_Grounded_Adaptive_Pseudo(
    GroundedAdaptivePseudoMixin, UniCDR_Grounded_Confidence_Pseudo
):
    """UniCDR with adaptive direct and residual pseudo supervision."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_adaptive_pseudo(
            config, dataloader, self.feature_dim
        )

    def _adaptive_backbone_loss(self, interaction, epoch_idx):
        return UniCDR.calculate_loss(self, interaction, epoch_idx)

    def _adaptive_source_user_representation(self, users):
        return self._cold_source_user_embedding(users)

    def _adaptive_target_item_embeddings(self, item_ids=None):
        if item_ids is None:
            return self.emb_item_tgt.weight
        return self.emb_item_tgt(item_ids)
