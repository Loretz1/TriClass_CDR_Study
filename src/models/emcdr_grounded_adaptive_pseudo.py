from models.emcdr import EMCDR
from models.emcdr_grounded_confidence_pseudo import (
    EMCDR_Grounded_Confidence_Pseudo,
)
from models.grounded_adaptive_pseudo import GroundedAdaptivePseudoMixin


class EMCDR_Grounded_Adaptive_Pseudo(
    GroundedAdaptivePseudoMixin, EMCDR_Grounded_Confidence_Pseudo
):
    """EMCDR with adaptive direct and residual pseudo supervision."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_adaptive_pseudo(
            config, dataloader, self.feature_dim
        )

    def _adaptive_backbone_loss(self, interaction, epoch_idx):
        return EMCDR.calculate_loss(self, interaction, epoch_idx)

    def _adaptive_source_user_representation(self, users):
        return self.mapping_mlp(
            self.source_user_mlp(self.source_user_embedding(users))
        )

    def _adaptive_target_item_embeddings(self, item_ids=None):
        if item_ids is None:
            item_embedding = self.target_item_embedding.weight
        else:
            item_embedding = self.target_item_embedding(item_ids)
        return self.target_item_mlp(item_embedding)
