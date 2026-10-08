from models.emcdr_grounded_backbone import EMCDR_Grounded_Backbone
from models.grounded_residual_pseudo import GroundedResidualPseudoMixin


class EMCDR_Grounded_Residual_Pseudo(
    GroundedResidualPseudoMixin, EMCDR_Grounded_Backbone
):
    """EMCDR with a frozen-backbone grounded residual ranker."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_residual_pseudo(
            config, dataloader, self.feature_dim
        )

    def _residual_target_item_embeddings(self):
        return self.target_item_mlp(self.target_item_embedding.weight)
