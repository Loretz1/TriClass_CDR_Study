from models.grounded_residual_pseudo import GroundedResidualPseudoMixin
from models.unicdr_grounded_backbone import UniCDR_Grounded_Backbone


class UniCDR_Grounded_Residual_Pseudo(
    GroundedResidualPseudoMixin, UniCDR_Grounded_Backbone
):
    """UniCDR with a frozen-backbone grounded residual ranker."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_residual_pseudo(
            config, dataloader, self.feature_dim
        )

    def _residual_target_item_embeddings(self):
        return self.emb_item_tgt.weight
