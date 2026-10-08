from models.cd_cdr_normal import CD_CDR_Normal
from models.grounded_residual_pseudo import GroundedResidualPseudoMixin


class CD_CDR_Normal_Grounded_Residual_Pseudo(
    GroundedResidualPseudoMixin, CD_CDR_Normal
):
    """Corrected CD-CDR with a frozen-backbone residual ranker."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_residual_pseudo(
            config, dataloader, self.embedding_size
        )

    def _residual_target_item_embeddings(self):
        return self.item_emb_tgt.weight

