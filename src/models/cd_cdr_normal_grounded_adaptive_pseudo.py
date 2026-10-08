from models.cd_cdr_normal_grounded_confidence_pseudo import (
    CD_CDR_Normal_Grounded_Confidence_Pseudo,
)
from models.grounded_adaptive_pseudo import GroundedAdaptivePseudoMixin


class CD_CDR_Normal_Grounded_Adaptive_Pseudo(
    GroundedAdaptivePseudoMixin,
    CD_CDR_Normal_Grounded_Confidence_Pseudo,
):
    """Corrected CD-CDR with joint direct and residual supervision."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_adaptive_pseudo(
            config, dataloader, self.embedding_size
        )

    def _adaptive_backbone_loss(self, interaction, epoch_idx):
        return self._base_loss(interaction, epoch_idx)

    def _adaptive_source_user_representation(self, users):
        return self._pseudo_user_embedding(users)

    def _adaptive_target_item_embeddings(self, item_ids=None):
        if item_ids is None:
            return self.item_emb_tgt.weight
        return self.item_emb_tgt(item_ids)

