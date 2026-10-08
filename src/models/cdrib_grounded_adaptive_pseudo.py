import torch

from models.cdrib import CDRIB
from models.cdrib_grounded_confidence_pseudo import (
    CDRIB_Grounded_Confidence_Pseudo,
)
from models.grounded_adaptive_pseudo import GroundedAdaptivePseudoMixin


class CDRIB_Grounded_Adaptive_Pseudo(
    GroundedAdaptivePseudoMixin, CDRIB_Grounded_Confidence_Pseudo
):
    """CDRIB with adaptive direct and residual pseudo supervision."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_adaptive_pseudo(
            config, dataloader, self.feature_dim * int(config["GNN"])
        )

    def _adaptive_backbone_loss(self, interaction, epoch_idx):
        self._source_graph_output = None
        self._target_graph_output = None
        return CDRIB.calculate_loss(self, interaction, epoch_idx)

    def _adaptive_source_user_representation(self, users):
        if self._source_graph_output is None:
            raise RuntimeError("CDRIB source graph embeddings are unavailable.")
        return self._source_graph_output[0][users - 1]

    def _adaptive_target_item_embeddings(self, item_ids=None):
        if self._target_graph_output is None:
            raise RuntimeError("CDRIB target graph embeddings are unavailable.")
        target_items = self._target_graph_output[1]
        if item_ids is not None:
            return target_items[item_ids - 1]
        padding = torch.zeros(
            1,
            target_items.size(1),
            dtype=target_items.dtype,
            device=target_items.device,
        )
        return torch.cat([padding, target_items], dim=0)
