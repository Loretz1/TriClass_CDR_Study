import torch

from models.cdrib_grounded_backbone import CDRIB_Grounded_Backbone
from models.grounded_residual_pseudo import GroundedResidualPseudoMixin


class CDRIB_Grounded_Residual_Pseudo(
    GroundedResidualPseudoMixin, CDRIB_Grounded_Backbone
):
    """CDRIB with a frozen-backbone grounded residual ranker."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_residual_pseudo(
            config, dataloader, self.feature_dim * int(config["GNN"])
        )

    def _residual_target_item_embeddings(self):
        target_user = self.target_user_embedding.weight[1:]
        target_item = self.target_item_embedding.weight[1:]
        _, learned_item = self.target_GNN(
            target_user,
            target_item,
            self.target_UV,
            self.target_VU,
        )
        padding = torch.zeros(
            1,
            learned_item.size(1),
            dtype=learned_item.dtype,
            device=learned_item.device,
        )
        return torch.cat([padding, learned_item], dim=0)
