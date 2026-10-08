import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender


class Random_Retrieval(GeneralRecommender):
    """Training-free random target-item ranking baseline."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.random_seed = int(config.get("random_retrieval_seed", config.get("seed", 999)))
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def _to_global_user_id(self, users, is_warm):
        if not is_warm:
            return users

        offset = self.num_users_src - self.num_users_overlap
        return torch.where(users <= self.num_users_overlap, users, users + offset)

    def calculate_loss(self, interaction, epoch_idx):
        return self.dummy_param.sum() * 0.0

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        users_global = self._to_global_user_id(users, is_warm=is_warm)

        scores = torch.empty(
            users_global.size(0),
            self.num_items_tgt + 1,
            dtype=torch.float32,
            device=users_global.device,
        )
        for row, uid in enumerate(users_global.detach().cpu().tolist()):
            generator = torch.Generator(device=users_global.device)
            generator.manual_seed(self.random_seed + int(uid) * 1009)
            scores[row] = torch.rand(
                self.num_items_tgt + 1,
                generator=generator,
                device=users_global.device,
            )

        scores[:, 0] = 0.0
        return scores
