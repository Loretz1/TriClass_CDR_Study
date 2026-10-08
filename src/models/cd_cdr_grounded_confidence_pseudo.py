from models.cd_cdr_simple import CD_CDR_simple
from models.cd_cdr_simple_without_diffusion import CD_CDR_simple_without_diffusion
from models.grounded_confidence_pseudo import GroundedConfidencePseudoSupervisor
from models.target_pattern_pseudo import apply_model_domain_profile, config_scalar


class CD_CDR_Grounded_Confidence_Pseudo(CD_CDR_simple):
    """CD-CDR with confidence-weighted fixed pseudo interactions."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.config = config
        self.disable_diffusion = bool(config.get("disable_diffusion", False))
        if self.disable_diffusion:
            for parameter in self.diffu_mlp.parameters():
                parameter.requires_grad = False
            for parameter in self.step_mlp.parameters():
                parameter.requires_grad = False
        self.pseudo_supervisor = GroundedConfidencePseudoSupervisor(
            config, dataloader
        )

    def _pseudo_weight(self):
        return float(config_scalar(self.config.get("pseudo_loss_weight"), 0.0))

    def _base_loss(self, interaction, epoch_idx):
        if self.disable_diffusion:
            return CD_CDR_simple_without_diffusion.calculate_loss(
                self, interaction, epoch_idx
            )
        return CD_CDR_simple.calculate_loss(self, interaction, epoch_idx)

    def _pseudo_user_embedding(self, users):
        source_items = self.history_items_src[users]
        source_length = self.history_len_src[users]

        target_users = users.clone()
        target_users[target_users > self.num_users_overlap] = 0
        target_items = self.history_items_tgt[target_users]
        target_length = self.history_len_tgt[target_users]
        source_embedding = self.get_user_representation(
            users, source_items, source_length, user_domain=0, item_domain=0
        )
        target_embedding = self.get_user_representation(
            users, target_items, target_length, user_domain=0, item_domain=1
        )
        return self.domain_condition_generator(source_embedding, target_embedding, 1)

    def _pseudo_loss(self):
        users, positive_items, negative_items, confidence = (
            self.pseudo_supervisor.sample(self.config["pseudo_num_per_batch"])
        )
        return self.pseudo_supervisor.weighted_bpr_loss(
            self._pseudo_user_embedding(users),
            self.item_emb_tgt(positive_items),
            self.item_emb_tgt(negative_items),
            confidence,
        )

    def calculate_loss(self, interaction, epoch_idx):
        loss = self._base_loss(interaction, epoch_idx)
        if epoch_idx < int(self.config["pseudo_start_epoch"]):
            return loss
        return loss + self._pseudo_weight() * self._pseudo_loss()

    def full_sort_predict(self, interaction, is_warm):
        if self.disable_diffusion:
            return CD_CDR_simple_without_diffusion.full_sort_predict(
                self, interaction, is_warm
            )
        return CD_CDR_simple.full_sort_predict(self, interaction, is_warm)
