from models.cd_cdr_simple import CD_CDR_simple
from models.cd_cdr_simple_without_diffusion import CD_CDR_simple_without_diffusion
from models.target_pattern_pseudo import apply_model_domain_profile


class CD_CDR_Grounded_Backbone(CD_CDR_simple):
    """CD-CDR reference using the same domain profile as the pseudo plugins."""

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

    def calculate_loss(self, interaction, epoch_idx):
        if self.disable_diffusion:
            return CD_CDR_simple_without_diffusion.calculate_loss(
                self, interaction, epoch_idx
            )
        return CD_CDR_simple.calculate_loss(self, interaction, epoch_idx)

    def full_sort_predict(self, interaction, is_warm):
        if self.disable_diffusion:
            return CD_CDR_simple_without_diffusion.full_sort_predict(
                self, interaction, is_warm
            )
        return CD_CDR_simple.full_sort_predict(self, interaction, is_warm)
