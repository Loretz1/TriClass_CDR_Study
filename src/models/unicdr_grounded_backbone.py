from models.target_pattern_pseudo import apply_model_domain_profile
from models.unicdr import UniCDR


class UniCDR_Grounded_Backbone(UniCDR):
    """UniCDR reference using the same domain profile as the pseudo plugins."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
