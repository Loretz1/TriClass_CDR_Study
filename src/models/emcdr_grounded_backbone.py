from models.emcdr import EMCDR
from models.target_pattern_pseudo import apply_model_domain_profile


class EMCDR_Grounded_Backbone(EMCDR):
    """EMCDR reference using the same domain profile as the pseudo plugins."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
