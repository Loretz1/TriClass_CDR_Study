from models.grounded_expandable_pseudo import GroundedExpandablePseudoMixin
from models.unicdr_grounded_adaptive_pseudo import UniCDR_Grounded_Adaptive_Pseudo


class UniCDR_Grounded_Expandable_Pseudo(
    GroundedExpandablePseudoMixin, UniCDR_Grounded_Adaptive_Pseudo
):
    """UniCDR stage v3 built on its selected adaptive stage-v2.1 model."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_expandable(config, dataloader)

    def _new_dynamic_teacher(self):
        return UniCDR_Grounded_Adaptive_Pseudo(
            self.config, self._dynamic_dataloader
        )
