from models.cd_cdr_normal_grounded_adaptive_pseudo import (
    CD_CDR_Normal_Grounded_Adaptive_Pseudo,
)
from models.grounded_expandable_pseudo import GroundedExpandablePseudoMixin


class CD_CDR_Normal_Grounded_Expandable_Pseudo(
    GroundedExpandablePseudoMixin,
    CD_CDR_Normal_Grounded_Adaptive_Pseudo,
):
    """Corrected CD-CDR stage v3 built on adaptive stage v2.1."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_expandable(config, dataloader)

    def _new_dynamic_teacher(self):
        return CD_CDR_Normal_Grounded_Adaptive_Pseudo(
            self.config, self._dynamic_dataloader
        )
