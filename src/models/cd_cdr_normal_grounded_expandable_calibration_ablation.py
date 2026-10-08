from models.cd_cdr_normal_grounded_adaptive_pseudo import (
    CD_CDR_Normal_Grounded_Adaptive_Pseudo,
)
from models.grounded_expandable_calibration_ablation import (
    GroundedExpandableCalibrationAblationMixin,
)


class CD_CDR_Normal_Grounded_Expandable_Calibration_Ablation(
    GroundedExpandableCalibrationAblationMixin,
    CD_CDR_Normal_Grounded_Adaptive_Pseudo,
):
    """Corrected CD-CDR v3 confidence-calibration ablations."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_expandable(config, dataloader)

    def _new_dynamic_teacher(self):
        return CD_CDR_Normal_Grounded_Adaptive_Pseudo(
            self.config, self._dynamic_dataloader
        )
