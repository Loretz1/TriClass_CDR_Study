from models.grounded_expandable_calibration_ablation import (
    GroundedExpandableCalibrationAblationMixin,
)
from models.unicdr_grounded_adaptive_pseudo import UniCDR_Grounded_Adaptive_Pseudo


class UniCDR_Grounded_Expandable_Calibration_Ablation(
    GroundedExpandableCalibrationAblationMixin,
    UniCDR_Grounded_Adaptive_Pseudo,
):
    """UniCDR v3 confidence-calibration ablations."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_expandable(config, dataloader)

    def _new_dynamic_teacher(self):
        return UniCDR_Grounded_Adaptive_Pseudo(
            self.config, self._dynamic_dataloader
        )
