import torch
import torch.nn.functional as F

from models.cdrib_grounded_adaptive_pseudo import CDRIB_Grounded_Adaptive_Pseudo
from models.grounded_expandable_calibration_ablation import (
    GroundedExpandableCalibrationAblationMixin,
)


class CDRIB_Grounded_Expandable_Calibration_Ablation(
    GroundedExpandableCalibrationAblationMixin,
    CDRIB_Grounded_Adaptive_Pseudo,
):
    """CDRIB v3 confidence-calibration ablations."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self._init_grounded_expandable(config, dataloader)

    def _new_dynamic_teacher(self):
        return CDRIB_Grounded_Adaptive_Pseudo(
            self.config, self._dynamic_dataloader
        )

    def _prepare_dynamic_teacher(self):
        teacher = self._ema_teacher
        with torch.no_grad():
            source_user, _source_item = teacher.source_GNN(
                teacher.source_user_embedding.weight[1:],
                teacher.source_item_embedding.weight[1:],
                teacher.source_UV,
                teacher.source_VU,
            )
            _target_user, target_item = teacher.target_GNN(
                teacher.target_user_embedding.weight[1:],
                teacher.target_item_embedding.weight[1:],
                teacher.target_UV,
                teacher.target_VU,
            )
            zero = torch.zeros(1, source_user.size(1), device=source_user.device)
            object.__setattr__(
                teacher,
                "_dynamic_source_users",
                torch.cat([zero, source_user], dim=0),
            )
            zero = torch.zeros(1, target_item.size(1), device=target_item.device)
            object.__setattr__(
                teacher,
                "_dynamic_target_items",
                torch.cat([zero, target_item], dim=0),
            )

    def _dynamic_teacher_scores(self, users):
        teacher = self._ema_teacher
        base_scores = torch.matmul(
            teacher._dynamic_source_users[users],
            teacher._dynamic_target_items.t(),
        )
        if (
            not teacher._adaptive_residual_active
            or teacher.adaptive_residual_beta == 0
        ):
            base_scores[:, 0] = -torch.inf
            return base_scores

        source_users = teacher._dynamic_source_users[users]
        target_items = F.normalize(teacher._dynamic_target_items, dim=-1)
        residual_users = teacher.adaptive_residual_adapter(source_users)
        residual_scores = torch.matmul(residual_users, target_items.t())
        base_scale = base_scores[:, 1:].std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-8)
        residual_scale = residual_scores[:, 1:].std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-8)
        score_scale = (base_scale / residual_scale).clamp(
            max=teacher.adaptive_score_scale_clip
        )
        gate = teacher.adaptive_user_gate[users].unsqueeze(1)
        scores = base_scores + (
            teacher.adaptive_residual_beta
            * gate
            * score_scale
            * residual_scores
        )
        scores[:, 0] = -torch.inf
        return scores

    def _clear_dynamic_teacher(self):
        teacher = self._ema_teacher
        object.__setattr__(teacher, "_dynamic_source_users", None)
        object.__setattr__(teacher, "_dynamic_target_items", None)
