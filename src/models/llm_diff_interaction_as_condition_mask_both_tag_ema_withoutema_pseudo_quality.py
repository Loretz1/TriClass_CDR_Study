import random

import numpy as np
import torch

from models.llm_diff_interaction_as_condition_mask_both_tag_ema_pseudo_quality import (
    PseudoQualityRecorderMixin,
)
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_withoutema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_withoutEMA as _AblationModel,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_withoutEMA_pseudo_quality(
    PseudoQualityRecorderMixin,
    _AblationModel,
):
    """Observer-only pseudo-quality recorder for the w/o EMA ablation."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        if self.quality_topk != int(self.config["top_k_pos"]):
            raise ValueError(
                "For exact w/o EMA reproduction, pseudo_quality_topk must equal "
                "top_k_pos so the recorder can read the canonical pseudo set."
            )
        expected_recall = config.get("pseudo_quality_expected_initial_recall")
        if expected_recall is not None:
            recall_key = f"recall@{self.quality_topk}"
            actual_recall = float(self.latest_quality_metrics[recall_key])
            tolerance = float(
                config.get("pseudo_quality_initial_recall_tolerance", 1e-8)
            )
            if abs(actual_recall - float(expected_recall)) > tolerance:
                raise ValueError(
                    f"Initial {recall_key}={actual_recall:.8f} does not match "
                    f"the LSDCDR curve start {float(expected_recall):.8f}."
                )

    @torch.no_grad()
    def rebuild_src_only_pseudo(self):
        # Keep the stochastic sampling and pseudo-set construction identical to
        # the non-recording ablation model.
        _AblationModel.rebuild_src_only_pseudo(self)

        # Recall@K depends on the selected set, not its internal order. Reading
        # that set avoids a second diffusion sample and leaves training unchanged.
        rng_state = _capture_rng_state()
        try:
            quality_topk = self.latest_quality_topk.copy()
            for user_id, row in self.quality_user_to_row.items():
                selected_items = self.src_only_pseudo_pos.get(user_id)
                if not selected_items:
                    continue
                quality_topk[row] = np.fromiter(
                    selected_items,
                    dtype=np.int32,
                    count=self.quality_topk,
                )

            self.rebuild_count += 1
            change_rate = self._topk_change_rate(
                self._prev_update_topk, quality_topk
            )
            self._prev_update_topk = quality_topk.copy()
            self.latest_quality_topk = quality_topk
            self.latest_quality_source = "online_rebuild_without_ema"
            self.latest_quality_metrics = self._evaluate_topk(quality_topk)
            self._append_metrics(
                self.update_metrics_path,
                event="online_rebuild_without_ema",
                epoch=self._quality_epoch_idx,
                global_step=self.global_step,
                source=self.latest_quality_source,
                metrics=self.latest_quality_metrics,
                change_rate=change_rate,
            )
        finally:
            _restore_rng_state(rng_state)
