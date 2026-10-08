import random

import numpy as np
import torch

from models.llm_diff_interaction_as_condition_mask_both_tag_ema_pseudo_quality import (
    PseudoQualityRecorderMixin,
)
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_withoutoverlapanchor import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_WithoutOverlapAnchor as _AblationModel,
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


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_WithoutOverlapAnchor_pseudo_quality(
    PseudoQualityRecorderMixin,
    _AblationModel,
):
    """Observer-only pseudo-quality variant of w/o Overlap-anchor."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        if self.quality_topk > int(self.config["top_k_pos"]):
            raise ValueError(
                "The observer-only recorder requires pseudo_quality_topk <= top_k_pos."
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
    def rebuild_src_only_pseudo_with_ema(self):
        # This is the only stochastic operation in the method. It is exactly the
        # same canonical rebuild used by the non-recording ablation model.
        _AblationModel.rebuild_src_only_pseudo_with_ema(self)

        # Recording is isolated from the training RNG stream. The rankings are
        # read from the canonical rebuild rather than recomputed.
        rng_state = _capture_rng_state()
        try:
            quality_topk = self.latest_quality_topk.copy()
            for user_id, row in self.quality_user_to_row.items():
                ranked_items = self._last_rebuild_rankings.get(user_id)
                if ranked_items is None:
                    continue
                quality_topk[row, : self.quality_topk] = np.asarray(
                    ranked_items[: self.quality_topk], dtype=np.int32
                )

            self.rebuild_count += 1
            change_rate = self._topk_change_rate(
                self._prev_update_topk, quality_topk
            )
            self._prev_update_topk = quality_topk.copy()
            self.latest_quality_topk = quality_topk
            self.latest_quality_source = "ema_rebuild"
            self.latest_quality_metrics = self._evaluate_topk(quality_topk)
            self._append_metrics(
                self.update_metrics_path,
                event="ema_rebuild",
                epoch=self._quality_epoch_idx,
                global_step=self.global_step,
                source=self.latest_quality_source,
                metrics=self.latest_quality_metrics,
                change_rate=change_rate,
            )
        finally:
            _restore_rng_state(rng_state)
