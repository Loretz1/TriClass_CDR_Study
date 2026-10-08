from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_elec import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec as _BaseModel,
)
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_pseudo_quality import (
    PseudoQualityRecorderMixin,
)


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec_pseudo_quality(
    PseudoQualityRecorderMixin,
    _BaseModel,
):
    """Elec/Phone EMA LSDCDR with pseudo-label quality logging."""
