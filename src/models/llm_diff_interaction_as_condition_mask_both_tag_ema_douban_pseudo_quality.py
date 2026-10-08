from models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban as _BaseModel,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_pseudo_quality import (
    PseudoQualityRecorderMixin,
)


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban_pseudo_quality(
    PseudoQualityRecorderMixin,
    _BaseModel,
):
    """Douban EMA LSDCDR with pseudo-label quality logging."""
