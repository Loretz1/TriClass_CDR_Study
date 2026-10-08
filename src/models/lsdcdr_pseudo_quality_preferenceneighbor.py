from models.lsdcdr_pseudo_quality_overlaponly import (
    LSDCDR_Pseudo_Quality_Base,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


class LSDCDR_Pseudo_Quality_PreferenceNeighbor(LSDCDR_Pseudo_Quality_Base):
    pseudo_variant = "preference_neighbor"
