from models.lsdcdr_gpt52_te3large import (
    LSDCDRRobustnessBase,
    embed_text_batches_openai,
    extract_src_only_tgt_only_modality_data,
    generate_src_only_tgt_only_embs,
    generate_src_only_tgt_only_final_embs,
)


FEWERTAGS_USER_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Summarize the user's cross-domain purchase interests as very few transferable concept-level tags.

Requirements:
1. Extract exactly 1-2 short tags that describe the strongest high-level preferences reusable across product domains.
2. Keep only the most central preference signals; ignore weaker or noisy evidence.
3. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
4. Output only a JSON list of strings, for example ["comfort focused"].
"""


FEWERTAGS_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Infer the very few strongest transferable user preferences this item may satisfy.

Requirements:
1. Extract exactly 1-2 short tags that describe high-level user preferences, not item facts.
2. Keep only the most central preference signals; ignore weaker or noisy evidence.
3. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
4. Output only a JSON list of strings, for example ["comfort focused"].
"""


class LSDCDR_Prompt_FewerTags(LSDCDRRobustnessBase):
    modality_name = "CrossDomain_semantics_both_tag_prompt_fewertags_src_only_tgt_only"


def extract_CrossDomain_semantics_both_tag_prompt_fewertags_src_only_tgt_only_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    return extract_src_only_tgt_only_modality_data(
        config=config,
        modality=modality,
        interaction=interaction,
        id_mapping=id_mapping,
        raw_data_list=raw_data_list,
        user_system_prompt=FEWERTAGS_USER_SYSTEM_PROMPT,
        item_system_prompt=FEWERTAGS_ITEM_SYSTEM_PROMPT,
    )


def generate_CrossDomain_semantics_both_tag_prompt_fewertags_src_only_tgt_only_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    return generate_src_only_tgt_only_embs(
        config=config,
        modality=modality,
        interaction=interaction,
        id_mapping=id_mapping,
        modality_data=modality_data,
        embed_text_batches=embed_text_batches_openai,
    )


def generate_CrossDomain_semantics_both_tag_prompt_fewertags_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
