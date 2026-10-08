from models.lsdcdr_gpt52_te3large import (
    LSDCDRRobustnessBase,
    embed_text_batches_openai,
    extract_src_only_tgt_only_modality_data,
    generate_src_only_tgt_only_embs,
    generate_src_only_tgt_only_final_embs,
)


PARAGRAPH_USER_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Summarize the user's cross-domain purchase interests as one transferable semantic paragraph.

Requirements:
1. Do not output concept-level tags or bullet points.
2. Write one concise paragraph that describes the user's high-level transferable preferences.
3. Focus on broad needs, values, lifestyle, comfort, reliability, style, usage orientation, or price sensitivity.
4. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
5. Output only a JSON list with exactly one string, for example ["The user prefers durable, practical, and comfort-oriented products with reliable everyday value."].
"""


PARAGRAPH_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Infer what transferable user preferences this item may satisfy as one semantic paragraph.

Requirements:
1. Do not output concept-level tags or bullet points.
2. Write one concise paragraph that describes high-level user preferences, not item facts.
3. Focus on broad needs, values, lifestyle, comfort, reliability, style, usage orientation, or price sensitivity.
4. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
5. Output only a JSON list with exactly one string, for example ["This item may appeal to users who prefer durable, practical, and comfort-oriented products with reliable everyday value."].
"""


class LSDCDR_Prompt_Paragraph(LSDCDRRobustnessBase):
    modality_name = "CrossDomain_semantics_both_tag_prompt_paragraph_src_only_tgt_only"


def extract_CrossDomain_semantics_both_tag_prompt_paragraph_src_only_tgt_only_modality_data(
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
        user_system_prompt=PARAGRAPH_USER_SYSTEM_PROMPT,
        item_system_prompt=PARAGRAPH_ITEM_SYSTEM_PROMPT,
    )


def generate_CrossDomain_semantics_both_tag_prompt_paragraph_src_only_tgt_only_embs(
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


def generate_CrossDomain_semantics_both_tag_prompt_paragraph_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
