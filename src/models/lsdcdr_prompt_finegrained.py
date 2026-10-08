from models.lsdcdr_gpt52_te3large import (
    FINEGRAINED_ITEM_SYSTEM_PROMPT,
    FINEGRAINED_USER_SYSTEM_PROMPT,
    LSDCDRRobustnessBase,
    embed_text_batches_openai,
    extract_src_only_tgt_only_modality_data,
    generate_src_only_tgt_only_embs,
    generate_src_only_tgt_only_final_embs,
)


class LSDCDR_Prompt_FineGrained(LSDCDRRobustnessBase):
    modality_name = "CrossDomain_semantics_both_tag_prompt_finegrained_src_only_tgt_only"


def extract_CrossDomain_semantics_both_tag_prompt_finegrained_src_only_tgt_only_modality_data(
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
        user_system_prompt=FINEGRAINED_USER_SYSTEM_PROMPT,
        item_system_prompt=FINEGRAINED_ITEM_SYSTEM_PROMPT,
    )


def generate_CrossDomain_semantics_both_tag_prompt_finegrained_src_only_tgt_only_embs(
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


def generate_CrossDomain_semantics_both_tag_prompt_finegrained_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
