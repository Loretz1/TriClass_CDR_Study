from models.lsdcdr_gpt52_te3large import (
    BASE_TGT_ITEM_SYSTEM_PROMPT,
    BASE_USER_SYSTEM_PROMPT,
    LSDCDRRobustnessBase,
    extract_src_only_tgt_only_modality_data,
    generate_src_only_tgt_only_embs,
    generate_src_only_tgt_only_final_embs,
)
from models.lsdcdr_qwen3max_qwen3emb8b_retrieval import _embed_text_batches as embed_text_batches_dashscope


class LSDCDR_Qwen3Max_Qwen3Emb8B(LSDCDRRobustnessBase):
    modality_name = "CrossDomain_semantics_both_tag_qwen3max_qwen3emb8b_src_only_tgt_only"


def extract_CrossDomain_semantics_both_tag_qwen3max_qwen3emb8b_src_only_tgt_only_modality_data(
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
        user_system_prompt=BASE_USER_SYSTEM_PROMPT,
        item_system_prompt=BASE_TGT_ITEM_SYSTEM_PROMPT,
    )


def generate_CrossDomain_semantics_both_tag_qwen3max_qwen3emb8b_src_only_tgt_only_embs(
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
        embed_text_batches=embed_text_batches_dashscope,
    )


def generate_CrossDomain_semantics_both_tag_qwen3max_qwen3emb8b_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
