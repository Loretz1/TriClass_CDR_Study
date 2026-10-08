import logging

import numpy as np

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    CROSSDOMAIN_USER_SYSTEM_PROMPT,
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
    TGT_ITEM_SYSTEM_PROMPT,
    _build_all_user_prompt_strings,
    _build_item_prompt_string,
    _collect_domain_items,
    _collect_domain_user_items,
    _user_item_profile_to_string,
)
from models.lsdcdr_gpt52_te3large_retrieval import (
    _embed_text_batches as embed_text_batches_openai,
    _id2raw,
    _infer_num_overlap_users,
    _run_prompt_summarization_multiprocess,
)


logger = logging.getLogger()

BASE_USER_SYSTEM_PROMPT = CROSSDOMAIN_USER_SYSTEM_PROMPT
BASE_TGT_ITEM_SYSTEM_PROMPT = TGT_ITEM_SYSTEM_PROMPT


CONCISE_USER_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Summarize the user's cross-domain purchase interests as transferable concept-level tags.

Requirements:
1. Extract 2-4 short tags that describe high-level preferences reusable across product domains.
2. Focus on broad needs, values, lifestyle, comfort, reliability, style, or price sensitivity.
3. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
4. Output only a JSON list of strings, for example ["minimal style", "value focused"].
"""


CONCISE_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Infer what transferable user preferences this item may satisfy.

Requirements:
1. Extract 2-4 short tags that describe high-level user preferences, not item facts.
2. Focus on broad needs, values, lifestyle, comfort, reliability, style, or price sensitivity.
3. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
4. Output only a JSON list of strings, for example ["minimal style", "value focused"].
"""


FINEGRAINED_USER_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Summarize the user's cross-domain purchase interests as transferable concept-level tags.

Requirements:
1. Extract 4-6 short tags that describe high-level preferences reusable across product domains.
2. Cover as many of these aspects as the evidence supports:
   functional needs, style or aesthetic preference, quality or reliability expectation,
   usage or lifestyle orientation, and price or value sensitivity.
3. Keep tags abstract and preference-oriented.
4. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
5. Output only a JSON list of strings, for example ["durability focused", "active lifestyle"].
"""


FINEGRAINED_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Infer what transferable user preferences this item may satisfy.

Requirements:
1. Extract 4-6 short tags that describe high-level user preferences, not item facts.
2. Cover as many of these aspects as the evidence supports:
   functional needs, style or aesthetic preference, quality or reliability expectation,
   usage or lifestyle orientation, and price or value sensitivity.
3. Keep tags abstract and preference-oriented.
4. Do not mention product names, brands, specific item categories, concrete objects, or exact scenarios.
5. Output only a JSON list of strings, for example ["durability focused", "active lifestyle"].
"""


class _ModalityAliasDataloader:
    def __init__(self, dataloader, source_name, alias_name="CrossDomain_semantics_both_tag"):
        self._dataloader = dataloader
        self.source_name = source_name
        self.alias_name = alias_name

    def __getattr__(self, name):
        return getattr(self._dataloader, name)

    def get_modality_embs(self):
        modality_embs = dict(self._dataloader.get_modality_embs())
        if self.source_name not in modality_embs:
            raise KeyError(
                f"Required modality '{self.source_name}' was not loaded. "
                f"Available modalities: {list(modality_embs.keys())}"
            )
        modality_embs[self.alias_name] = modality_embs[self.source_name]
        return modality_embs


class LSDCDRRobustnessBase(LLM_Diff_interaction_as_condition_mask_both_tag_EMA):
    modality_name = None

    def __init__(self, config, dataloader):
        if self.modality_name is None:
            raise ValueError("Subclasses must define modality_name.")
        super().__init__(config, _ModalityAliasDataloader(dataloader, self.modality_name))


def _get_raw_reviews_metadata(config, raw_data_list):
    if config["dataset"] == "Amazon2014":
        reviews, metadata = raw_data_list
    elif config["dataset"] == "Douban":
        reviews = raw_data_list[0]
        metadata = raw_data_list[2]
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    return reviews, metadata


def _get_src_only_users(id_mapping):
    num_overlap = _infer_num_overlap_users(id_mapping)
    src_id2user = id_mapping["src"]["id2user"]
    num_src_user = len(src_id2user) - 1
    user_ids = list(range(num_overlap + 1, num_src_user + 1))
    raw_users = [_id2raw(src_id2user, uid) for uid in user_ids]
    return user_ids, raw_users


def extract_src_only_tgt_only_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
    user_system_prompt,
    item_system_prompt,
):
    if modality.get("generation_scope", "src_only_tgt_only") != "src_only_tgt_only":
        raise ValueError("This LSDCDR robustness modality expects generation_scope=src_only_tgt_only.")
    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for LLM tag generation.")

    reviews, metadata = _get_raw_reviews_metadata(config, raw_data_list)
    src_only_user_ids, src_only_raw_users = _get_src_only_users(id_mapping)

    user_item_profiles = {}
    _collect_domain_user_items(
        interaction_df=interaction["src"],
        domain="src",
        id_mapping=id_mapping,
        metadata=metadata,
        reviews=reviews,
        user_item_dict=user_item_profiles,
    )
    _collect_domain_user_items(
        interaction_df=interaction["tgt"],
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        reviews=reviews,
        user_item_dict=user_item_profiles,
    )

    item_profiles_tgt = {}
    _collect_domain_items(
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_tgt,
    )

    tgt_items = {
        _id2raw(id_mapping["tgt"]["id2item"], iid)
        for iid in range(1, len(id_mapping["tgt"]["id2item"]))
    }
    profile_tgt_items = set(item_profiles_tgt.keys())
    if profile_tgt_items != tgt_items:
        raise ValueError(f"Mismatch in tgt items: profile={len(profile_tgt_items)}, mapping={len(tgt_items)}")

    missing_src_only_users = [user for user in src_only_raw_users if user not in user_item_profiles]
    if missing_src_only_users:
        raise ValueError(
            f"Missing source-domain train profiles for {len(missing_src_only_users)} src-only users, "
            f"examples={missing_src_only_users[:5]}"
        )

    src_only_profiles = {user: user_item_profiles[user] for user in src_only_raw_users}
    user_profile_strings = {
        user_id: _user_item_profile_to_string(profile)
        for user_id, profile in src_only_profiles.items()
    }
    user_prompts = _build_all_user_prompt_strings(user_profile_strings)
    item_prompts_tgt = {
        item_id: _build_item_prompt_string(profile)
        for item_id, profile in item_profiles_tgt.items()
    }

    model_name = modality.get("llm_chat_model", "gpt-4o")
    max_retry = int(modality.get("max_retry", 2))
    num_workers = int(modality.get("num_workers", 8))
    timeout = float(modality.get("request_timeout", 60))
    openai_max_retries = int(modality.get("openai_max_retries", 0))
    base_url = modality.get("llm_base_url") or config.get("openai_base_url")

    logger.info(
        "[DATASET] Src-only/tgt-only semantic generation (%s): %d src-only users + %d target items. "
        "Overlap users, target-only users, and source items will be zero-filled.",
        modality["name"],
        len(user_prompts),
        len(item_prompts_tgt),
    )

    data_users, error_users = _run_prompt_summarization_multiprocess(
        prompts=user_prompts,
        system_prompt=user_system_prompt,
        model_name=model_name,
        api_key=config["openai_api_key"],
        base_url=base_url,
        max_retry=max_retry,
        num_workers=num_workers,
        timeout=timeout,
        openai_max_retries=openai_max_retries,
    )

    data_items_tgt, error_items_tgt = _run_prompt_summarization_multiprocess(
        prompts=item_prompts_tgt,
        system_prompt=item_system_prompt,
        model_name=model_name,
        api_key=config["openai_api_key"],
        base_url=base_url,
        max_retry=max_retry,
        num_workers=num_workers,
        timeout=timeout,
        openai_max_retries=openai_max_retries,
    )

    return {
        "llm_chat_model": model_name,
        "embedding_model": modality["emb_model"],
        "generation_scope": "src_only_tgt_only",
        "src_only_user_ids_src": src_only_user_ids,
        "src_only_raw_users": src_only_raw_users,
        "data_users": data_users,
        "error_users": error_users,
        "data_items_src": {},
        "error_items_src": {},
        "data_items_tgt": data_items_tgt,
        "error_items_tgt": error_items_tgt,
    }


def _clean_tag_list(tags, entity_id):
    if not isinstance(tags, list):
        raise ValueError(f"Semantic tags for {entity_id} must be a list.")
    cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
    if not cleaned:
        raise ValueError(f"Semantic tags for {entity_id} are empty.")
    return cleaned


def _mean_pool_tags(tag_lists, flat_embs):
    pooled = []
    offset = 0
    for tags in tag_lists:
        count = len(tags)
        pooled.append(flat_embs[offset:offset + count].mean(axis=0))
        offset += count
    return pooled


def generate_src_only_tgt_only_embs(config, modality, interaction, id_mapping, modality_data, embed_text_batches):
    if modality_data.get("generation_scope") != "src_only_tgt_only":
        raise ValueError("Expected src_only_tgt_only modality data.")
    if modality_data.get("error_users"):
        raise ValueError(f"There are {len(modality_data['error_users'])} failed user summaries.")
    if modality_data.get("error_items_tgt"):
        raise ValueError(f"There are {len(modality_data['error_items_tgt'])} failed target item summaries.")

    num_overlap = _infer_num_overlap_users(id_mapping)
    num_src_user = len(id_mapping["src"]["id2user"]) - 1
    num_tgt_user = len(id_mapping["tgt"]["id2user"]) - 1
    num_users = num_src_user + num_tgt_user - num_overlap
    num_items_src = len(id_mapping["src"]["id2item"]) - 1
    num_items_tgt = len(id_mapping["tgt"]["id2item"]) - 1
    total_rows = num_users + num_items_src + num_items_tgt

    src_only_raw_users = modality_data["src_only_raw_users"]
    tgt_raw_items = [_id2raw(id_mapping["tgt"]["id2item"], iid) for iid in range(1, num_items_tgt + 1)]

    user_tag_lists = []
    for raw_user in src_only_raw_users:
        if raw_user not in modality_data["data_users"]:
            raise ValueError(f"Missing semantic tags for src-only user {raw_user}")
        user_tag_lists.append(_clean_tag_list(modality_data["data_users"][raw_user], raw_user))

    tgt_item_tag_lists = []
    for raw_item in tgt_raw_items:
        if raw_item not in modality_data["data_items_tgt"]:
            raise ValueError(f"Missing semantic tags for target item {raw_item}")
        tgt_item_tag_lists.append(_clean_tag_list(modality_data["data_items_tgt"][raw_item], raw_item))

    all_texts = []
    for tags in user_tag_lists + tgt_item_tag_lists:
        all_texts.extend(tags)
    if not all_texts:
        raise ValueError("No semantic tag texts to embed.")

    logger.info(
        "[DATASET] Embedding src-only/tgt-only semantic texts (%s): %d tag texts "
        "from %d src-only users and %d target items.",
        modality["name"],
        len(all_texts),
        len(src_only_raw_users),
        len(tgt_raw_items),
    )
    flat_embs = embed_text_batches(config, modality, all_texts)

    num_user_tag_texts = sum(len(tags) for tags in user_tag_lists)
    user_embs = _mean_pool_tags(user_tag_lists, flat_embs[:num_user_tag_texts])
    tgt_item_embs = _mean_pool_tags(tgt_item_tag_lists, flat_embs[num_user_tag_texts:])

    full_embs = np.zeros((total_rows, flat_embs.shape[1]), dtype=np.float32)
    for raw_user, emb in zip(src_only_raw_users, user_embs):
        src_uid = int(id_mapping["src"]["user2id"][raw_user])
        full_embs[src_uid - 1] = emb

    target_item_start = num_users + num_items_src
    for item_idx, emb in enumerate(tgt_item_embs):
        full_embs[target_item_start + item_idx] = emb

    return full_embs


def generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs):
    input_dim = modality_embs.shape[1]
    target_dim = int(modality["emb_pca"])
    if input_dim != target_dim:
        raise ValueError(
            "Src-only/tgt-only LSDCDR robustness variants expect emb_pca to equal emb_dim. "
            f"Got emb_dim={input_dim}, emb_pca={target_dim}."
        )
    return modality_embs.astype(np.float32)


class LSDCDR_GPT52_TE3Large(LSDCDRRobustnessBase):
    modality_name = "CrossDomain_semantics_both_tag_gpt52_te3large_src_only_tgt_only"


def extract_CrossDomain_semantics_both_tag_gpt52_te3large_src_only_tgt_only_modality_data(
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


def generate_CrossDomain_semantics_both_tag_gpt52_te3large_src_only_tgt_only_embs(
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


def generate_CrossDomain_semantics_both_tag_gpt52_te3large_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
