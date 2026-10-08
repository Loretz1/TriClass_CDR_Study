import json
import logging
import os
import re
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    CROSSDOMAIN_USER_SYSTEM_PROMPT,
    TGT_ITEM_SYSTEM_PROMPT,
    _build_all_user_prompt_strings,
    _build_item_prompt_string,
    _collect_domain_items,
    _collect_domain_user_items,
    _extract_string_list,
    _user_item_profile_to_string,
)


logger = logging.getLogger()


class LSDCDR_Gemini3Pro_GeminiEmb001_Retrieval(GeneralRecommender):
    """Cold-only semantic retrieval baseline with Gemini 3 Pro + Gemini Embedding."""

    modality_name = "CrossDomain_semantics_both_tag_gemini3pro_geminiemb001_cold_only"

    def __init__(self, config, dataloader):
        super(LSDCDR_Gemini3Pro_GeminiEmb001_Retrieval, self).__init__(config, dataloader)

        semantic = dataloader.get_modality_embs()[self.modality_name]
        semantic = torch.from_numpy(semantic).float()

        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        expected_rows = num_users + self.num_items_src + self.num_items_tgt
        if semantic.size(0) < expected_rows:
            raise ValueError(
                f"{self.modality_name} has {semantic.size(0)} rows, "
                f"but at least {expected_rows} rows are required."
            )

        user = semantic[:num_users]
        tgt_item = semantic[-self.num_items_tgt:]

        self.register_buffer("user_semantic_emb", self._with_padding(user))
        self.register_buffer("tgt_item_semantic_emb", self._with_padding(tgt_item))
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _with_padding(emb):
        pad = torch.zeros(1, emb.size(1), dtype=emb.dtype)
        return torch.cat([pad, emb], dim=0)

    def _to_global_user_id(self, users, is_warm):
        if not is_warm:
            return users

        offset = self.num_users_src - self.num_users_overlap
        return torch.where(users <= self.num_users_overlap, users, users + offset)

    def calculate_loss(self, interaction, epoch_idx):
        return self.dummy_param.sum() * 0.0

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        users_global = self._to_global_user_id(users, is_warm=is_warm)

        user_emb = self.user_semantic_emb[users_global]
        scores = torch.matmul(user_emb, self.tgt_item_semantic_emb.t())
        scores[:, 0] = 0.0
        return scores


def _extract_string_list_robust(text):
    tags = _extract_string_list(text)
    if tags is not None:
        return tags

    match = re.search(r"\[[\s\S]*\]", text or "")
    if match is None:
        return None
    return _extract_string_list(match.group(0))


def _make_openai_client(api_key, base_url, timeout, max_retries):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        timeout=timeout,
        max_retries=max_retries,
    )


def _process_single_prompt(args):
    key, prompt, system_prompt, model_name, max_retry, api_key, base_url, timeout, openai_max_retries = args

    client = _make_openai_client(api_key, base_url, timeout, openai_max_retries)
    last_response_text = None

    for retry_idx in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            last_response_text = text

            tags = _extract_string_list_robust(text)
            if tags is not None:
                return {"user_id": key, "summary": tags, "success": True}
        except Exception as e:
            last_response_text = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** retry_idx, 8))

    return {"user_id": key, "summary": last_response_text, "success": False}


def _run_prompt_summarization_multiprocess(
    prompts,
    system_prompt,
    model_name,
    api_key,
    base_url,
    max_retry=2,
    num_workers=None,
    timeout=60,
    openai_max_retries=0,
):
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    tasks = [
        (key, prompt, system_prompt, model_name, max_retry, api_key, base_url, timeout, openai_max_retries)
        for key, prompt in prompts.items()
    ]

    results = {}
    errors = {}
    with Pool(processes=num_workers) as pool:
        for out in pool.imap_unordered(_process_single_prompt, tasks):
            key = out["user_id"]
            if out["success"]:
                results[key] = out["summary"]
            else:
                errors[key] = out["summary"]
    return results, errors


def _format_split_value(value):
    return str(value)


def _get_joint_path(config):
    data_path = config.get("data_path", "../data/")
    joint_dataset_name = "+".join(config["domains"])
    dataset_type = "only_overlap_users" if config["only_overlap_users"] else "all_users"
    split_dir = (
        f"WarmValid{_format_split_value(config['warm_valid_ratio'])}_"
        f"WarmTest{_format_split_value(config['warm_test_ratio'])}_"
        f"ColdValid{_format_split_value(config['t_cold_valid'])}_"
        f"ColdTest{_format_split_value(config['t_cold_test'])}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_dir += f"_{config['k_cores']}cores"

    return os.path.join(
        data_path,
        config["dataset"],
        joint_dataset_name,
        dataset_type,
        split_dir,
    )


def _id2raw(id2entity, idx):
    if isinstance(id2entity, dict):
        return id2entity[str(idx)]
    return id2entity[int(idx)]


def _get_cold_eval_users(config, id_mapping):
    all_users_path = os.path.join(_get_joint_path(config), "all_users.json")
    if not os.path.exists(all_users_path):
        raise FileNotFoundError(f"Cannot find cold user split file: {all_users_path}")

    with open(all_users_path, "r") as f:
        all_users = json.load(f)

    cold_user_ids = list(all_users.get("valid_cold_users", [])) + list(all_users.get("test_cold_users", []))
    src_id2user = id_mapping["src"]["id2user"]
    cold_raw_users = [_id2raw(src_id2user, uid) for uid in cold_user_ids]
    return cold_user_ids, cold_raw_users


def _infer_num_overlap_users(id_mapping):
    src_id2user = id_mapping["src"]["id2user"]
    tgt_id2user = id_mapping["tgt"]["id2user"]
    max_check = min(len(src_id2user), len(tgt_id2user))

    num_overlap = 0
    for idx in range(1, max_check):
        if _id2raw(src_id2user, idx) == _id2raw(tgt_id2user, idx):
            num_overlap += 1
        else:
            break
    return num_overlap


def _raw_source_user_to_global_row(raw_user, id_mapping):
    src_uid = id_mapping["src"]["user2id"][raw_user]
    return int(src_uid) - 1


def _mean_pool_tags(tag_lists, flat_embs):
    pooled = []
    offset = 0
    for tags in tag_lists:
        count = len(tags)
        if count == 0:
            raise ValueError("Encountered an entity with zero semantic tags.")
        pooled.append(flat_embs[offset:offset + count].mean(axis=0))
        offset += count
    return pooled


def _embed_text_batches(config, modality, texts):
    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for Yunwu/OpenAI-compatible Gemini embeddings.")

    client = _make_openai_client(
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
        timeout=float(modality.get("embedding_request_timeout", modality.get("request_timeout", 60))),
        max_retries=int(modality.get("openai_max_retries", 0)),
    )

    embedding_model = modality["emb_model"]
    batch_size = int(modality["emb_batch_size"])
    normalize = modality.get("normalize_semantic_emb", False)
    force_dimensions = bool(modality.get("force_embedding_dimensions", False))

    flat_embs = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        kwargs = {
            "model": embedding_model,
            "input": texts[start:end],
        }
        if force_dimensions:
            kwargs["dimensions"] = int(modality["emb_dim"])
        response = client.embeddings.create(**kwargs)

        for emb_obj in response.data:
            emb = np.asarray(emb_obj.embedding, dtype=np.float32)
            if normalize:
                emb = emb / (np.linalg.norm(emb) + 1e-12)
            flat_embs.append(emb)

    flat_embs = np.stack(flat_embs, axis=0)
    expected_dim = int(modality["emb_dim"])
    if flat_embs.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch for {embedding_model}: got {flat_embs.shape[1]}, "
            f"expected {expected_dim}. If Yunwu supports dimensions for this model, "
            "set force_embedding_dimensions: true."
        )
    return flat_embs


def extract_CrossDomain_semantics_both_tag_gemini3pro_geminiemb001_cold_only_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    if config["dataset"] == "Amazon2014":
        reviews, metadata = raw_data_list
    elif config["dataset"] == "Douban":
        reviews = raw_data_list[0]
        metadata = raw_data_list[2]
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    if modality.get("generation_scope", "cold_eval_tgt_only") != "cold_eval_tgt_only":
        raise ValueError(
            "LSDCDR_Gemini3Pro_GeminiEmb001_Retrieval only supports "
            "generation_scope=cold_eval_tgt_only."
        )

    cold_user_ids, cold_raw_users = _get_cold_eval_users(config, id_mapping)

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

    tgt_items = set(id_mapping["tgt"]["id2item"][1:])
    profile_tgt_items = set(item_profiles_tgt.keys())
    assert profile_tgt_items == tgt_items, \
        f"Mismatch in tgt items: profile={len(profile_tgt_items)}, mapping={len(tgt_items)}"

    missing_cold_users = [user for user in cold_raw_users if user not in user_item_profiles]
    if missing_cold_users:
        raise ValueError(
            f"Missing source-domain train profiles for {len(missing_cold_users)} cold users, "
            f"examples={missing_cold_users[:5]}"
        )

    cold_user_profiles = {user: user_item_profiles[user] for user in cold_raw_users}
    user_profile_strings = {
        user_id: _user_item_profile_to_string(profile)
        for user_id, profile in cold_user_profiles.items()
    }
    item_prompts_tgt = {
        item_id: _build_item_prompt_string(profile)
        for item_id, profile in item_profiles_tgt.items()
    }
    user_prompts = _build_all_user_prompt_strings(user_profile_strings)

    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for Yunwu/OpenAI-compatible Gemini chat.")

    model_name = modality.get("llm_chat_model", "gemini-3-pro-preview")
    max_retry = int(modality.get("max_retry", 2))
    num_workers = int(modality.get("num_workers", 8))
    timeout = float(modality.get("request_timeout", 60))
    openai_max_retries = int(modality.get("openai_max_retries", 0))

    logger.info(
        "[DATASET] Yunwu/OpenAI-compatible Gemini cold-only semantic generation: "
        "%d cold users + %d target items. Source items and non-eval users are skipped.",
        len(user_prompts),
        len(item_prompts_tgt),
    )

    results, error_users = _run_prompt_summarization_multiprocess(
        prompts=user_prompts,
        system_prompt=CROSSDOMAIN_USER_SYSTEM_PROMPT,
        model_name=model_name,
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
        max_retry=max_retry,
        num_workers=num_workers,
        timeout=timeout,
        openai_max_retries=openai_max_retries,
    )

    data_items_tgt, error_items_tgt = _run_prompt_summarization_multiprocess(
        prompts=item_prompts_tgt,
        system_prompt=TGT_ITEM_SYSTEM_PROMPT,
        model_name=model_name,
        api_key=config["openai_api_key"],
        base_url=config.get("openai_base_url"),
        max_retry=max_retry,
        num_workers=num_workers,
        timeout=timeout,
        openai_max_retries=openai_max_retries,
    )

    return {
        "llm_chat_model": model_name,
        "embedding_model": modality["emb_model"],
        "generation_scope": "cold_eval_tgt_only",
        "cold_eval_user_ids_src": cold_user_ids,
        "cold_eval_raw_users": cold_raw_users,
        "data_users": results,
        "error_users": error_users,
        "data_items_src": {},
        "error_items_src": {},
        "data_items_tgt": data_items_tgt,
        "error_items_tgt": error_items_tgt,
    }


def generate_CrossDomain_semantics_both_tag_gemini3pro_geminiemb001_cold_only_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    if modality_data.get("generation_scope") != "cold_eval_tgt_only":
        raise ValueError("Expected cold_eval_tgt_only modality data.")
    if modality_data.get("error_users"):
        raise ValueError(f"There are {len(modality_data['error_users'])} failed cold user summaries.")
    if modality_data.get("error_items_tgt"):
        raise ValueError(f"There are {len(modality_data['error_items_tgt'])} failed target item summaries.")

    num_overlap = _infer_num_overlap_users(id_mapping)
    num_src_user = len(id_mapping["src"]["id2user"]) - 1
    num_tgt_user = len(id_mapping["tgt"]["id2user"]) - 1
    num_users = num_src_user + num_tgt_user - num_overlap
    num_items_src = len(id_mapping["src"]["id2item"]) - 1
    num_items_tgt = len(id_mapping["tgt"]["id2item"]) - 1
    total_rows = num_users + num_items_src + num_items_tgt

    cold_raw_users = modality_data["cold_eval_raw_users"]
    tgt_raw_items = id_mapping["tgt"]["id2item"][1:]

    user_tag_lists = []
    for raw_user in cold_raw_users:
        if raw_user not in modality_data["data_users"]:
            raise ValueError(f"Missing semantic tags for cold user {raw_user}")
        user_tag_lists.append(modality_data["data_users"][raw_user])

    tgt_item_tag_lists = []
    for raw_item in tgt_raw_items:
        if raw_item not in modality_data["data_items_tgt"]:
            raise ValueError(f"Missing semantic tags for target item {raw_item}")
        tgt_item_tag_lists.append(modality_data["data_items_tgt"][raw_item])

    all_texts = []
    for tags in user_tag_lists + tgt_item_tag_lists:
        all_texts.extend(tags)
    if not all_texts:
        raise ValueError("No cold-only semantic texts to embed.")

    logger.info(
        "[DATASET] Gemini embedding cold-only semantic texts: %d tag texts from %d cold users and %d target items.",
        len(all_texts),
        len(cold_raw_users),
        len(tgt_raw_items),
    )
    flat_embs = _embed_text_batches(config, modality, all_texts)

    num_user_tag_texts = sum(len(tags) for tags in user_tag_lists)
    user_embs = _mean_pool_tags(user_tag_lists, flat_embs[:num_user_tag_texts])
    tgt_item_embs = _mean_pool_tags(tgt_item_tag_lists, flat_embs[num_user_tag_texts:])

    full_embs = np.zeros((total_rows, flat_embs.shape[1]), dtype=np.float32)
    for raw_user, emb in zip(cold_raw_users, user_embs):
        full_embs[_raw_source_user_to_global_row(raw_user, id_mapping)] = emb

    target_item_start = num_users + num_items_src
    for item_idx, emb in enumerate(tgt_item_embs):
        full_embs[target_item_start + item_idx] = emb

    return full_embs


def generate_CrossDomain_semantics_both_tag_gemini3pro_geminiemb001_cold_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    input_dim = modality_embs.shape[1]
    target_dim = int(modality["emb_pca"])
    if input_dim != target_dim:
        raise ValueError(
            "Cold-only semantic retrieval expects emb_pca to equal emb_dim. "
            f"Got emb_dim={input_dim}, emb_pca={target_dim}."
        )
    return modality_embs.astype(np.float32)
