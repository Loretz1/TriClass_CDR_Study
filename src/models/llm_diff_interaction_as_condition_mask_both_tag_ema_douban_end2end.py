import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

import numpy as np
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban import (
    CROSSDOMAIN_USER_SYSTEM_PROMPT,
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban,
    TGT_ITEM_SYSTEM_PROMPT,
    _build_all_user_prompt_strings,
    _build_item_prompt_string,
    _collect_domain_items,
    _collect_domain_user_items,
    _extract_string_list,
    _user_item_profile_to_string,
)


logger = logging.getLogger()

MODALITY_NAME = "CrossDomain_semantics_both_tag_douban_end2end_src_only_tgt_only"


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


def _id2raw(id2entity, idx):
    if isinstance(id2entity, dict):
        return id2entity[str(idx)]
    return id2entity[int(idx)]


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

    for retry_idx in range(max(1, max_retry)):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            last_response_text = text

            tags = _extract_string_list_robust(text)
            if tags is not None:
                return {
                    "user_id": key,
                    "summary": tags,
                    "success": True,
                }
        except Exception as e:
            last_response_text = f"{type(e).__name__}: {e}"
            if retry_idx + 1 < max(1, max_retry):
                time.sleep(min(2 ** retry_idx, 8))

    return {
        "user_id": key,
        "summary": last_response_text,
        "success": False,
    }


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
    if not prompts:
        return {}, {}
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    tasks = [
        (key, prompt, system_prompt, model_name, max_retry, api_key, base_url, timeout, openai_max_retries)
        for key, prompt in prompts.items()
    ]

    results = {}
    errors = {}
    with Pool(processes=max(1, int(num_workers))) as pool:
        for out in pool.imap_unordered(_process_single_prompt, tasks):
            key = out["user_id"]
            if out["success"]:
                results[key] = out["summary"]
            else:
                errors[key] = out["summary"]
    return results, errors


def run_until_no_prompt_errors(
    prompts,
    run_once_fn,
    label,
    max_rounds=0,
    sleep_seconds=0.0,
):
    if not prompts:
        return {}, {}

    pending = dict(prompts)
    success = {}
    round_idx = 0

    while pending:
        round_idx += 1
        logger.info(
            "[DATASET][REPAIR] %s round %d: processing %d pending prompts.",
            label,
            round_idx,
            len(pending),
        )
        ok, err = run_once_fn(pending)
        success.update(ok)

        missing = {key: pending[key] for key in pending.keys() if key not in ok and key not in err}
        pending = {key: prompts[key] for key in err.keys()}
        pending.update(missing)

        if not pending:
            logger.info(
                "[DATASET][REPAIR] %s completed after %d round(s); all prompts succeeded.",
                label,
                round_idx,
            )
            return success, {}

        logger.warning(
            "[DATASET][REPAIR] %s still has %d failed prompt(s) after round %d; retrying only failures.",
            label,
            len(pending),
            round_idx,
        )
        if max_rounds and round_idx >= max_rounds:
            raise RuntimeError(
                f"{label} still has {len(pending)} failed prompt(s) after {round_idx} repair rounds."
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return success, {}


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
        if count == 0:
            raise ValueError("Encountered an entity with zero semantic tags.")
        pooled.append(flat_embs[offset:offset + count].mean(axis=0))
        offset += count
    return pooled


def _embed_one_text_batch(args):
    (
        batch_idx,
        batch_texts,
        api_key,
        base_url,
        timeout,
        openai_max_retries,
        emb_model,
        normalize,
        embedding_max_retry,
    ) = args
    client = _make_openai_client(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=openai_max_retries,
    )

    last_error = None
    for retry_idx in range(max(1, embedding_max_retry)):
        try:
            response = client.embeddings.create(
                model=emb_model,
                input=batch_texts,
            )
            response_data = sorted(
                enumerate(response.data),
                key=lambda item: getattr(item[1], "index", item[0]),
            )

            batch_embs = []
            for _, emb_obj in response_data:
                emb = np.asarray(emb_obj.embedding, dtype=np.float32)
                if normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-12)
                batch_embs.append(emb)

            if len(batch_embs) != len(batch_texts):
                raise RuntimeError(
                    f"Embedding batch {batch_idx} returned {len(batch_embs)} vectors "
                    f"for {len(batch_texts)} texts."
                )
            return batch_idx, batch_embs
        except Exception as e:
            last_error = e
            if retry_idx + 1 < max(1, embedding_max_retry):
                time.sleep(min(2 ** retry_idx, 8))

    raise RuntimeError(
        f"Embedding batch {batch_idx} failed after {max(1, embedding_max_retry)} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _embed_text_batches(config, modality, texts):
    if "text-embedding-3" not in modality["emb_model"]:
        raise ValueError(f"Unsupported embedding model: {modality['emb_model']}")

    batch_size = int(modality["emb_batch_size"])
    normalize = modality.get("normalize_semantic_emb", False)
    timeout = float(modality.get("embedding_request_timeout", modality.get("request_timeout", 60)))
    openai_max_retries = int(modality.get("openai_max_retries", 0))
    embedding_num_workers = int(modality.get("embedding_num_workers", 1))
    embedding_max_retry = int(modality.get("embedding_max_retry", 3))

    tasks = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        tasks.append(
            (
                len(tasks),
                texts[start:end],
                config["openai_api_key"],
                config.get("openai_base_url"),
                timeout,
                openai_max_retries,
                modality["emb_model"],
                normalize,
                embedding_max_retry,
            )
        )

    logger.info(
        "[DATASET] Embedding API requests: %d batches, batch_size=%d, workers=%d, timeout=%s, max_retry=%d.",
        len(tasks),
        batch_size,
        embedding_num_workers,
        timeout,
        embedding_max_retry,
    )

    if embedding_num_workers <= 1 or len(tasks) <= 1:
        outputs = [_embed_one_text_batch(task) for task in tasks]
    else:
        outputs = []
        with ThreadPoolExecutor(max_workers=min(embedding_num_workers, len(tasks))) as executor:
            futures = [executor.submit(_embed_one_text_batch, task) for task in tasks]
            for future in as_completed(futures):
                outputs.append(future.result())

    flat_embs = []
    for _, batch_embs in sorted(outputs, key=lambda item: item[0]):
        flat_embs.extend(batch_embs)

    return np.stack(flat_embs, axis=0)


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
            "LSDCDR Douban End2End expects emb_pca to equal emb_dim for zero-filled semantic embeddings. "
            f"Got emb_dim={input_dim}, emb_pca={target_dim}."
        )
    return modality_embs.astype(np.float32)


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban_End2End(
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban
):
    """LSDCDR Douban variant that only generates semantic emb needed for pseudo initialization."""

    def __init__(self, config, dataloader):
        super().__init__(config, _ModalityAliasDataloader(dataloader, MODALITY_NAME))


def _get_src_only_users(id_mapping):
    num_overlap = _infer_num_overlap_users(id_mapping)
    src_id2user = id_mapping["src"]["id2user"]
    num_src_user = len(src_id2user) - 1
    user_ids = list(range(num_overlap + 1, num_src_user + 1))
    raw_users = [_id2raw(src_id2user, uid) for uid in user_ids]
    return user_ids, raw_users


def extract_CrossDomain_semantics_both_tag_douban_end2end_src_only_tgt_only_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    if config["dataset"] != "Douban":
        raise ValueError("This end-to-end LSDCDR variant is intended for Douban.")
    if modality.get("generation_scope", "src_only_tgt_only") != "src_only_tgt_only":
        raise ValueError("Expected generation_scope=src_only_tgt_only.")
    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for LLM tag generation.")

    reviews = raw_data_list[0]
    metadata = raw_data_list[2]
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

    missing_users = [raw_user for raw_user in src_only_raw_users if raw_user not in user_item_profiles]
    if missing_users:
        raise ValueError(
            f"Missing source-domain train profiles for {len(missing_users)} src-only users, "
            f"examples={missing_users[:5]}"
        )

    src_only_profiles = {raw_user: user_item_profiles[raw_user] for raw_user in src_only_raw_users}
    user_profile_strings = {
        raw_user: _user_item_profile_to_string(profile)
        for raw_user, profile in src_only_profiles.items()
    }
    user_prompts = _build_all_user_prompt_strings(user_profile_strings)

    item_profiles_tgt = {}
    _collect_domain_items(
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_tgt,
    )
    item_prompts_tgt = {
        raw_item: _build_item_prompt_string(profile)
        for raw_item, profile in item_profiles_tgt.items()
    }

    model_name = modality.get("llm_chat_model", "gpt-4o-mini")
    max_retry = int(modality.get("max_retry", 2))
    profile_num_workers = int(modality.get("profile_num_workers", modality.get("num_workers", 20)))
    timeout = float(modality.get("request_timeout", 60))
    openai_max_retries = int(modality.get("openai_max_retries", 0))
    base_url = modality.get("llm_base_url") or config.get("openai_base_url")

    logger.info(
        "[DATASET] LSDCDR Douban End2End semantic generation (%s): "
        "%d src-only users + %d target items. "
        "Overlap users, target-only users, and source items will be zero-filled.",
        modality["name"],
        len(user_prompts),
        len(item_prompts_tgt),
    )
    logger.info(
        "[DATASET] Profile/chat workers=%d, embedding workers=%d, embedding batch size=%d.",
        profile_num_workers,
        int(modality.get("embedding_num_workers", 1)),
        int(modality.get("emb_batch_size", 1)),
    )

    repair_max_rounds = int(modality.get("repair_max_rounds", 0))
    repair_sleep_seconds = float(modality.get("repair_sleep_seconds", 0.0))

    data_users, error_users = run_until_no_prompt_errors(
        prompts=user_prompts,
        run_once_fn=lambda pending: _run_prompt_summarization_multiprocess(
            prompts=pending,
            system_prompt=CROSSDOMAIN_USER_SYSTEM_PROMPT,
            model_name=model_name,
            api_key=config["openai_api_key"],
            base_url=base_url,
            max_retry=max_retry,
            num_workers=profile_num_workers,
            timeout=timeout,
            openai_max_retries=openai_max_retries,
        ),
        label=f"{modality['name']} src-only user profiles",
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )

    data_items_tgt, error_items_tgt = run_until_no_prompt_errors(
        prompts=item_prompts_tgt,
        run_once_fn=lambda pending: _run_prompt_summarization_multiprocess(
            prompts=pending,
            system_prompt=TGT_ITEM_SYSTEM_PROMPT,
            model_name=model_name,
            api_key=config["openai_api_key"],
            base_url=base_url,
            max_retry=max_retry,
            num_workers=profile_num_workers,
            timeout=timeout,
            openai_max_retries=openai_max_retries,
        ),
        label=f"{modality['name']} target item profiles",
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
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


def generate_CrossDomain_semantics_both_tag_douban_end2end_src_only_tgt_only_embs(
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
        embed_text_batches=_embed_text_batches,
    )


def generate_CrossDomain_semantics_both_tag_douban_end2end_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
