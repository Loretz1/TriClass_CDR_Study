import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
    _build_item_prompt_string,
    _collect_domain_items,
    _collect_domain_user_items,
    _user_item_profile_to_string,
)
from models.lsdcdr_gpt52_te3large import _ModalityAliasDataloader
from models.lsdcdr_gpt52_te3large_retrieval import (
    _embed_one_text_batch,
    _embed_text_batches as embed_text_batches_openai,
    _id2raw,
    _infer_num_overlap_users,
)


logger = logging.getLogger()


MODALITY_NAME = "CrossDomain_semantics_both_tag_withoutllm"


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_withoutLLM(
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA
):
    """LSDCDR w/o LLM summarization.

    This variant keeps the original LSDCDR diffusion, EMA, self-evolving pseudo
    interactions, and training objective unchanged. The only change is the
    semantic embedding source: raw user/item profiles are directly embedded by
    the text embedding model instead of first being summarized into LLM tags.
    """

    modality_name = MODALITY_NAME

    def __init__(self, config, dataloader):
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


def _truncate_text(text, max_chars):
    if max_chars is None or int(max_chars) <= 0:
        return text
    max_chars = int(max_chars)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _build_direct_user_text(profile_string):
    return (
        "SOURCE_DOMAIN_INTERACTIONS:\n"
        f"{profile_string['src']}\n\n"
        "TARGET_DOMAIN_INTERACTIONS:\n"
        f"{profile_string['tgt']}"
    )


def _pack_embedding_tasks(texts, max_batch_size, max_chars_per_request):
    tasks = []
    current = []
    current_chars = 0

    for text in texts:
        text_chars = len(text)
        should_flush = (
            current
            and (
                len(current) >= max_batch_size
                or current_chars + text_chars > max_chars_per_request
            )
        )
        if should_flush:
            tasks.append(current)
            current = []
            current_chars = 0

        current.append(text)
        current_chars += text_chars

    if current:
        tasks.append(current)
    return tasks


def _dataset_path(config):
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_dir += f"_{config['k_cores']}cores"
        user_dir = "only_overlap_users"
    else:
        user_dir = "all_users"
    return os.path.join(
        config["data_path"],
        config["dataset"],
        "+".join(config["domains"]),
        user_dir,
        split_dir,
    )


def _embedding_cache_dir(config, modality, texts, max_batch_size, max_chars_per_request):
    signature = hashlib.sha1()
    signature.update(str(modality["emb_model"]).encode("utf-8"))
    signature.update(str(modality.get("emb_dim", "")).encode("utf-8"))
    signature.update(str(max_batch_size).encode("utf-8"))
    signature.update(str(max_chars_per_request).encode("utf-8"))
    signature.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        encoded = str(text).encode("utf-8", errors="ignore")
        signature.update(str(len(encoded)).encode("utf-8"))
        signature.update(encoded[:256])
        signature.update(encoded[-256:])
    cache_name = (
        f".{modality['name']}_{modality['emb_model']}_{modality.get('emb_dim', '')}_"
        f"{signature.hexdigest()[:16]}_batch_cache"
    )
    return os.path.join(_dataset_path(config), "modality_emb", cache_name)


def _load_cached_embedding_batch(cache_path, batch_idx, expected_size):
    if not os.path.exists(cache_path):
        return None
    arr = np.load(cache_path)
    if arr.shape[0] != expected_size:
        logger.warning(
            "[DATASET] Ignore stale without-LLM embedding batch cache %s: "
            "expected %d rows, got %d.",
            cache_path,
            expected_size,
            arr.shape[0],
        )
        return None
    return batch_idx, [arr[i].astype(np.float32, copy=False) for i in range(arr.shape[0])]


def _run_embedding_batch_with_cache(task, cache_path):
    cached = _load_cached_embedding_batch(cache_path, task[0], len(task[1]))
    if cached is not None:
        return cached

    batch_idx, batch_embs = _embed_one_text_batch(task)
    arr = np.asarray(batch_embs, dtype=np.float32)
    tmp_path = f"{cache_path}.tmp"
    np.save(tmp_path, arr)
    os.replace(f"{tmp_path}.npy", cache_path)
    return batch_idx, [arr[i] for i in range(arr.shape[0])]


def _embed_text_batches_withoutllm(config, modality, texts):
    if "text-embedding-3" not in modality["emb_model"]:
        return embed_text_batches_openai(config, modality, texts)

    max_batch_size = int(modality["emb_batch_size"])
    max_chars_per_request = int(modality.get("max_chars_per_embedding_request", 120000))
    normalize = modality.get("normalize_semantic_emb", False)
    timeout = float(modality.get("embedding_request_timeout", modality.get("request_timeout", 60)))
    openai_max_retries = int(modality.get("openai_max_retries", 0))
    embedding_num_workers = int(modality.get("embedding_num_workers", 1))
    embedding_max_retry = int(modality.get("embedding_max_retry", 3))

    packed_batches = _pack_embedding_tasks(texts, max_batch_size, max_chars_per_request)
    cache_enabled = bool(modality.get("embedding_batch_cache", True))
    cache_dir = None
    if cache_enabled:
        cache_dir = _embedding_cache_dir(config, modality, texts, max_batch_size, max_chars_per_request)
        os.makedirs(cache_dir, exist_ok=True)

    tasks = [
        (
            batch_idx,
            batch_texts,
            config["openai_api_key"],
            config.get("openai_base_url"),
            timeout,
            openai_max_retries,
            modality["emb_model"],
            normalize,
            embedding_max_retry,
        )
        for batch_idx, batch_texts in enumerate(packed_batches)
    ]
    cache_paths = [
        os.path.join(cache_dir, f"batch_{batch_idx:06d}.npy") if cache_dir else None
        for batch_idx in range(len(tasks))
    ]
    cached_count = sum(1 for path in cache_paths if path and os.path.exists(path))

    logger.info(
        "[DATASET] Without-LLM embedding API requests: %d batches, max_batch_size=%d, "
        "max_chars_per_request=%d, workers=%d, timeout=%s, max_retry=%d, cached_batches=%d.",
        len(tasks),
        max_batch_size,
        max_chars_per_request,
        embedding_num_workers,
        timeout,
        embedding_max_retry,
        cached_count,
    )
    if cache_dir:
        logger.info("[DATASET] Without-LLM embedding batch cache: %s", cache_dir)

    if embedding_num_workers <= 1 or len(tasks) <= 1:
        outputs = [
            _run_embedding_batch_with_cache(task, cache_path) if cache_path else _embed_one_text_batch(task)
            for task, cache_path in zip(tasks, cache_paths)
        ]
    else:
        outputs = []
        with ThreadPoolExecutor(max_workers=min(embedding_num_workers, len(tasks))) as executor:
            futures = [
                executor.submit(
                    _run_embedding_batch_with_cache,
                    task,
                    cache_path,
                ) if cache_path else executor.submit(_embed_one_text_batch, task)
                for task, cache_path in zip(tasks, cache_paths)
            ]
            for future in as_completed(futures):
                outputs.append(future.result())

    flat_embs = []
    for _, batch_embs in sorted(outputs, key=lambda item: item[0]):
        flat_embs.extend(batch_embs)

    return np.stack(flat_embs, axis=0)


def _ordered_raw_users(id_mapping):
    num_overlap = _infer_num_overlap_users(id_mapping)
    src_id2user = id_mapping["src"]["id2user"]
    tgt_id2user = id_mapping["tgt"]["id2user"]
    num_src_user = len(src_id2user) - 1
    num_tgt_user = len(tgt_id2user) - 1

    users = []
    for uid in range(1, num_overlap + 1):
        users.append(_id2raw(src_id2user, uid))
    for uid in range(num_overlap + 1, num_src_user + 1):
        users.append(_id2raw(src_id2user, uid))
    for uid in range(num_overlap + 1, num_tgt_user + 1):
        users.append(_id2raw(tgt_id2user, uid))
    return users


def extract_CrossDomain_semantics_both_tag_withoutllm_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    reviews, metadata = _get_raw_reviews_metadata(config, raw_data_list)

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

    item_profiles_src = {}
    item_profiles_tgt = {}
    _collect_domain_items(
        domain="src",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_src,
    )
    _collect_domain_items(
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_tgt,
    )

    ordered_users = _ordered_raw_users(id_mapping)
    src_items = [_id2raw(id_mapping["src"]["id2item"], iid) for iid in range(1, len(id_mapping["src"]["id2item"]))]
    tgt_items = [_id2raw(id_mapping["tgt"]["id2item"], iid) for iid in range(1, len(id_mapping["tgt"]["id2item"]))]

    missing_users = [user for user in ordered_users if user not in user_item_profiles]
    if missing_users:
        raise ValueError(f"Missing raw interaction profiles for {len(missing_users)} users, examples={missing_users[:5]}")

    missing_src_items = [item for item in src_items if item not in item_profiles_src]
    missing_tgt_items = [item for item in tgt_items if item not in item_profiles_tgt]
    if missing_src_items:
        raise ValueError(f"Missing source item profiles for {len(missing_src_items)} items, examples={missing_src_items[:5]}")
    if missing_tgt_items:
        raise ValueError(f"Missing target item profiles for {len(missing_tgt_items)} items, examples={missing_tgt_items[:5]}")

    max_user_chars = modality.get("withoutllm_max_chars_per_user", config.get("withoutllm_max_chars_per_user", 12000))
    max_item_chars = modality.get("withoutllm_max_chars_per_item", config.get("withoutllm_max_chars_per_item", 4000))

    data_users = {}
    for raw_user in ordered_users:
        profile_string = _user_item_profile_to_string(user_item_profiles[raw_user])
        data_users[raw_user] = _truncate_text(_build_direct_user_text(profile_string), max_user_chars)

    data_items_src = {
        raw_item: _truncate_text(_build_item_prompt_string(item_profiles_src[raw_item]), max_item_chars)
        for raw_item in src_items
    }
    data_items_tgt = {
        raw_item: _truncate_text(_build_item_prompt_string(item_profiles_tgt[raw_item]), max_item_chars)
        for raw_item in tgt_items
    }

    logger.info(
        "[DATASET] Built without-LLM raw semantic texts (%s): users=%d, src_items=%d, tgt_items=%d, "
        "max_user_chars=%s, max_item_chars=%s. No chat/completions API is used.",
        modality["name"],
        len(data_users),
        len(data_items_src),
        len(data_items_tgt),
        max_user_chars,
        max_item_chars,
    )

    return {
        "semantic_source": "raw_profile_without_llm_summary",
        "embedding_model": modality["emb_model"],
        "max_user_chars": int(max_user_chars) if int(max_user_chars) > 0 else None,
        "max_item_chars": int(max_item_chars) if int(max_item_chars) > 0 else None,
        "ordered_raw_users": ordered_users,
        "ordered_src_items": src_items,
        "ordered_tgt_items": tgt_items,
        "data_users": data_users,
        "data_items_src": data_items_src,
        "data_items_tgt": data_items_tgt,
        "error_users": {},
        "error_items_src": {},
        "error_items_tgt": {},
    }


def generate_CrossDomain_semantics_both_tag_withoutllm_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    ordered_users = _ordered_raw_users(id_mapping)
    src_items = [_id2raw(id_mapping["src"]["id2item"], iid) for iid in range(1, len(id_mapping["src"]["id2item"]))]
    tgt_items = [_id2raw(id_mapping["tgt"]["id2item"], iid) for iid in range(1, len(id_mapping["tgt"]["id2item"]))]

    max_user_chars = modality.get("withoutllm_max_chars_per_user", config.get("withoutllm_max_chars_per_user", 3000))
    max_item_chars = modality.get("withoutllm_max_chars_per_item", config.get("withoutllm_max_chars_per_item", 1200))

    user_texts = [
        _truncate_text(modality_data["data_users"][raw_user], max_user_chars)
        for raw_user in ordered_users
    ]
    src_item_texts = [
        _truncate_text(modality_data["data_items_src"][raw_item], max_item_chars)
        for raw_item in src_items
    ]
    tgt_item_texts = [
        _truncate_text(modality_data["data_items_tgt"][raw_item], max_item_chars)
        for raw_item in tgt_items
    ]
    all_texts = user_texts + src_item_texts + tgt_item_texts

    if not all_texts:
        raise ValueError("No raw semantic texts to embed.")

    max_text_chars = max(len(text) for text in all_texts)
    avg_text_chars = sum(len(text) for text in all_texts) / len(all_texts)
    logger.info(
        "[DATASET] Embedding without-LLM raw texts (%s): %d users + %d src items + %d tgt items = %d texts. "
        "Truncated max_user_chars=%s, max_item_chars=%s, avg_chars=%.1f, max_chars=%d.",
        modality["name"],
        len(user_texts),
        len(src_item_texts),
        len(tgt_item_texts),
        len(all_texts),
        max_user_chars,
        max_item_chars,
        avg_text_chars,
        max_text_chars,
    )
    embs = _embed_text_batches_withoutllm(config, modality, all_texts)
    return embs.astype(np.float32)


def generate_CrossDomain_semantics_both_tag_withoutllm_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    input_dim = modality_embs.shape[1]
    target_dim = int(modality["emb_pca"])

    if input_dim == target_dim:
        final_embs = modality_embs.astype(np.float32)
    else:
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=target_dim,
            random_state=config.get("seed", 999),
        )
        final_embs = pca.fit_transform(modality_embs).astype(np.float32)

    if modality.get("normalize_semantic_emb", False):
        norm = np.linalg.norm(final_embs, axis=1, keepdims=True) + 1e-12
        final_embs = final_embs / norm

    return final_embs.astype(np.float32)
