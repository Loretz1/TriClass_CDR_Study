import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from models.llm4cdsr import (
    LLM4CDSR,
    _build_overall_prompt_from_summaries,
    _build_subseq_prompt_from_titles,
    _build_user_mixed_sequences,
    _compute_counts_from_id_mapping,
    _get_title,
    _raw_item_to_global_index,
    _run_prompts,
    generate_CrossDomain_semantics_LLM4CDSR_final_embs,
)


logger = logging.getLogger()

MODALITY_NAME = "CrossDomain_semantics_LLM4CDSR_end2end"


class _ModalityAliasDataloader:
    def __init__(self, dataloader, source_name, alias_name):
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


class LLM4CDSR_End2End(LLM4CDSR):
    """LLM4CDSR runtime variant that generates its own user and item semantic embeddings."""

    def __init__(self, config, dataloader):
        super().__init__(config, _ModalityAliasDataloader(dataloader, MODALITY_NAME, "CrossDomain_semantics_LLM4CDSR"))


def _id2raw(id2entity, idx):
    if isinstance(id2entity, dict):
        return id2entity[str(idx)]
    return id2entity[int(idx)]


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
    return os.path.join(
        config["data_path"],
        config["dataset"],
        "+".join(config["domains"]),
        "only_overlap_users" if config["only_overlap_users"] else "all_users",
        split_dir,
    )


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


def _make_openai_client(api_key, base_url, timeout, max_retries):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        timeout=timeout,
        max_retries=max_retries,
    )


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


def _raw_metadata(config, raw_data_list):
    if config["dataset"] == "Amazon2014":
        return raw_data_list[1]
    if config["dataset"] == "Douban":
        return raw_data_list[2]
    raise ValueError(f"Unsupported dataset: {config['dataset']}")


def _clean_text(value, max_chars=240):
    if value is None:
        return "None"
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_chars] if text else "None"


def _compact_categories(value, max_chars=160):
    cats = []
    if isinstance(value, list):
        for path in value:
            if isinstance(path, list):
                cats.extend(str(x) for x in path if str(x).strip())
            elif str(path).strip():
                cats.append(str(path))
    elif value is not None:
        cats.append(str(value))
    return _clean_text(" > ".join(dict.fromkeys(cats)), max_chars)


def _item_text_from_meta(meta, dataset):
    if not isinstance(meta, dict):
        return "None"
    if dataset == "Amazon2014":
        payload = {
            "title": _clean_text(meta.get("title"), 120),
            "categories": _compact_categories(meta.get("categories"), 160),
            "brand": _clean_text(meta.get("brand"), 80),
            "description": _clean_text(meta.get("description"), 240),
        }
    elif dataset == "Douban":
        payload = {
            "labels": _clean_text(meta.get("labels"), 180),
            "name": _clean_text(meta.get("name"), 120),
            "summary": _clean_text(meta.get("summary"), 240),
            "tag": _clean_text(meta.get("tag"), 160),
            "category": _clean_text(meta.get("CategoryID"), 80),
        }
    else:
        payload = {"metadata": _clean_text(meta, 240)}
    return json.dumps(payload, ensure_ascii=False)


def _ordered_item_texts(config, id_mapping, metadata, domain):
    id2item = id_mapping[domain]["id2item"]
    raw_items = [_id2raw(id2item, item_id) for item_id in range(1, len(id2item))]
    texts = [_item_text_from_meta(metadata[domain].get(raw_item), config["dataset"]) for raw_item in raw_items]
    return raw_items, texts


def _item_emb_cache_path(config, modality):
    cache_dir = os.path.join(_dataset_path(config), "modality_emb")
    os.makedirs(cache_dir, exist_ok=True)
    safe_model = str(modality["emb_model"]).replace("/", "_")
    return os.path.join(cache_dir, f"{modality['name']}_{safe_model}_item_embs.npy")


def _load_or_create_item_embs(config, modality, id_mapping, metadata):
    path = _item_emb_cache_path(config, modality)
    if os.path.exists(path):
        logger.info("[DATASET] LLM4CDSR End2End item embeddings already created: %s", path)
        return np.load(path), path

    src_raw_items, src_texts = _ordered_item_texts(config, id_mapping, metadata, "src")
    tgt_raw_items, tgt_texts = _ordered_item_texts(config, id_mapping, metadata, "tgt")
    logger.info(
        "[DATASET] LLM4CDSR End2End generating own item embeddings (%s): src_items=%d, tgt_items=%d.",
        modality["name"],
        len(src_raw_items),
        len(tgt_raw_items),
    )
    item_embs = _embed_text_batches(config, modality, src_texts + tgt_texts)
    np.save(path, item_embs.astype(np.float32))
    logger.info("[DATASET] LLM4CDSR End2End saved own item embeddings: %s", path)
    return item_embs.astype(np.float32), path


def extract_CrossDomain_semantics_LLM4CDSR_end2end_modality_data(config, modality, interaction, id_mapping, raw_data_list):
    metadata = _raw_metadata(config, raw_data_list)
    counts = _compute_counts_from_id_mapping(id_mapping)
    item_embs, item_emb_path = _load_or_create_item_embs(config, modality, id_mapping, metadata)

    expected_items = counts["num_src_items"] + counts["num_tgt_items"]
    if item_embs.shape[0] != expected_items:
        raise ValueError(f"Item embedding row mismatch: got {item_embs.shape[0]}, expected {expected_items}")

    from sklearn.cluster import KMeans

    K = int(modality.get("cluster_k", 2))
    kmeans = KMeans(n_clusters=K, random_state=999, n_init="auto")
    cluster_ids = kmeans.fit_predict(item_embs)

    user2seq = _build_user_mixed_sequences(interaction, id_mapping)
    subseq_prompts = {}
    for raw_user, seq in user2seq.items():
        buckets = [[] for _ in range(K)]
        for dom, raw_item in seq:
            idx = _raw_item_to_global_index(dom, raw_item, id_mapping, counts)
            cid = int(cluster_ids[idx])
            buckets[cid].append(_get_title(raw_item, metadata[dom], config["dataset"]))
        for k in range(K):
            subseq_prompts[f"{raw_user}__sub{k}"] = _build_subseq_prompt_from_titles(buckets[k])

    model_name = modality.get("llm_chat_model", "gpt-4o-mini")
    max_retry = int(modality.get("max_retry", 2))
    profile_num_workers = int(modality.get("profile_num_workers", modality.get("num_workers", 20)))
    repair_max_rounds = int(modality.get("repair_max_rounds", 0))
    repair_sleep_seconds = float(modality.get("repair_sleep_seconds", 0.0))
    logger.info(
        "[DATASET] LLM4CDSR End2End profile/chat workers=%d, embedding workers=%d, embedding batch size=%d.",
        profile_num_workers,
        int(modality.get("embedding_num_workers", 1)),
        int(modality.get("emb_batch_size", 1)),
    )

    data_sub, err_sub = run_until_no_prompt_errors(
        prompts=subseq_prompts,
        run_once_fn=lambda pending: _run_prompts(
            pending,
            model_name=model_name,
            api_key=config["openai_api_key"],
            base_url=config.get("openai_base_url"),
            max_retry=max_retry,
            num_workers=profile_num_workers,
        ),
        label=f"{modality['name']} subsequence profiles",
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )

    overall_prompts = {}
    for raw_user in user2seq.keys():
        subs = [data_sub.get(f"{raw_user}__sub{k}", "None") for k in range(K)]
        overall_prompts[raw_user] = _build_overall_prompt_from_summaries(subs)

    data_overall, err_overall = run_until_no_prompt_errors(
        prompts=overall_prompts,
        run_once_fn=lambda pending: _run_prompts(
            pending,
            model_name=model_name,
            api_key=config["openai_api_key"],
            base_url=config.get("openai_base_url"),
            max_retry=max_retry,
            num_workers=profile_num_workers,
        ),
        label=f"{modality['name']} overall user profiles",
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )

    return {
        "counts": counts,
        "cluster_k": K,
        "item_emb_path": item_emb_path,
        "data_subseq": data_sub,
        "error_subseq": err_sub,
        "data_users": data_overall,
        "error_users": err_overall,
    }


def generate_CrossDomain_semantics_LLM4CDSR_end2end_embs(config, modality, interaction, id_mapping, modality_data):
    counts = modality_data["counts"]
    num_users = counts["num_users"]
    num_src_items = counts["num_src_items"]
    num_tgt_items = counts["num_tgt_items"]

    src_id2user = id_mapping["src"]["id2user"][1:]
    src_users_set = set(src_id2user)
    tgt_id2user = id_mapping["tgt"]["id2user"][1:]
    tgt_only_users = [u for u in tgt_id2user if u not in src_users_set]
    ordered_users = list(src_id2user) + tgt_only_users
    if len(ordered_users) != num_users:
        raise ValueError("Union user size mismatch")

    data_users = modality_data["data_users"]
    user_texts = [data_users.get(raw_user, "None") for raw_user in ordered_users]
    user_text_emb = _embed_text_batches(config, modality, user_texts)

    item_emb = np.load(modality_data["item_emb_path"]).astype(np.float32)
    expected_items = num_src_items + num_tgt_items
    if item_emb.shape[0] != expected_items:
        raise ValueError(f"Item embedding row mismatch: got {item_emb.shape[0]}, expected {expected_items}")

    src_item_emb = item_emb[:num_src_items]
    tgt_item_emb = item_emb[num_src_items:num_src_items + num_tgt_items]
    return np.concatenate([user_text_emb, src_item_emb, tgt_item_emb], axis=0)


def generate_CrossDomain_semantics_LLM4CDSR_end2end_final_embs(config, modality, interaction, id_mapping, modality_embs):
    return generate_CrossDomain_semantics_LLM4CDSR_final_embs(config, modality, interaction, id_mapping, modality_embs)
