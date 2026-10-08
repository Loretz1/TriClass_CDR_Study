import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from models.music import (
    MuSiC,
    _collect_music_user_text,
    _run_music_llm_parallel,
    generate_CrossDomain_semantics_music_final_embs,
)


logger = logging.getLogger()

MODALITY_NAME = "CrossDomain_semantics_music_end2end"


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


class MuSiC_End2End(MuSiC):
    """MuSiC runtime variant that generates its own user and item semantic embeddings."""

    def __init__(self, config, dataloader):
        super().__init__(config, _ModalityAliasDataloader(dataloader, MODALITY_NAME, "CrossDomain_semantics_music"))


def _id2raw(id2entity, idx):
    if isinstance(id2entity, dict):
        return id2entity[str(idx)]
    return id2entity[int(idx)]


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


def _music_profile_once(user_text_dict, config, num_workers):
    results = _run_music_llm_parallel(user_text_dict, config, num_workers=num_workers)
    ok, err = {}, {}
    for key in user_text_dict.keys():
        summary = results.get(key)
        if isinstance(summary, str) and summary.strip() and not summary.strip().startswith("ERROR:"):
            ok[key] = summary.strip()
        else:
            err[key] = summary
    return ok, err


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


def _collect_item_texts(config, id_mapping, metadata, domain):
    id2item = id_mapping[domain]["id2item"]
    item_texts = {}
    for item_id in range(1, len(id2item)):
        raw_item = _id2raw(id2item, item_id)
        item_texts[raw_item] = _item_text_from_meta(metadata[domain].get(raw_item), config["dataset"])
    return item_texts


def extract_CrossDomain_semantics_music_end2end_modality_data(config, modality, interaction, id_mapping, raw_data_list):
    reviews = raw_data_list[0]
    metadata = _raw_metadata(config, raw_data_list)
    profile_num_workers = int(modality.get("profile_num_workers", modality.get("num_workers", 20)))
    repair_max_rounds = int(modality.get("repair_max_rounds", 0))
    repair_sleep_seconds = float(modality.get("repair_sleep_seconds", 0.0))

    user_text_src = {}
    for uid in range(1, len(id_mapping["src"]["id2user"])):
        raw_uid = _id2raw(id_mapping["src"]["id2user"], uid)
        user_text_src[raw_uid] = _collect_music_user_text(uid, interaction["src"], id_mapping, "src", reviews) or "None"

    user_text_tgt = {}
    for uid in range(1, len(id_mapping["tgt"]["id2user"])):
        raw_uid = _id2raw(id_mapping["tgt"]["id2user"], uid)
        user_text_tgt[raw_uid] = _collect_music_user_text(uid, interaction["tgt"], id_mapping, "tgt", reviews) or "None"

    logger.info(
        "[DATASET] MuSiC End2End semantic generation (%s): %d src users + %d tgt users + own src/tgt item texts.",
        modality["name"],
        len(user_text_src),
        len(user_text_tgt),
    )
    logger.info(
        "[DATASET] Profile/chat workers=%d, embedding workers=%d, embedding batch size=%d.",
        profile_num_workers,
        int(modality.get("embedding_num_workers", 1)),
        int(modality.get("emb_batch_size", 1)),
    )

    src_summary, src_error = run_until_no_prompt_errors(
        prompts=user_text_src,
        run_once_fn=lambda pending: _music_profile_once(pending, config, profile_num_workers),
        label=f"{modality['name']} source user summaries",
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )
    tgt_summary, tgt_error = run_until_no_prompt_errors(
        prompts=user_text_tgt,
        run_once_fn=lambda pending: _music_profile_once(pending, config, profile_num_workers),
        label=f"{modality['name']} target user summaries",
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )

    return {
        "src_user": src_summary,
        "tgt_user": tgt_summary,
        "error_src_user": src_error,
        "error_tgt_user": tgt_error,
        "src_item_text": _collect_item_texts(config, id_mapping, metadata, "src"),
        "tgt_item_text": _collect_item_texts(config, id_mapping, metadata, "tgt"),
    }


def generate_CrossDomain_semantics_music_end2end_embs(config, modality, interaction, id_mapping, modality_data):
    src_user_list = id_mapping["src"]["id2user"][1:]
    tgt_user_list = id_mapping["tgt"]["id2user"][1:]
    src_item_list = id_mapping["src"]["id2item"][1:]
    tgt_item_list = id_mapping["tgt"]["id2item"][1:]

    src_texts = [modality_data["src_user"][_id2raw(src_user_list, idx)] for idx in range(len(src_user_list))]
    tgt_texts = [modality_data["tgt_user"][_id2raw(tgt_user_list, idx)] for idx in range(len(tgt_user_list))]
    src_item_texts = [modality_data["src_item_text"][_id2raw(src_item_list, idx)] for idx in range(len(src_item_list))]
    tgt_item_texts = [modality_data["tgt_item_text"][_id2raw(tgt_item_list, idx)] for idx in range(len(tgt_item_list))]

    logger.info(
        "[DATASET] MuSiC End2End embedding texts (%s): users=%d, items=%d.",
        modality["name"],
        len(src_texts) + len(tgt_texts),
        len(src_item_texts) + len(tgt_item_texts),
    )

    src_user_emb = _embed_text_batches(config, modality, src_texts)
    tgt_user_emb = _embed_text_batches(config, modality, tgt_texts)
    src_item_emb = _embed_text_batches(config, modality, src_item_texts)
    tgt_item_emb = _embed_text_batches(config, modality, tgt_item_texts)

    return np.concatenate([src_user_emb, tgt_user_emb, src_item_emb, tgt_item_emb], axis=0)


def generate_CrossDomain_semantics_music_end2end_final_embs(config, modality, interaction, id_mapping, modality_embs):
    return generate_CrossDomain_semantics_music_final_embs(config, modality, interaction, id_mapping, modality_embs)
