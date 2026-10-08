import ast
import gzip
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
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


logger = logging.getLogger()


class LLM4Rerank(GeneralRecommender):
    """Training-free LLM reranking baseline adapted from LLM4Rerank."""

    modality_name = "CrossDomain_semantics_both_tag"

    def __init__(self, config, dataloader):
        super(LLM4Rerank, self).__init__(config, dataloader)
        self.config = config
        self.dataset = dataloader.dataset
        self.candidate_top_m = int(config.get("candidate_top_m", 100))
        self.rerank_model = config.get("rerank_model", "gpt-5.2")

        semantic = dataloader.get_modality_embs()[self.modality_name]
        semantic = torch.from_numpy(semantic).float()
        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        expected_rows = num_users + self.num_items_src + self.num_items_tgt
        if semantic.size(0) < expected_rows:
            raise ValueError(
                f"{self.modality_name} has {semantic.size(0)} rows, "
                f"but at least {expected_rows} rows are required."
            )

        self.user_semantic_emb = semantic[:num_users]
        self.tgt_item_semantic_emb = semantic[-self.num_items_tgt:]

        self.cache_dir = os.path.join(_dataset_path(config), "modality_emb")
        os.makedirs(self.cache_dir, exist_ok=True)
        smoke_suffix = ""
        if config.get("llm4rerank_max_users", None) is not None:
            smoke_suffix = f"_max{int(config['llm4rerank_max_users'])}"
        self.candidate_cache_path = os.path.join(
            self.cache_dir, f"llm4rerank_candidates_top{self.candidate_top_m}{smoke_suffix}.json"
        )
        safe_model_name = self.rerank_model.replace("/", "_")
        self.rerank_cache_path = os.path.join(
            self.cache_dir, f"llm4rerank_rerank_results_{safe_model_name}_top{self.candidate_top_m}{smoke_suffix}.json"
        )
        self.score_cache_path = os.path.join(
            self.cache_dir, f"llm4rerank_scores_{safe_model_name}_top{self.candidate_top_m}{smoke_suffix}.npy"
        )

        scores = self._load_or_build_scores()
        self.register_buffer("rerank_scores", torch.from_numpy(scores).float())

        self.dummy_param = nn.Parameter(torch.zeros(1))

    def calculate_loss(self, interaction, epoch_idx):
        return self.dummy_param.sum() * 0.0

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        if is_warm:
            offset = self.num_users_src - self.num_users_overlap
            users = torch.where(users <= self.num_users_overlap, users, users + offset)
            users = users.clamp(max=self.rerank_scores.size(0) - 1)
        return self.rerank_scores[users]

    def _load_or_build_scores(self):
        if os.path.exists(self.score_cache_path):
            logger.info(f"[LLM4Rerank] Loading cached scores: {self.score_cache_path}")
            return np.load(self.score_cache_path)

        candidates = self._load_or_build_candidates()
        rerank_results = self._load_or_build_rerank_results(candidates)
        scores = self._build_score_matrix(candidates, rerank_results)
        np.save(self.score_cache_path, scores)
        logger.info(f"[LLM4Rerank] Saved scores: {self.score_cache_path}")
        return scores

    def _cold_users(self):
        max_users = self.config.get("llm4rerank_max_users", None)
        users = list(self.dataset.all_users.get("valid_cold_users", []))
        users += list(self.dataset.all_users.get("test_cold_users", []))
        users = [int(u) for u in users]
        if max_users is not None:
            users = users[:int(max_users)]
        return users

    @torch.no_grad()
    def _load_or_build_candidates(self):
        if os.path.exists(self.candidate_cache_path):
            logger.info(f"[LLM4Rerank] Loading cached candidates: {self.candidate_cache_path}")
            with open(self.candidate_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        users = self._cold_users()
        top_m = min(self.candidate_top_m, self.num_items_tgt)
        item_emb = self.tgt_item_semantic_emb
        candidates = {}
        batch_size = int(self.config.get("candidate_build_batch_size", 256))

        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            batch_users = users[start:end]
            user_rows = torch.tensor([u - 1 for u in batch_users], dtype=torch.long)
            user_emb = self.user_semantic_emb[user_rows]
            sim = torch.matmul(user_emb, item_emb.t())
            topk = torch.topk(sim, k=top_m, dim=1).indices + 1
            topk_scores = torch.gather(sim, dim=1, index=topk - 1)

            for idx, uid in enumerate(batch_users):
                candidates[str(uid)] = {
                    "items": [int(x) for x in topk[idx].cpu().tolist()],
                    "scores": [float(x) for x in topk_scores[idx].cpu().tolist()],
                }

        with open(self.candidate_cache_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False)
        logger.info(f"[LLM4Rerank] Saved candidates: {self.candidate_cache_path}")
        return candidates

    def _load_or_build_rerank_results(self, candidates):
        if os.path.exists(self.rerank_cache_path):
            with open(self.rerank_cache_path, "r", encoding="utf-8") as f:
                rerank_results = json.load(f)
        else:
            rerank_results = {}

        missing_users = [uid for uid in candidates.keys() if uid not in rerank_results]
        if not missing_users:
            logger.info(f"[LLM4Rerank] Loading cached rerank results: {self.rerank_cache_path}")
            return rerank_results

        api_key = self.config["openai_api_key"]
        base_url = self.config.get("openai_base_url")
        timeout = float(self.config.get("request_timeout", 180))
        openai_max_retries = int(self.config.get("openai_max_retries", 0))
        max_retry = int(self.config.get("rerank_max_retry", 3))
        num_workers = int(self.config.get("num_workers", min(cpu_count(), 4)))

        raw_context = _build_raw_context(self.config, self.dataset, candidates)
        tasks = []
        for uid in missing_users:
            prompt = _build_rerank_prompt(
                user_id=int(uid),
                history=raw_context["user_history"].get(uid, []),
                candidates=raw_context["candidate_profiles"][uid],
                candidate_items=candidates[uid]["items"],
            )
            tasks.append(
                (
                    uid,
                    prompt,
                    self.rerank_model,
                    max_retry,
                    api_key,
                    base_url,
                    timeout,
                    openai_max_retries,
                )
            )

        logger.info(
            "[LLM4Rerank] Calling reranker for %d users with model=%s.",
            len(tasks),
            self.rerank_model,
        )
        with Pool(processes=num_workers) as pool:
            for idx, out in enumerate(pool.imap_unordered(_process_single_rerank, tasks), start=1):
                uid = out["user_id"]
                candidate_items = candidates[uid]["items"]
                reranked = _sanitize_reranked_items(out["reranked_items"], candidate_items)
                rerank_results[uid] = {
                    "items": reranked,
                    "success": bool(out["success"]),
                    "raw_response": out.get("raw_response"),
                    "error": out.get("error"),
                }
                if idx % int(self.config.get("rerank_save_interval", 10)) == 0:
                    _save_json_atomic(self.rerank_cache_path, rerank_results)

        _save_json_atomic(self.rerank_cache_path, rerank_results)
        logger.info(f"[LLM4Rerank] Saved rerank results: {self.rerank_cache_path}")
        return rerank_results

    def _build_score_matrix(self, candidates, rerank_results):
        scores = np.full((self.num_users_src + 1, self.num_items_tgt + 1), -1e9, dtype=np.float32)
        scores[:, 0] = 0.0

        for uid_str, cand in candidates.items():
            uid = int(uid_str)
            candidate_items = [int(x) for x in cand["items"]]
            result = rerank_results.get(uid_str, {})
            reranked = _sanitize_reranked_items(result.get("items", []), candidate_items)
            for rank, item in enumerate(reranked):
                if 1 <= item <= self.num_items_tgt:
                    scores[uid, item] = float(self.candidate_top_m - rank)
        return scores


def _make_openai_client(api_key, base_url, timeout, max_retries):
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        timeout=timeout,
        max_retries=max_retries,
    )


def _process_single_rerank(args):
    user_id, prompt, model_name, max_retry, api_key, base_url, timeout, openai_max_retries = args
    client = _make_openai_client(api_key, base_url, timeout, openai_max_retries)
    last_text = None
    last_error = None

    for retry_idx in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a recommendation reranker. Return only a JSON array "
                            "of candidate item ids, ordered from most to least relevant."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            last_text = response.choices[0].message.content.strip()
            items = _parse_item_list(last_text)
            if items:
                return {
                    "user_id": user_id,
                    "reranked_items": items,
                    "success": True,
                    "raw_response": last_text,
                    "error": None,
                }
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** retry_idx, 8))

    return {
        "user_id": user_id,
        "reranked_items": [],
        "success": False,
        "raw_response": last_text,
        "error": last_error,
    }


def _parse_item_list(text):
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.strip()
    candidates = [text]
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                continue
        if isinstance(obj, list):
            items = []
            for value in obj:
                try:
                    items.append(int(value))
                except Exception:
                    pass
            if items:
                return items
    return []


def _sanitize_reranked_items(reranked_items, candidate_items):
    candidate_items = [int(x) for x in candidate_items]
    candidate_set = set(candidate_items)
    seen = set()
    cleaned = []
    for item in reranked_items:
        item = int(item)
        if item in candidate_set and item not in seen:
            cleaned.append(item)
            seen.add(item)

    for item in candidate_items:
        if item not in seen:
            cleaned.append(item)
            seen.add(item)
    return cleaned


def _build_rerank_prompt(user_id, history, candidates, candidate_items):
    payload = {
        "task": "Rerank target-domain candidate items for this cold-start user.",
        "user_id": int(user_id),
        "source_domain_history": history,
        "candidate_target_items": candidates,
        "candidate_item_ids": [int(x) for x in candidate_items],
        "output_format": "Return only a JSON array of all candidate item_id values in reranked order.",
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_raw_context(config, dataset, candidates):
    if config["dataset"] != "Amazon2014":
        raise ValueError("LLM4Rerank raw-profile prompts currently support Amazon2014 only.")

    src_domain, tgt_domain = config["domains"]
    data_root = os.path.join(config["data_path"], config["dataset"])
    src_meta_path = os.path.join(data_root, src_domain, "raw", f"meta_{src_domain}.json.gz")
    tgt_meta_path = os.path.join(data_root, tgt_domain, "raw", f"meta_{tgt_domain}.json.gz")
    src_review_path = os.path.join(data_root, src_domain, "raw", f"reviews_{src_domain}_5.json.gz")

    cold_user_ids = [int(uid) for uid in candidates.keys()]
    history_len = int(config.get("rerank_user_history_len", 20))
    train_src = _get_dataset_frame(dataset, "train_src")
    src_hist = train_src[train_src["user"].isin(cold_user_ids)].copy()
    src_hist = src_hist.groupby("user").tail(history_len)

    src_id2user = dataset.id_mapping["src"]["id2user"]
    src_id2item = dataset.id_mapping["src"]["id2item"]
    tgt_id2item = dataset.id_mapping["tgt"]["id2item"]

    needed_src_items = {int(i) for i in src_hist["item"].values}
    needed_tgt_items = set()
    for cand in candidates.values():
        needed_tgt_items.update(int(i) for i in cand["items"])

    raw_src_items = {_id2raw(src_id2item, iid) for iid in needed_src_items}
    raw_tgt_items = {_id2raw(tgt_id2item, iid) for iid in needed_tgt_items}

    src_meta = _load_amazon_metadata(src_meta_path, raw_src_items)
    tgt_meta = _load_amazon_metadata(tgt_meta_path, raw_tgt_items)

    needed_review_pairs = set()
    for row in src_hist.itertuples(index=False):
        raw_user = _id2raw(src_id2user, int(row.user))
        raw_item = _id2raw(src_id2item, int(row.item))
        needed_review_pairs.add((raw_user, raw_item))
    src_reviews = _load_amazon_reviews(src_review_path, needed_review_pairs)

    user_history = {}
    for uid, group in src_hist.groupby("user"):
        entries = []
        for row in group.itertuples(index=False):
            raw_user = _id2raw(src_id2user, int(row.user))
            raw_item = _id2raw(src_id2item, int(row.item))
            meta = src_meta.get(raw_item, {})
            entries.append(
                {
                    "item_id": int(row.item),
                    "title": _clean_text(meta.get("title"), 80),
                    "categories": _compact_categories(meta.get("categories"), 80),
                    "brand": _clean_text(meta.get("brand"), 40),
                    "review": _clean_text(src_reviews.get((raw_user, raw_item)), 120),
                }
            )
        user_history[str(int(uid))] = entries

    candidate_profiles = {}
    for uid, cand in candidates.items():
        profiles = []
        for item_id in cand["items"]:
            raw_item = _id2raw(tgt_id2item, int(item_id))
            meta = tgt_meta.get(raw_item, {})
            profiles.append(
                {
                    "item_id": int(item_id),
                    "title": _clean_text(meta.get("title"), 100),
                    "categories": _compact_categories(meta.get("categories"), 100),
                    "brand": _clean_text(meta.get("brand"), 50),
                }
            )
        candidate_profiles[str(uid)] = profiles

    return {
        "user_history": user_history,
        "candidate_profiles": candidate_profiles,
    }


def _load_amazon_metadata(path, needed_items):
    result = {}
    needed_items = set(needed_items)
    if not needed_items:
        return result
    for obj in _iter_gz_jsonish(path):
        asin = obj.get("asin")
        if asin in needed_items:
            result[asin] = obj
            if len(result) == len(needed_items):
                break
    return result


def _get_dataset_frame(dataset, name):
    if hasattr(dataset, name):
        return getattr(dataset, name)
    if hasattr(dataset, "df") and name in dataset.df:
        return dataset.df[name]
    raise AttributeError(f"RecDataset does not contain dataframe '{name}'.")


def _load_amazon_reviews(path, needed_pairs):
    result = {}
    needed_pairs = set(needed_pairs)
    if not needed_pairs:
        return result
    for obj in _iter_gz_jsonish(path):
        pair = (obj.get("reviewerID"), obj.get("asin"))
        if pair in needed_pairs:
            result[pair] = obj.get("reviewText", "")
            if len(result) == len(needed_pairs):
                break
    return result


def _iter_gz_jsonish(path):
    with gzip.open(path, "rb") as f:
        for line in f:
            line = line.replace(b"true", b"True").replace(b"false", b"False")
            yield eval(line)


def _clean_text(value, max_chars):
    if value is None:
        return "None"
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return "None"
    return text[:max_chars]


def _compact_categories(value, max_chars):
    cats = []
    if isinstance(value, list):
        for path in value:
            if isinstance(path, list):
                cats.extend(str(x) for x in path if str(x).strip())
            elif str(path).strip():
                cats.append(str(path))
    elif value is not None:
        cats.append(str(value))
    text = " > ".join(dict.fromkeys(cats))
    return _clean_text(text, max_chars)


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


def _save_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)
