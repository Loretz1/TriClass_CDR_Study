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
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
from models.llm4rerank import (
    _build_rerank_prompt,
    _clean_text,
    _compact_categories,
    _dataset_path,
    _get_dataset_frame,
    _id2raw,
    _iter_gz_jsonish,
    _load_amazon_metadata,
    _load_amazon_reviews,
    _process_single_rerank,
    _sanitize_reranked_items,
    _save_json_atomic,
)


logger = logging.getLogger()


class LLM4Rerank_End2End(GeneralRecommender):
    """End-to-end LLM4Rerank-style baseline with a trainable GMF candidate generator."""

    def __init__(self, config, dataloader):
        super(LLM4Rerank_End2End, self).__init__(config, dataloader)
        self.config = config
        self.dataset = dataloader.dataset
        self.feature_dim = int(config.get("feature_dim", config.get("embedding_dim", 64)))
        self.num_users_all = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        self.num_items_all = self.num_items_src + self.num_items_tgt
        self.user_embedding = nn.Embedding(self.num_users_all + 1, self.feature_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.num_items_all + 1, self.feature_dim, padding_idx=0)
        self.bpr_loss = BPRLoss()

        self.candidate_top_m = int(config.get("candidate_top_m", 20))
        self.rerank_model = config.get("rerank_model", "gpt-5.2")
        self.rerank_scores = None

        self.cache_dir = os.path.join(_dataset_path(config), "modality_emb")
        os.makedirs(self.cache_dir, exist_ok=True)
        smoke_suffix = ""
        if config.get("llm4rerank_end2end_max_users", None) is not None:
            smoke_suffix = f"_max{int(config['llm4rerank_end2end_max_users'])}"
        safe_model_name = str(self.rerank_model).replace("/", "_")
        self.candidate_cache_path = os.path.join(
            self.cache_dir,
            f"llm4rerank_end2end_gmf_candidates_top{self.candidate_top_m}{smoke_suffix}.json",
        )
        self.rerank_cache_path = os.path.join(
            self.cache_dir,
            f"llm4rerank_end2end_rerank_results_{safe_model_name}_top{self.candidate_top_m}{smoke_suffix}.json",
        )
        self.score_cache_path = os.path.join(
            self.cache_dir,
            f"llm4rerank_end2end_scores_{safe_model_name}_top{self.candidate_top_m}{smoke_suffix}.npy",
        )

        self.apply(xavier_uniform_initialization)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.item_embedding.weight[0].fill_(0)

        logger.info(
            "[LLM4Rerank_End2End] Initialized GMF candidate generator: users=%d, src_items=%d, tgt_items=%d, dim=%d, top_m=%d",
            self.num_users_all,
            self.num_items_src,
            self.num_items_tgt,
            self.feature_dim,
            self.candidate_top_m,
        )

    def calculate_loss(self, interaction, epoch_idx):
        users_src = interaction["users_src"].long()
        pos_src = interaction["pos_items_src"].long()
        neg_src = interaction["neg_items_src"].long()

        users_tgt = self._map_tgt_users_to_global(interaction["users_tgt"].long())
        pos_tgt = self._map_tgt_items_to_global(interaction["pos_items_tgt"].long())
        neg_tgt = self._map_tgt_items_to_global(interaction["neg_items_tgt"].long())

        src_loss = self._bpr(users_src, pos_src, neg_src)
        tgt_loss = self._bpr(users_tgt, pos_tgt, neg_tgt)
        src_weight = float(self.config.get("src_loss_weight", 1.0))
        tgt_weight = float(self.config.get("tgt_loss_weight", 1.0))
        return src_weight * src_loss + tgt_weight * tgt_loss

    def full_sort_predict(self, interaction, is_warm):
        if self.rerank_scores is None:
            scores = self._load_or_build_scores()
            self.rerank_scores = torch.from_numpy(scores).float().to(self.device)

        users = interaction[0].long()
        if is_warm:
            users = torch.where(
                users <= self.num_users_overlap,
                users,
                users + (self.num_users_src - self.num_users_overlap),
            )
            users = users.clamp(max=self.rerank_scores.size(0) - 1)
        return self.rerank_scores[users]

    def _bpr(self, users, pos_items, neg_items):
        user_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1)
        return self.bpr_loss(pos_scores, neg_scores)

    def _map_tgt_users_to_global(self, users_tgt):
        return torch.where(
            users_tgt <= self.num_users_overlap,
            users_tgt,
            users_tgt + (self.num_users_src - self.num_users_overlap),
        )

    def _map_tgt_items_to_global(self, items_tgt):
        return items_tgt + self.num_items_src

    def _cold_users(self):
        users = list(self.dataset.all_users.get("valid_cold_users", []))
        users += list(self.dataset.all_users.get("test_cold_users", []))
        users = [int(u) for u in users]
        max_users = self.config.get("llm4rerank_end2end_max_users", None)
        if max_users is not None:
            users = users[:int(max_users)]
        return users

    def _load_or_build_scores(self):
        total_start = time.perf_counter()
        if os.path.exists(self.score_cache_path):
            logger.info(f"[LLM4Rerank_End2End] Loading cached scores: {self.score_cache_path}")
            scores = np.load(self.score_cache_path)
            logger.info(
                "[LLM4Rerank_End2End][TIME] End-to-end rerank/inference preparation time: %.2fs (cached score matrix)",
                time.perf_counter() - total_start,
            )
            return scores

        candidates = self._load_or_build_candidates()
        rerank_results = self._load_or_build_rerank_results(candidates)
        matrix_start = time.perf_counter()
        scores = self._build_score_matrix(candidates, rerank_results)
        np.save(self.score_cache_path, scores)
        logger.info(f"[LLM4Rerank_End2End] Saved scores: {self.score_cache_path}")
        logger.info("[LLM4Rerank_End2End][TIME] Score matrix build time: %.2fs", time.perf_counter() - matrix_start)
        logger.info(
            "[LLM4Rerank_End2End][TIME] End-to-end rerank/inference preparation time: %.2fs",
            time.perf_counter() - total_start,
        )
        return scores

    @torch.no_grad()
    def _load_or_build_candidates(self):
        start_time = time.perf_counter()
        if os.path.exists(self.candidate_cache_path):
            logger.info(f"[LLM4Rerank_End2End] Loading cached GMF candidates: {self.candidate_cache_path}")
            with open(self.candidate_cache_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
            logger.info(
                "[LLM4Rerank_End2End][TIME] Candidate generation time: %.2fs (cached)",
                time.perf_counter() - start_time,
            )
            return candidates

        users = self._cold_users()
        top_m = min(self.candidate_top_m, self.num_items_tgt)
        batch_size = int(self.config.get("candidate_build_batch_size", 256))
        tgt_global_ids = torch.arange(
            self.num_items_src + 1,
            self.num_items_src + self.num_items_tgt + 1,
            device=self.device,
            dtype=torch.long,
        )
        target_emb = self.item_embedding(tgt_global_ids)
        candidates = {}

        was_training = self.training
        self.eval()
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            batch_users = torch.tensor(users[start:end], dtype=torch.long, device=self.device)
            user_emb = self.user_embedding(batch_users)
            scores = torch.matmul(user_emb, target_emb.t())
            topk = torch.topk(scores, k=top_m, dim=1).indices + 1
            topk_scores = torch.gather(scores, dim=1, index=topk - 1)
            for idx, uid in enumerate(batch_users.cpu().tolist()):
                candidates[str(int(uid))] = {
                    "items": [int(x) for x in topk[idx].cpu().tolist()],
                    "scores": [float(x) for x in topk_scores[idx].cpu().tolist()],
                }
        if was_training:
            self.train()

        with open(self.candidate_cache_path, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False)
        logger.info(f"[LLM4Rerank_End2End] Saved GMF candidates: {self.candidate_cache_path}")
        logger.info("[LLM4Rerank_End2End][TIME] Candidate generation time: %.2fs", time.perf_counter() - start_time)
        return candidates

    def _load_or_build_rerank_results(self, candidates):
        start_time = time.perf_counter()
        if os.path.exists(self.rerank_cache_path):
            with open(self.rerank_cache_path, "r", encoding="utf-8") as f:
                rerank_results = json.load(f)
        else:
            rerank_results = {}

        missing_users = [uid for uid in candidates.keys() if uid not in rerank_results]
        if not missing_users:
            logger.info(f"[LLM4Rerank_End2End] Loading cached rerank results: {self.rerank_cache_path}")
            logger.info(
                "[LLM4Rerank_End2End][TIME] LLM reranking time: %.2fs (cached)",
                time.perf_counter() - start_time,
            )
            return rerank_results

        api_key = self.config["openai_api_key"]
        base_url = self.config.get("openai_base_url")
        timeout = float(self.config.get("request_timeout", 180))
        openai_max_retries = int(self.config.get("openai_max_retries", 0))
        max_retry = int(self.config.get("rerank_max_retry", 3))
        num_workers = int(self.config.get("num_workers", min(cpu_count(), 4)))

        raw_context = _build_llm4rerank_end2end_raw_context(self.config, self.dataset, candidates)
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
            "[LLM4Rerank_End2End] Calling LLM reranker for %d users with model=%s.",
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
        logger.info(f"[LLM4Rerank_End2End] Saved rerank results: {self.rerank_cache_path}")
        logger.info("[LLM4Rerank_End2End][TIME] LLM reranking time: %.2fs", time.perf_counter() - start_time)
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


def _build_llm4rerank_end2end_raw_context(config, dataset, candidates):
    if config["dataset"] == "Amazon2014":
        from models.llm4rerank import _build_raw_context

        return _build_raw_context(config, dataset, candidates)
    if config["dataset"] == "Douban":
        return _build_douban_raw_context(config, dataset, candidates)
    raise ValueError(f"LLM4Rerank_End2End raw-profile prompts do not support dataset={config['dataset']}.")


def _build_douban_raw_context(config, dataset, candidates):
    src_domain, tgt_domain = config["domains"]
    data_root = os.path.join(config["data_path"], config["dataset"])
    src_meta_path = os.path.join(data_root, src_domain, "raw", "item_meta.json.gz")
    tgt_meta_path = os.path.join(data_root, tgt_domain, "raw", "item_meta.json.gz")
    src_review_path = os.path.join(data_root, src_domain, "raw", "reviews.json.gz")

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
    src_meta = _load_douban_metadata(src_meta_path, raw_src_items)
    tgt_meta = _load_douban_metadata(tgt_meta_path, raw_tgt_items)

    needed_review_pairs = set()
    for row in src_hist.itertuples(index=False):
        raw_user = _id2raw(src_id2user, int(row.user))
        raw_item = _id2raw(src_id2item, int(row.item))
        needed_review_pairs.add((raw_user, raw_item))
    src_reviews = _load_douban_reviews(src_review_path, needed_review_pairs)

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
                    "title": _douban_item_title(meta),
                    "labels": _clean_text(meta.get("labels"), 120),
                    "metadata": _compact_douban_metadata(meta, 160),
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
                    "title": _douban_item_title(meta),
                    "labels": _clean_text(meta.get("labels"), 120),
                    "metadata": _compact_douban_metadata(meta, 180),
                }
            )
        candidate_profiles[str(uid)] = profiles

    return {
        "user_history": user_history,
        "candidate_profiles": candidate_profiles,
    }


def _load_douban_metadata(path, needed_items):
    result = {}
    needed_items = set(needed_items)
    if not needed_items:
        return result
    for obj in _iter_gz_jsonish(path):
        iid = obj.get("iid")
        if iid in needed_items:
            result[iid] = obj
            if len(result) == len(needed_items):
                break
    return result


def _load_douban_reviews(path, needed_pairs):
    result = {}
    needed_pairs = set(needed_pairs)
    if not needed_pairs:
        return result
    for obj in _iter_gz_jsonish(path):
        pair = (obj.get("uid"), obj.get("iid"))
        if pair in needed_pairs:
            result[pair] = obj.get("comments", "")
            if len(result) == len(needed_pairs):
                break
    return result


def _douban_item_title(meta):
    for key in ("name", "title", "fullname"):
        value = meta.get(key)
        if value:
            return _clean_text(value, 100)
    return "None"


def _compact_douban_metadata(meta, max_chars):
    fields = []
    for key in (
        "labels",
        "name",
        "director",
        "summary",
        "writer",
        "country",
        "pubdate",
        "language",
        "rating",
        "tag",
        "fullname",
        "CategoryID",
    ):
        value = meta.get(key)
        if value:
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            fields.append(f"{key}: {value}")
    text = re.sub(r"\s+", " ", "; ".join(fields)).strip()
    return _clean_text(text, max_chars)
