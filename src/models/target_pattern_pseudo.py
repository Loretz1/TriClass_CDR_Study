#!/usr/bin/env python3
"""Build one target-pattern pseudo-interaction cache shared by CDR backbones."""

import argparse
import ast
import copy
import gzip
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

LOGGER = logging.getLogger("target-pattern-pseudo")
CACHE_FORMAT_VERSION = 3
USER_INTEREST_SCHEMA_VERSION = 2
BRIDGE_TYPE = "target_user_semantic_to_collaborative_v1"
PHASES = (
    "metadata",
    "target-patterns",
    "user-interests",
    "embeddings",
    "lightgcn",
    "bridge",
    "retrieve",
)


def config_scalar(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def apply_model_domain_profile(config):
    """Apply fixed per-direction backbone settings without changing Config."""
    profiles = config.get("domain_profiles") or {}
    if not profiles:
        return
    domain_key = "+".join(config["domains"])
    profile = profiles.get(domain_key)
    if profile is None:
        supported = ", ".join(sorted(profiles))
        raise ValueError(
            "No model domain profile for '{}'. Supported profiles: {}".format(
                domain_key, supported
            )
        )
    for key, value in profile.items():
        config[key] = copy.deepcopy(value)


def dataset_split_directory(config):
    split_name = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_name += f"_{config['k_cores']}cores"
    return os.path.abspath(
        os.path.join(
            config["data_path"],
            config["dataset"],
            "+".join(config["domains"]),
            "only_overlap_users" if config["only_overlap_users"] else "all_users",
            split_name,
        )
    )


def mapping_fingerprint(id_mapping):
    identity = {
        "src_id2user": id_mapping["src"]["id2user"],
        "tgt_id2item": id_mapping["tgt"]["id2item"],
    }
    payload = json.dumps(
        identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TargetPatternBridge(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, output_dim=64, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, semantic_embedding):
        return F.normalize(self.network(semantic_embedding), dim=-1)


class TargetPatternPseudoSupervisor:
    """Load one immutable pseudo-interaction cache shared by all backbones."""

    def __init__(self, config, dataloader):
        self.logger = logging.getLogger()
        self.device = config["device"]
        self.num_items_tgt = dataloader.dataset.num_items_tgt
        self.num_users_overlap = dataloader.dataset.num_users_overlap
        self.num_users_src = dataloader.dataset.num_users_src
        self.id_mapping = dataloader.dataset.id_mapping
        self.rng = np.random.default_rng(
            int(
                config_scalar(
                    config.get("pseudo_sampler_seed"),
                    config_scalar(config["seed"], 999),
                )
            )
        )

        cache_path = config.get("pseudo_cache_path", "auto")
        if not cache_path or cache_path == "auto":
            cache_path = os.path.join(
                dataset_split_directory(config),
                "target_pattern_pseudo",
                "pseudo_rankings.npz",
            )
        self.cache_path = os.path.abspath(cache_path)
        self.manifest_path = os.path.join(
            os.path.dirname(self.cache_path), "manifest.json"
        )
        self._load_and_validate(dataloader)

    def _load_and_validate(self, dataloader):
        if not os.path.isfile(self.cache_path):
            raise FileNotFoundError(
                "Shared target-pattern pseudo cache does not exist: {}. "
                "Run `python -m models.target_pattern_pseudo` first.".format(
                    self.cache_path
                )
            )
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(
                "Pseudo cache manifest does not exist: {}".format(self.manifest_path)
            )

        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        expected = {
            "format_version": CACHE_FORMAT_VERSION,
            "dataset": dataloader.dataset.config["dataset"],
            "domains": list(dataloader.dataset.config["domains"]),
            "num_users_overlap": self.num_users_overlap,
            "num_users_src": self.num_users_src,
            "num_items_tgt": self.num_items_tgt,
            "mapping_sha256": mapping_fingerprint(self.id_mapping),
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            detail = ", ".join(
                "{}: cache={!r}, expected={!r}".format(key, actual, wanted)
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(
                "Pseudo cache is incompatible with the active split ({}).".format(
                    detail
                )
            )

        cache = np.load(self.cache_path, allow_pickle=False)
        user_ids = np.asarray(cache["user_ids"], dtype=np.int64)
        ranked_item_ids = np.asarray(cache["item_ids"], dtype=np.int64)
        ranked_scores = np.asarray(cache["scores"], dtype=np.float32)
        official_top_k = int(manifest["official_top_k"])

        if ranked_item_ids.ndim != 2 or ranked_scores.shape != ranked_item_ids.shape:
            raise ValueError(
                "Pseudo cache item_ids/scores must be equally shaped 2-D arrays."
            )
        if len(user_ids) != ranked_item_ids.shape[0]:
            raise ValueError("Pseudo cache user count does not match ranking rows.")
        if official_top_k <= 0 or official_top_k > ranked_item_ids.shape[1]:
            raise ValueError(
                "manifest.official_top_k is outside the cached ranking width."
            )
        if len(np.unique(user_ids)) != len(user_ids):
            raise ValueError("Pseudo cache contains duplicate user rows.")
        if np.any(user_ids <= self.num_users_overlap) or np.any(
            user_ids > self.num_users_src
        ):
            raise ValueError("Pseudo cache must contain source-side cold users only.")

        fixed_items = ranked_item_ids[:, :official_top_k]
        if np.any(fixed_items <= 0) or np.any(fixed_items > self.num_items_tgt):
            raise ValueError("Pseudo cache contains an invalid target item id.")

        self.official_top_k = official_top_k
        self.user_ids = user_ids
        self.item_ids = fixed_items
        self.pair_users = np.repeat(user_ids, official_top_k)
        self.pair_items = fixed_items.reshape(-1)
        self.pseudo_positive = {
            int(user): set(map(int, items))
            for user, items in zip(user_ids.tolist(), fixed_items.tolist())
        }
        self.target_positive = self._target_training_positives(dataloader)
        self.logger.info(
            "[PSEUDO PLUGIN] Loaded %d users x Top-%d shared pseudo items from %s",
            len(self.user_ids),
            self.official_top_k,
            self.cache_path,
        )

    def _target_training_positives(self, dataloader):
        positives = {}
        src_id2user = self.id_mapping["src"]["id2user"]
        tgt_user2id = self.id_mapping["tgt"]["user2id"]
        target_sets = dataloader.dataset.positive_items_tgt
        for source_user in self.user_ids.tolist():
            raw_user = src_id2user[int(source_user)]
            target_user = tgt_user2id.get(raw_user)
            positives[int(source_user)] = (
                set(target_sets.get(target_user, set()))
                if target_user is not None
                else set()
            )
        return positives

    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("pseudo_num_per_batch must be positive.")
        pair_index = self.rng.integers(0, len(self.pair_users), size=batch_size)
        users = self.pair_users[pair_index]
        positives = self.pair_items[pair_index]
        negatives = np.empty(batch_size, dtype=np.int64)

        for row, user in enumerate(users.tolist()):
            forbidden = self.pseudo_positive[int(user)] | self.target_positive[int(user)]
            if len(forbidden) >= self.num_items_tgt:
                raise RuntimeError(
                    "No target negative remains for source user {}.".format(user)
                )
            while True:
                candidate = int(self.rng.integers(1, self.num_items_tgt + 1))
                if candidate not in forbidden:
                    negatives[row] = candidate
                    break

        return (
            torch.as_tensor(users, dtype=torch.long, device=self.device),
            torch.as_tensor(positives, dtype=torch.long, device=self.device),
            torch.as_tensor(negatives, dtype=torch.long, device=self.device),
        )

    @staticmethod
    def bpr_loss(user_embedding, positive_embedding, negative_embedding):
        positive_score = (user_embedding * positive_embedding).sum(dim=-1)
        negative_score = (user_embedding * negative_embedding).sum(dim=-1)
        return -F.logsigmoid(positive_score - negative_score).mean()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fixed target-pattern pseudo-interaction set."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["Amazon2014", "Douban"]
    )
    parser.add_argument("--domains", nargs=2, required=True, metavar=("SRC", "TGT"))
    parser.add_argument(
        "--config",
        default=os.path.join(
            SRC_ROOT, "configs", "model", "TargetPatternPseudo.yaml"
        ),
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=("all",) + PHASES,
        help="Run all phases or one phase and its cached prerequisites.",
    )
    parser.add_argument("--force-phase", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--user-scope",
        choices=("cold", "overlap"),
        default="cold",
        help=(
            "Build interests for source-side cold users (default) or overlapped "
            "users. Overlap mode writes separate overlap_* artifacts and reuses "
            "the frozen target LightGCN and semantic bridge for retrieval."
        ),
    )
    return parser.parse_args()


def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "build.log"), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def deep_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path, dataset, domains, device_override="auto"):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    profiles = config.pop("domain_profiles", {}) or {}
    domain_key = "+".join(domains)
    if domain_key not in profiles:
        supported = ", ".join(sorted(profiles))
        raise ValueError(
            "No pseudo-builder domain profile for {}. Supported: {}".format(
                domain_key, supported
            )
        )
    deep_update(config, profiles[domain_key])
    split = config.pop("split", None)
    if not isinstance(split, dict):
        raise ValueError("Pseudo-builder YAML must define a split mapping.")
    for key, value in split.items():
        config[key] = value
    config["dataset"] = dataset
    config["domains"] = list(domains)
    data_path = config["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.abspath(os.path.join(SRC_ROOT, data_path))
    config["data_path"] = data_path
    requested_device = device_override
    if requested_device == "auto":
        requested_device = config.get("device", "auto")
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    config["device"] = str(requested_device)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_torch_save(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".pt", dir=os.path.dirname(path)
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=".npz", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path, key_field):
    records = {}
    if not os.path.isfile(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[str(record[key_field])] = record
            except Exception as error:
                raise ValueError(
                    "Invalid JSONL record at {}:{}: {}".format(path, line_number, error)
                )
    return records


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def compact_jsonl(path, records, sort_key):
    ordered = sorted(records.values(), key=lambda row: row[sort_key])
    temporary = path + ".compact.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_split(config):
    split_dir = dataset_split_directory(config)
    required = (
        "all_users.json",
        "id_mapping.json",
        "train_src.pkl",
        "train_tgt.pkl",
    )
    missing = [
        name for name in required if not os.path.isfile(os.path.join(split_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            "Dataset split is incomplete at {} (missing {}). Run the benchmark data "
            "preparation once before building pseudo interactions.".format(
                split_dir, ", ".join(missing)
            )
        )
    with open(
        os.path.join(split_dir, "all_users.json"), "r", encoding="utf-8"
    ) as handle:
        all_users = json.load(handle)
    with open(
        os.path.join(split_dir, "id_mapping.json"), "r", encoding="utf-8"
    ) as handle:
        id_mapping = json.load(handle)
    train_src = pd.read_pickle(os.path.join(split_dir, "train_src.pkl"))
    train_tgt = pd.read_pickle(os.path.join(split_dir, "train_tgt.pkl"))
    return split_dir, all_users, id_mapping, train_src, train_tgt


def parse_gzip_records(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                continue
            except json.JSONDecodeError:
                pass
            try:
                normalized = re.sub(r"\btrue\b", "True", line, flags=re.IGNORECASE)
                normalized = re.sub(
                    r"\bfalse\b", "False", normalized, flags=re.IGNORECASE
                )
                normalized = re.sub(
                    r"\bnull\b", "None", normalized, flags=re.IGNORECASE
                )
                value = ast.literal_eval(normalized)
                if isinstance(value, dict):
                    yield value
                    continue
            except Exception:
                pass
            raise ValueError("Cannot parse {} line {}".format(path, line_number))


def clean_text(value, max_chars=800):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(clean_text(part, max_chars=max_chars) for part in value)
    elif isinstance(value, dict):
        value = " ".join(
            "{} {}".format(key, clean_text(part, max_chars=max_chars))
            for key, part in sorted(value.items())
        )
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value[:max_chars]


def category_paths(value):
    if not value:
        return []
    if isinstance(value, str):
        return [clean_text(value, 400)]
    if not isinstance(value, (list, tuple)):
        return []
    if value and all(isinstance(part, str) for part in value):
        return [
            " / ".join(clean_text(part, 120) for part in value if clean_text(part, 120))
        ]
    paths = []
    for path in value:
        if isinstance(path, str):
            cleaned = clean_text(path, 400)
        elif isinstance(path, (list, tuple)):
            cleaned = " / ".join(
                clean_text(part, 120) for part in path if clean_text(part, 120)
            )
        else:
            cleaned = ""
        if cleaned:
            paths.append(cleaned)
    return list(dict.fromkeys(paths))


def normalize_metadata(record, domain, dataset):
    if dataset == "Douban":
        labels = [
            clean_text(label, 120)
            for label in str(record.get("labels") or "").split("|")
            if clean_text(label, 120)
        ]
        return {
            "title": "",
            "categories": list(dict.fromkeys(labels))[:32],
            "brand": "",
            "features": "",
            "description": "",
            "domain": domain,
            "metadata_kind": "tags",
        }
    paths = category_paths(record.get("categories"))
    if not paths:
        paths = category_paths(record.get("category"))
    return {
        "title": clean_text(record.get("title"), 300),
        "categories": paths[:8],
        "brand": clean_text(record.get("brand"), 120),
        "features": clean_text(record.get("feature") or record.get("features"), 500),
        "description": clean_text(record.get("description"), 700),
        "domain": domain,
        "metadata_kind": "categories",
    }


def metadata_text(metadata, compact=False):
    categories = "; ".join(metadata.get("categories", [])) or "unspecified category"
    category_label = (
        "Item tags"
        if metadata.get("metadata_kind") == "tags"
        else "Categories"
    )
    if compact:
        fields = []
        if metadata.get("title"):
            fields.append(
                "Title: {}".format(clean_text(metadata["title"], 180))
            )
        fields.append(
            "{}: {}".format(category_label, clean_text(categories, 300))
        )
        for label, key, limit in (
            ("Brand", "brand", 80),
            ("Features", "features", 180),
            ("Description", "description", 180),
        ):
            value = clean_text(metadata.get(key), limit)
            if value:
                fields.append("{}: {}".format(label, value))
        return " ".join(fields)
    fields = []
    if metadata.get("title"):
        fields.append("Title: {}".format(metadata["title"]))
    fields.append("{}: {}".format(category_label, categories))
    if metadata.get("brand"):
        fields.append("Brand: {}".format(metadata["brand"]))
    if metadata.get("features"):
        fields.append("Features: {}".format(metadata["features"]))
    if metadata.get("description"):
        fields.append("Description: {}".format(metadata["description"]))
    return " ".join(fields)


def target_attribute_context(target_patterns, settings, seed):
    """Build one deterministic, multi-attribute target-domain context."""
    records = sorted(target_patterns.values(), key=lambda row: row["item_id"])
    sample_size = min(
        int(settings.get("target_context_examples", 24)), len(records)
    )
    if sample_size <= 0:
        raise ValueError("llm.target_context_examples must be positive.")
    if sample_size < len(records):
        rng = random.Random(int(seed))
        records = sorted(
            rng.sample(records, sample_size), key=lambda row: row["item_id"]
        )
    max_chars = int(settings.get("target_context_pattern_max_chars", 500))
    examples = [
        "- {}".format(clean_text(record["target_item_pattern"], max_chars))
        for record in records
    ]
    return (
        "Target-item attributes may include product type or category, function, material, "
        "style or theme, technical properties, and use context. Treat category as one "
        "attribute rather than the sole basis of transfer. Representative target-item "
        "attribute patterns:\n{}"
    ).format("\n".join(examples))


def load_domain_metadata(config, domain, valid_items):
    if config["dataset"] == "Douban":
        filename = "item_meta.json.gz"
        item_key = "iid"
    else:
        filename = "meta_{}.json.gz".format(domain)
        item_key = "asin"
    path = os.path.join(
        config["data_path"], config["dataset"], domain, "raw", filename
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    valid_items = set(valid_items)
    metadata = {}
    for record in parse_gzip_records(path):
        raw_item = str(record.get(item_key, ""))
        if raw_item in valid_items:
            metadata[raw_item] = normalize_metadata(
                record, domain, config["dataset"]
            )
    for raw_item in valid_items:
        metadata.setdefault(
            raw_item,
            {
                "title": "",
                "categories": [domain.replace("_", " ")],
                "brand": "",
                "features": "",
                "description": "",
                "domain": domain,
                "metadata_kind": (
                    "tags" if config["dataset"] == "Douban" else "categories"
                ),
            },
        )
    return metadata


def load_source_reviews(config, domain, required_pairs):
    if config["dataset"] == "Douban":
        filename = "reviews.json.gz"
        user_key, item_key = "uid", "iid"
    else:
        filename = "reviews_{}_5.json.gz".format(domain)
        user_key, item_key = "reviewerID", "asin"
    path = os.path.join(
        config["data_path"], config["dataset"], domain, "raw", filename
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    pending = set(required_pairs)
    reviews = {}
    for record in parse_gzip_records(path):
        key = (str(record.get(user_key, "")), str(record.get(item_key, "")))
        if key not in pending:
            continue
        if config["dataset"] == "Douban":
            comment = clean_text(record.get("comments"), 500)
            rating = record.get("rating")
            parts = []
            if rating is not None:
                parts.append("Rating: {}/5".format(rating))
            if comment:
                parts.append("Comment: {}".format(comment))
            text = "; ".join(parts)
        else:
            text = clean_text(
                record.get("reviewText") or record.get("summary"), 500
            )
        if text:
            reviews[key] = text
        pending.discard(key)
        if not pending:
            break
    return reviews


def build_metadata_artifacts(
    config, output_dir, all_users, id_mapping, train_src, user_scope="cold"
):
    if user_scope not in ("cold", "overlap"):
        raise ValueError("Unsupported user scope: {}".format(user_scope))
    artifact_name = "metadata.json" if user_scope == "cold" else "overlap_metadata.json"
    artifact_path = os.path.join(output_dir, artifact_name)
    if os.path.isfile(artifact_path):
        with open(artifact_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    overlap_count = len(all_users["overlap_users"])
    if user_scope == "cold":
        scoped_train = train_src[train_src["user"] > overlap_count]
    else:
        scoped_train = train_src[
            (train_src["user"] > 0) & (train_src["user"] <= overlap_count)
        ]
    source_item_ids = sorted(
        set(int(value) for value in scoped_train["item"].tolist())
    )
    source_raw_items = [
        id_mapping["src"]["id2item"][value] for value in source_item_ids
    ]
    target_raw_items = id_mapping["tgt"]["id2item"][1:]
    source_metadata = load_domain_metadata(
        config, config["domains"][0], source_raw_items
    )
    target_metadata = load_domain_metadata(
        config, config["domains"][1], target_raw_items
    )

    source_pairs = set()
    for user, item in scoped_train[["user", "item"]].itertuples(
        index=False, name=None
    ):
        source_pairs.add(
            (
                id_mapping["src"]["id2user"][int(user)],
                id_mapping["src"]["id2item"][int(item)],
            )
        )
    source_reviews = load_source_reviews(config, config["domains"][0], source_pairs)

    artifact = {
        "source_metadata": source_metadata,
        "target_metadata": target_metadata,
        "source_reviews": [
            {"user": user, "item": item, "text": text}
            for (user, item), text in sorted(source_reviews.items())
        ],
    }
    atomic_json(artifact_path, artifact)
    LOGGER.info(
        "Prepared %s-user metadata for %d source items, %d target items, and %d "
        "source reviews.",
        user_scope,
        len(source_metadata),
        len(target_metadata),
        len(source_reviews),
    )
    return artifact


class OpenAIClientPool(object):
    def __init__(
        self,
        settings,
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        host_header_env="OPENAI_HOST_HEADER",
    ):
        api_key = settings.get("api_key") or os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                "OpenAI-compatible API key is empty. Set {} or fill the plugin YAML.".format(
                    api_key_env
                )
            )
        self.api_key = api_key
        self.base_url = settings.get("base_url") or os.environ.get(base_url_env) or None
        self.host_header = (
            settings.get("host_header") or os.environ.get(host_header_env) or None
        )
        self.timeout = float(settings.get("request_timeout", 180))
        self.client_retries = int(settings.get("client_max_retries", 0))
        self.local = threading.local()

    def get(self):
        if not hasattr(self.local, "client"):
            from openai import OpenAI

            client_kwargs = {}
            if self.host_header:
                client_kwargs["default_headers"] = {"Host": self.host_header}
            self.local.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.client_retries,
                **client_kwargs,
            )
        return self.local.client


def response_text(response):
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                parts.append(str(part.text))
        return "".join(parts)
    return str(content)


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object response.")
    return value


class LLMCallError(RuntimeError):
    def __init__(self, message, last_response=None):
        super().__init__(message)
        self.last_response = last_response


def call_chat_json(pool, settings, system_prompt, user_prompt, validator):
    attempts = int(settings["max_retries"])
    last_error = None
    last_response = None
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages = list(base_messages)
    for attempt in range(attempts):
        raw_text = None
        try:
            temperature = settings.get("temperature", 0)
            if attempt:
                temperature = settings.get("retry_temperature", temperature)
            kwargs = {
                "model": settings["model"],
                "messages": messages,
                "temperature": float(temperature),
                "response_format": {"type": "json_object"},
            }
            max_output_tokens = settings.get("max_output_tokens")
            if max_output_tokens is not None:
                kwargs["max_tokens"] = int(max_output_tokens)
            response = pool.get().chat.completions.create(**kwargs)
            raw_text = response_text(response)
            last_response = raw_text
            return validator(parse_json_object(raw_text))
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                if raw_text and bool(
                    settings.get("retry_include_invalid_response", True)
                ):
                    messages = base_messages + [
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                "The previous JSON failed validation: {}. Return a corrected, "
                                "complete JSON object only. Recheck every required identifier, "
                                "include each exactly once, and preserve the requested schema."
                            ).format(clean_text(str(error), 1200)),
                        },
                    ]
                else:
                    messages = [
                        base_messages[0],
                        {
                            "role": "user",
                            "content": user_prompt
                            + (
                                "\n\nA previous attempt failed validation: {}. Regenerate "
                                "the entire concise JSON object from scratch. Do not continue "
                                "or reproduce the previous response."
                            ).format(clean_text(str(error), 600)),
                        },
                    ]
                delay = float(settings.get("retry_base_seconds", 2)) * (
                    2 ** min(attempt, 5)
                )
                time.sleep(delay)
    raise LLMCallError(
        "LLM request failed after {} attempts: {}".format(attempts, last_error),
        last_response=last_response,
    )


def interest_system_prompt():
    return (
        "You construct aligned cross-domain recommendation supervision. Return strict JSON "
        "only. In one response, first summarize each coherent SOURCE-DOMAIN item-interest "
        "group, then transfer that same group into a TARGET-DOMAIN item-attribute pattern. "
        "Both patterns must describe items and their attributes, never user identity, "
        "demographics, personality, or generic user traits. Keep each source pattern and its "
        "transferred target pattern in the same JSON object."
    )


def shortened_indices(values, limit=30):
    values = list(values)
    return values if len(values) <= limit else values[:limit] + ["..."]


def validate_aligned_interests(value, expected_indices, max_interests):
    interests = value.get("interests")
    if not isinstance(interests, list) or not 1 <= len(interests) <= max_interests:
        raise ValueError(
            "interests must contain between 1 and {} groups.".format(max_interests)
        )
    expected = set(int(index) for index in expected_indices)
    seen = []
    normalized = []
    for group_index, interest in enumerate(interests):
        if not isinstance(interest, dict):
            raise ValueError("Each interest must be an object.")
        indices = interest.get("source_item_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("source_item_indices must be a non-empty list.")
        try:
            indices = [int(index) for index in indices]
        except (TypeError, ValueError):
            raise ValueError("Every source_item_indices value must be an integer.")
        source_pattern = clean_text(interest.get("source_interest_pattern"), 1200)
        target_pattern = clean_text(interest.get("target_item_pattern"), 1200)
        if not source_pattern:
            raise ValueError("Every interest needs a source_interest_pattern.")
        if not target_pattern:
            raise ValueError("Every interest needs a target_item_pattern.")
        seen.extend(indices)
        normalized.append(
            {
                "interest_id": group_index,
                "name": clean_text(
                    interest.get("interest_name")
                    or interest.get("name")
                    or "interest {}".format(group_index + 1),
                    120,
                ),
                "source_item_indices": sorted(indices),
                "source_interest_pattern": source_pattern,
                "target_item_pattern": target_pattern,
            }
        )
    counts = Counter(seen)
    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    if missing or unexpected or duplicates:
        raise ValueError(
            "source_item_indices are not an exact partition; missing={}, unexpected={}, "
            "duplicates={}.".format(
                shortened_indices(missing),
                shortened_indices(unexpected),
                shortened_indices(duplicates),
            )
        )
    return normalized


def repair_aligned_interests(value, expected_indices, max_interests, items):
    """Deterministically turn a nearly valid LLM grouping into an exact partition."""
    interests = value.get("interests")
    if not isinstance(interests, list) or not interests:
        raise ValueError("Cannot repair an empty interests response.")

    expected = sorted(set(int(index) for index in expected_indices))
    expected_set = set(expected)
    candidates = []
    for original_index, interest in enumerate(interests):
        if not isinstance(interest, dict):
            continue
        source_pattern = clean_text(
            interest.get("source_interest_pattern"), 1200
        )
        target_pattern = clean_text(interest.get("target_item_pattern"), 1200)
        if not source_pattern or not target_pattern:
            continue
        raw_indices = interest.get("source_item_indices")
        indices = []
        if isinstance(raw_indices, list):
            for raw_index in raw_indices:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if index in expected_set and index not in indices:
                    indices.append(index)
        candidates.append(
            {
                "original_index": original_index,
                "name": clean_text(
                    interest.get("interest_name")
                    or interest.get("name")
                    or "interest {}".format(original_index + 1),
                    120,
                ),
                "source_item_indices": indices,
                "source_interest_pattern": source_pattern,
                "target_item_pattern": target_pattern,
            }
        )
    if not candidates:
        raise ValueError("Cannot repair interests without usable pattern text.")

    def tokens(text):
        normalized = clean_text(text).lower()
        output = set(re.findall(r"[a-z0-9]+", normalized))
        cjk = re.findall(r"[\u3400-\u9fff]", normalized)
        output.update(
            "".join(cjk[index : index + 2])
            for index in range(max(0, len(cjk) - 1))
        )
        return output

    def pattern_similarity(left, right):
        left_tokens = tokens(left)
        right_tokens = tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / float(
            len(left_tokens | right_tokens)
        )

    # Keep the most supported groups and merge any excess groups into their
    # closest retained source-interest pattern.
    if len(candidates) > max_interests:
        ranked = sorted(
            candidates,
            key=lambda group: (
                -len(group["source_item_indices"]),
                group["original_index"],
            ),
        )
        retained = ranked[:max_interests]
        for extra in ranked[max_interests:]:
            destination = max(
                range(len(retained)),
                key=lambda index: (
                    pattern_similarity(
                        extra["source_interest_pattern"],
                        retained[index]["source_interest_pattern"],
                    ),
                    -retained[index]["original_index"],
                ),
            )
            retained[destination]["source_item_indices"].extend(
                extra["source_item_indices"]
            )
            retained[destination]["source_interest_pattern"] = clean_text(
                retained[destination]["source_interest_pattern"]
                + "; "
                + extra["source_interest_pattern"],
                1200,
            )
            retained[destination]["target_item_pattern"] = clean_text(
                retained[destination]["target_item_pattern"]
                + "; "
                + extra["target_item_pattern"],
                1200,
            )
        candidates = sorted(retained, key=lambda group: group["original_index"])

    item_text = {}
    for item in items:
        index = int(item["index"])
        text = metadata_text(item["metadata"], compact=False)
        if item.get("review"):
            text += " " + str(item["review"])
        item_text[index] = text

    assignments = [[] for _ in candidates]
    claimed_by = defaultdict(list)
    for group_index, group in enumerate(candidates):
        for index in group["source_item_indices"]:
            claimed_by[index].append(group_index)

    for index in expected:
        eligible = claimed_by.get(index) or list(range(len(candidates)))
        destination = max(
            eligible,
            key=lambda group_index: (
                pattern_similarity(
                    item_text.get(index, ""),
                    candidates[group_index]["source_interest_pattern"],
                ),
                -len(assignments[group_index]),
                -candidates[group_index]["original_index"],
            ),
        )
        assignments[destination].append(index)

    repaired = []
    for group, indices in zip(candidates, assignments):
        if not indices:
            continue
        repaired.append(
            {
                "interest_id": len(repaired),
                "interest_name": group["name"],
                "source_item_indices": sorted(indices),
                "source_interest_pattern": group["source_interest_pattern"],
                "target_item_pattern": group["target_item_pattern"],
            }
        )
    return validate_aligned_interests(
        {"interests": repaired}, expected, max_interests
    )


def build_interest_prompt(
    items,
    source_domain,
    target_domain,
    target_context,
    max_interests,
    pattern_max_chars=240,
    response_max_chars=10000,
    compact=False,
):
    item_lines = []
    for item in items:
        fields = [
            "index={}".format(item["index"]),
            metadata_text(item["metadata"], compact=compact),
        ]
        if item.get("review"):
            review = clean_text(item["review"], 180) if compact else item["review"]
            fields.append("Observed review: {}".format(review))
        item_lines.append(" | ".join(fields))
    expected = [item["index"] for item in items]
    return (
        "Source domain: {source}.\n"
        "Target domain: {target}.\n\n"
        "Target-domain item context:\n{target_context}\n\n"
        "Source-domain training interactions:\n{items}\n\n"
        "Required source indices: {expected}.\n"
        "Create between 1 and {max_interests} coherent interest groups. Perform these two "
        "aligned steps for every group: (1) summarize the shared SOURCE-DOMAIN item "
        "attributes as source_interest_pattern; (2) infer what concrete TARGET-DOMAIN item "
        "attributes would satisfy that source interest as target_item_pattern. Consider all "
        "available attributes, including product type, function, material, style or theme, "
        "technical properties, and use context. Category is only one possible attribute. "
        "Summarize instead of copying the input: use at most 10 distinct attribute phrases "
        "and at most {pattern_max_chars} characters in each source_interest_pattern or "
        "target_item_pattern. Never repeat an attribute or concatenate all item tags. Keep "
        "interest_name under 8 words. Write arrays compactly and keep the complete response "
        "under {response_max_chars} characters. "
        "Assign every required source index to exactly one group, with no missing, duplicate, "
        "or unexpected indices. Use consecutive interest_id values starting at 0. Return the "
        "complete JSON object in this exact shape:\n"
        '{{"interests":[{{"interest_id":0,"interest_name":"...",'
        '"source_item_indices":[0,1],"source_interest_pattern":"...",'
        '"target_item_pattern":"..."}}]}}'
    ).format(
        source=source_domain.replace("_", " "),
        target=target_domain.replace("_", " "),
        target_context=target_context,
        items="\n".join(item_lines),
        expected=json.dumps(expected, separators=(",", ":")),
        max_interests=max_interests,
        pattern_max_chars=pattern_max_chars,
        response_max_chars=response_max_chars,
    )


def infer_user_interests(
    pool, settings, items, source_domain, target_domain, target_context
):
    max_interests = int(settings["max_interests"])
    max_items = int(settings.get("max_source_items_per_user", 256))
    if len(items) > max_items:
        raise ValueError(
            "User history has {} items, exceeding llm.max_source_items_per_user={}. "
            "Increase the limit only after checking the model context window.".format(
                len(items), max_items
            )
        )
    expected = [item["index"] for item in items]
    validator = lambda value: validate_aligned_interests(
        value, expected, max_interests
    )
    pattern_max_chars = int(settings.get("interest_pattern_max_chars", 240))
    response_max_chars = int(settings.get("interest_response_max_chars", 10000))
    compact_min_items = int(settings.get("compact_user_prompt_min_items", 0))
    use_compact_prompt = compact_min_items > 0 and len(items) >= compact_min_items
    prompt = build_interest_prompt(
        items,
        source_domain,
        target_domain,
        target_context,
        max_interests,
        pattern_max_chars=pattern_max_chars,
        response_max_chars=response_max_chars,
        compact=use_compact_prompt,
    )
    max_prompt_chars = int(settings.get("max_user_prompt_chars", 240000))
    if len(prompt) > max_prompt_chars and not use_compact_prompt:
        prompt = build_interest_prompt(
            items,
            source_domain,
            target_domain,
            target_context,
            max_interests,
            pattern_max_chars=pattern_max_chars,
            response_max_chars=response_max_chars,
            compact=True,
        )
    if len(prompt) > max_prompt_chars:
        raise ValueError(
            "User prompt remains {} characters after attribute-preserving compaction, "
            "exceeding llm.max_user_prompt_chars={}.".format(
                len(prompt), max_prompt_chars
            )
        )
    try:
        merged = call_chat_json(
            pool,
            settings,
            interest_system_prompt(),
            prompt,
            validator,
        )
    except LLMCallError as error:
        if not bool(settings.get("repair_invalid_interest_partitions", True)):
            raise
        if not error.last_response:
            raise
        try:
            merged = repair_aligned_interests(
                parse_json_object(error.last_response),
                expected,
                max_interests,
                items,
            )
        except Exception:
            raise error
        LOGGER.warning(
            "Repaired an invalid LLM interest partition locally for a history of %d "
            "source interactions.",
            len(items),
        )

    expected = list(range(len(items)))
    assigned = [index for group in merged for index in group["source_item_indices"]]
    if sorted(assigned) != expected or len(assigned) != len(set(assigned)):
        raise RuntimeError("Merged interests lost or duplicated source interactions.")
    denominator = float(len(items))
    for group in merged:
        group["support"] = len(group["source_item_indices"])
        group["weight"] = group["support"] / denominator
        group["source_item_ids"] = [
            items[index]["item_id"] for index in group["source_item_indices"]
        ]
        group["source_raw_item_ids"] = [
            items[index]["raw_item_id"] for index in group["source_item_indices"]
        ]
    if abs(sum(group["weight"] for group in merged) - 1.0) > 1e-8:
        raise RuntimeError("Interest support weights do not sum to one.")
    return merged


def target_pattern_system_prompt():
    return (
        "Summarize target-domain items as target-item attribute patterns for recommendation. "
        "Return strict JSON only. Describe any supported item attributes, such as type or "
        "category, function, style, theme, creator, genre, material, technical properties, "
        "and use context. Do not infer user traits or invent facts."
    )


def validate_target_patterns(value, expected):
    rows = value.get("items")
    if not isinstance(rows, list):
        raise ValueError("items must be a list.")
    output = {}
    for row in rows:
        item_id = int(row["item_id"])
        pattern = clean_text(row.get("target_item_pattern"), 1200)
        if not pattern or item_id in output:
            raise ValueError("Invalid or duplicate target item pattern.")
        output[item_id] = pattern
    if set(output) != set(expected):
        raise ValueError("Target item response does not exactly cover the request.")
    return output


def build_target_pattern_prompt(batch, target_domain, compact=False):
    rows = []
    for item_id, metadata in batch:
        rows.append(
            "item_id={} | {}".format(item_id, metadata_text(metadata, compact=compact))
        )
    return (
        "Target domain: {}. Convert every item below into one concise target-item attribute "
        "pattern. Preserve every item_id exactly once.\n{}\n\nReturn "
        '{{"items":[{{"item_id":1,"target_item_pattern":"..."}}]}}.'
    ).format(target_domain.replace("_", " "), "\n".join(rows))


def create_target_patterns(config, output_dir, artifact, id_mapping, force=False):
    path = os.path.join(output_dir, "target_item_patterns.jsonl")
    failure_path = os.path.join(output_dir, "target_item_pattern_failures.jsonl")
    if force:
        for candidate in (path, failure_path):
            if os.path.isfile(candidate):
                os.remove(candidate)
    records = load_jsonl(path, "item_id")
    all_items = [
        (item_id, artifact["target_metadata"][raw_item])
        for item_id, raw_item in enumerate(id_mapping["tgt"]["id2item"][1:], start=1)
    ]
    pending = [row for row in all_items if str(row[0]) not in records]
    if not pending:
        return records

    settings = config["llm"]
    if not bool(settings.get("summarize_target_items", True)):
        for item_id, metadata in pending:
            record = {
                "item_id": item_id,
                "raw_item_id": id_mapping["tgt"]["id2item"][item_id],
                "target_item_pattern": metadata_text(metadata),
            }
            append_jsonl(path, record)
            records[str(item_id)] = record
        compact_jsonl(path, records, "item_id")
        return records

    pool = OpenAIClientPool(settings)
    batch_size = int(settings.get("target_items_per_prompt", 20))
    batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]

    def worker(batch):
        expected = [item_id for item_id, _ in batch]
        validator = lambda value: validate_target_patterns(value, expected)
        try:
            return call_chat_json(
                pool,
                settings,
                target_pattern_system_prompt(),
                build_target_pattern_prompt(batch, config["domains"][1], compact=False),
                validator,
            )
        except RuntimeError:
            return call_chat_json(
                pool,
                settings,
                target_pattern_system_prompt(),
                build_target_pattern_prompt(batch, config["domains"][1], compact=True),
                validator,
            )

    workers = int(settings["workers"])
    submission_batch_size = max(
        workers, int(settings.get("target_submission_batch_size", 128))
    )
    retry_rounds = max(
        1, int(settings.get("failed_target_batch_retry_rounds", 2))
    )
    failure_records = load_jsonl(failure_path, "batch_id")
    unresolved = list(batches)
    completed = 0
    LOGGER.info(
        "Generating %d missing target-item patterns in %d request batches.",
        len(pending),
        len(batches),
    )
    for retry_round in range(1, retry_rounds + 1):
        round_failures = []
        for group_start in range(0, len(unresolved), submission_batch_size):
            batch_group = unresolved[
                group_start : group_start + submission_batch_size
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(worker, batch): batch for batch in batch_group
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    batch_id = str(batch[0][0])
                    try:
                        values = future.result()
                    except Exception as error:
                        raw_response = getattr(error, "last_response", None)
                        failure = {
                            "batch_id": batch_id,
                            "item_ids": [item_id for item_id, _ in batch],
                            "retry_round": retry_round,
                            "error": clean_text(str(error), 2000),
                            "last_response": clean_text(raw_response, 4000)
                            if raw_response
                            else "",
                            "failed_at": datetime.now().astimezone().isoformat(),
                        }
                        append_jsonl(failure_path, failure)
                        failure_records[batch_id] = failure
                        round_failures.append(batch)
                        LOGGER.warning(
                            "Target-pattern batch %s failed in round %d: %s",
                            batch_id,
                            retry_round,
                            failure["error"],
                        )
                        continue

                    for item_id, pattern in values.items():
                        record = {
                            "item_id": item_id,
                            "raw_item_id": id_mapping["tgt"]["id2item"][item_id],
                            "target_item_pattern": pattern,
                        }
                        append_jsonl(path, record)
                        records[str(item_id)] = record
                    failure_records.pop(batch_id, None)
                    completed += len(values)
                    if completed % 500 == 0 or completed == len(pending):
                        LOGGER.info(
                            "Target-item patterns: %d/%d newly completed.",
                            completed,
                            len(pending),
                        )
        unresolved = round_failures
        if not unresolved:
            break
        LOGGER.warning(
            "Target-pattern round %d completed with %d failed request batches.",
            retry_round,
            len(unresolved),
        )

    compact_jsonl(path, records, "item_id")
    if failure_records:
        compact_jsonl(failure_path, failure_records, "batch_id")
    elif os.path.isfile(failure_path):
        os.remove(failure_path)
    if unresolved:
        raise RuntimeError(
            "Target-item pattern generation left {} unresolved request batches after "
            "{} rounds. Successful items were saved; inspect {} and rerun to resume only "
            "missing items.".format(len(unresolved), retry_rounds, failure_path)
        )
    return records


def create_user_interests(
    config,
    output_dir,
    artifact,
    target_patterns,
    all_users,
    id_mapping,
    train_src,
    user_scope="cold",
    force=False,
):
    if user_scope not in ("cold", "overlap"):
        raise ValueError("Unsupported user scope: {}".format(user_scope))
    if user_scope == "cold":
        path = os.path.join(output_dir, "source_user_interests.jsonl")
        failure_path = os.path.join(
            output_dir, "source_user_interest_failures.jsonl"
        )
    else:
        path = os.path.join(output_dir, "overlap_user_interests.jsonl")
        failure_path = os.path.join(
            output_dir, "overlap_user_interest_failures.jsonl"
        )
    if force:
        for candidate in (path, failure_path):
            if os.path.isfile(candidate):
                os.remove(candidate)
    overlap_count = len(all_users["overlap_users"])
    histories = defaultdict(list)
    for user, item in train_src[["user", "item"]].itertuples(index=False, name=None):
        user, item = int(user), int(item)
        if (user_scope == "cold" and user > overlap_count) or (
            user_scope == "overlap" and 0 < user <= overlap_count
        ):
            histories[user].append(item)
    if user_scope == "cold":
        selected_users = list(
            range(overlap_count + 1, len(id_mapping["src"]["id2user"]))
        )
    else:
        selected_users = list(range(1, overlap_count + 1))
    missing_histories = [
        user for user in selected_users if not histories.get(user)
    ]
    if missing_histories:
        raise ValueError(
            "{} {} source users have no source training history.".format(
                len(missing_histories), user_scope
            )
        )

    settings = config["llm"]
    loaded_records = load_jsonl(path, "user_id")
    records = {}
    stale_records = 0
    for key, record in loaded_records.items():
        user = int(record.get("user_id", -1))
        interests = record.get("interests")
        valid = False
        if (
            int(record.get("schema_version", -1)) == USER_INTEREST_SCHEMA_VERSION
            and user in histories
            and isinstance(interests, list)
            and bool(interests)
        ):
            try:
                validate_aligned_interests(
                    {"interests": interests},
                    range(len(histories[user])),
                    int(settings["max_interests"]),
                )
                weights = [float(interest["weight"]) for interest in interests]
                valid = abs(sum(weights) - 1.0) <= 1e-8
            except (KeyError, TypeError, ValueError):
                valid = False
        if valid:
            records[key] = record
        else:
            stale_records += 1
    if stale_records:
        compact_jsonl(path, records, "user_id")
        LOGGER.info(
            "Discarded %d legacy or incomplete user-interest records; target-item "
            "patterns remain cached.",
            stale_records,
        )

    review_lookup = {
        (row["user"], row["item"]): row["text"] for row in artifact["source_reviews"]
    }
    pending = [user for user in selected_users if str(user) not in records]
    if not pending:
        return records
    pool = OpenAIClientPool(settings)
    source_id2item = id_mapping["src"]["id2item"]
    source_id2user = id_mapping["src"]["id2user"]
    target_context = target_attribute_context(
        target_patterns, settings, config["seed"]
    )

    def worker(user):
        raw_user = source_id2user[user]
        items = []
        for index, item_id in enumerate(histories[user]):
            raw_item = source_id2item[item_id]
            items.append(
                {
                    "index": index,
                    "item_id": item_id,
                    "raw_item_id": raw_item,
                    "metadata": artifact["source_metadata"][raw_item],
                    "review": review_lookup.get((raw_user, raw_item), ""),
                }
            )
        interests = infer_user_interests(
            pool,
            settings,
            items,
            config["domains"][0],
            config["domains"][1],
            target_context,
        )
        return {
            "schema_version": USER_INTEREST_SCHEMA_VERSION,
            "user_id": user,
            "raw_user_id": raw_user,
            "history_size": len(items),
            "interests": interests,
        }

    workers = int(settings["workers"])
    submission_batch_size = max(
        workers, int(settings.get("submission_batch_size", 256))
    )
    retry_rounds = max(1, int(settings.get("failed_user_retry_rounds", 2)))
    failure_records = load_jsonl(failure_path, "user_id")
    unresolved = list(pending)
    completed = 0
    LOGGER.info(
        "Generating aligned source/target interests for %d missing %s source users "
        "with %d workers in batches of at most %d.",
        len(pending),
        user_scope,
        workers,
        submission_batch_size,
    )

    for retry_round in range(1, retry_rounds + 1):
        round_failures = []
        for batch_start in range(0, len(unresolved), submission_batch_size):
            user_batch = unresolved[
                batch_start : batch_start + submission_batch_size
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(worker, user): user for user in user_batch
                }
                for future in as_completed(futures):
                    user = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:
                        raw_response = getattr(error, "last_response", None)
                        failure = {
                            "user_id": user,
                            "raw_user_id": source_id2user[user],
                            "history_size": len(histories[user]),
                            "retry_round": retry_round,
                            "error": clean_text(str(error), 2000),
                            "last_response": clean_text(raw_response, 4000)
                            if raw_response
                            else "",
                            "failed_at": datetime.now().astimezone().isoformat(),
                        }
                        append_jsonl(failure_path, failure)
                        failure_records[str(user)] = failure
                        round_failures.append(user)
                        LOGGER.warning(
                            "User-interest generation failed for user %d in round %d: %s",
                            user,
                            retry_round,
                            failure["error"],
                        )
                        continue

                    append_jsonl(path, record)
                    records[str(record["user_id"])] = record
                    failure_records.pop(str(user), None)
                    completed += 1
                    if completed % 100 == 0 or completed == len(pending):
                        LOGGER.info(
                            "Source-user interests: %d/%d newly completed; current "
                            "round failures=%d.",
                            completed,
                            len(pending),
                            len(round_failures),
                        )
        unresolved = sorted(set(round_failures))
        if not unresolved:
            break
        LOGGER.warning(
            "User-interest round %d completed with %d failures.",
            retry_round,
            len(unresolved),
        )

    compact_jsonl(path, records, "user_id")
    if failure_records:
        compact_jsonl(failure_path, failure_records, "user_id")
    elif os.path.isfile(failure_path):
        os.remove(failure_path)
    if unresolved:
        raise RuntimeError(
            "User-interest generation left {} unresolved users after {} rounds. "
            "Successful users were saved; inspect {} and rerun to resume only missing "
            "users.".format(len(unresolved), retry_rounds, failure_path)
        )
    return records


def embedding_paths(output_dir, prefix):
    return {
        "array": os.path.join(output_dir, prefix + ".npy"),
        "complete": os.path.join(output_dir, prefix + "_complete.npy"),
        "metadata": os.path.join(output_dir, prefix + "_metadata.json"),
    }


def embedding_text_fingerprint(keys, texts, settings):
    digest = hashlib.sha256()
    digest.update(str(settings["model"]).encode("utf-8"))
    digest.update(str(settings["dimensions"]).encode("ascii"))
    for key, value in zip(keys, texts):
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def call_embeddings(pool, settings, texts):
    attempts = int(settings["max_retries"])
    last_error = None
    for attempt in range(attempts):
        try:
            kwargs = {"model": settings["model"], "input": texts}
            dimensions = settings.get("dimensions")
            if dimensions:
                kwargs["dimensions"] = int(dimensions)
            response = pool.get().embeddings.create(**kwargs)
            rows = sorted(response.data, key=lambda row: row.index)
            values = np.asarray([row.embedding for row in rows], dtype=np.float32)
            expected_shape = (len(texts), int(settings["dimensions"]))
            if values.shape != expected_shape:
                raise ValueError(
                    "Embedding API returned shape {}, expected {}.".format(
                        values.shape, expected_shape
                    )
                )
            return values
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                delay = float(settings.get("retry_base_seconds", 2)) * (
                    2 ** min(attempt, 5)
                )
                time.sleep(delay)
    raise RuntimeError(
        "Embedding request failed after {} attempts: {}".format(attempts, last_error)
    )


def create_embeddings(config, output_dir, prefix, records, force=False):
    """Embed ordered key/text pairs with a resumable NPY memmap."""
    settings = config["embedding"]
    keys = [str(key) for key, _ in records]
    texts = [text for _, text in records]
    if len(keys) != len(set(keys)):
        raise ValueError("{} contains duplicate embedding keys.".format(prefix))
    fingerprint = embedding_text_fingerprint(keys, texts, settings)
    paths = embedding_paths(output_dir, prefix)
    if force:
        for path in paths.values():
            if os.path.isfile(path):
                os.remove(path)

    expected_metadata = {
        "model": settings["model"],
        "dimensions": int(settings["dimensions"]),
        "count": len(keys),
        "keys": keys,
        "text_sha256": fingerprint,
    }
    if os.path.isfile(paths["metadata"]):
        with open(paths["metadata"], "r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        if existing_metadata != expected_metadata:
            raise ValueError(
                "{} embedding inputs changed. Remove its three cached embedding files "
                "or rerun with --force-phase.".format(prefix)
            )
    else:
        atomic_json(paths["metadata"], expected_metadata)

    shape = (len(keys), int(settings["dimensions"]))
    if not os.path.isfile(paths["array"]):
        embeddings = np.lib.format.open_memmap(
            paths["array"], mode="w+", dtype=np.float32, shape=shape
        )
        embeddings[:] = 0.0
        embeddings.flush()
    else:
        embeddings = np.lib.format.open_memmap(paths["array"], mode="r+")
        if embeddings.shape != shape or embeddings.dtype != np.float32:
            raise ValueError(
                "{} embedding cache has an incompatible shape or dtype.".format(prefix)
            )

    if not os.path.isfile(paths["complete"]):
        complete = np.lib.format.open_memmap(
            paths["complete"], mode="w+", dtype=np.bool_, shape=(len(keys),)
        )
        complete[:] = False
        complete.flush()
    else:
        complete = np.lib.format.open_memmap(paths["complete"], mode="r+")
        if complete.shape != (len(keys),):
            raise ValueError(
                "{} completion cache has an incompatible shape.".format(prefix)
            )

    missing = np.flatnonzero(~np.asarray(complete))
    if len(missing) == 0:
        LOGGER.info("%s embeddings are complete (%d rows).", prefix, len(keys))
        return np.asarray(embeddings), keys

    pool = OpenAIClientPool(settings)
    batch_size = int(settings["batch_size"])
    batches = [
        missing[start : start + batch_size]
        for start in range(0, len(missing), batch_size)
    ]

    def worker(indices):
        indices = np.asarray(indices, dtype=np.int64)
        values = call_embeddings(pool, settings, [texts[index] for index in indices])
        return indices, values

    LOGGER.info(
        "Embedding %d missing %s records in %d batches.",
        len(missing),
        prefix,
        len(batches),
    )
    completed_rows = 0
    workers = int(settings["workers"])
    submission_batch_size = max(
        workers, int(settings.get("submission_batch_size", 64))
    )
    retry_rounds = max(1, int(settings.get("failed_batch_retry_rounds", 2)))
    unresolved = list(batches)
    for retry_round in range(1, retry_rounds + 1):
        round_failures = []
        for group_start in range(0, len(unresolved), submission_batch_size):
            batch_group = unresolved[
                group_start : group_start + submission_batch_size
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(worker, indices): indices
                    for indices in batch_group
                }
                for future in as_completed(futures):
                    try:
                        indices, values = future.result()
                    except Exception as error:
                        indices = futures[future]
                        round_failures.append(indices)
                        LOGGER.warning(
                            "%s embedding batch starting at row %d failed in round %d: %s",
                            prefix,
                            int(indices[0]),
                            retry_round,
                            clean_text(str(error), 1200),
                        )
                        continue
                    embeddings[indices] = values
                    embeddings.flush()
                    complete[indices] = True
                    complete.flush()
                    completed_rows += len(indices)
                    if completed_rows % 1000 == 0 or completed_rows == len(missing):
                        LOGGER.info(
                            "%s embeddings: %d/%d newly completed.",
                            prefix,
                            completed_rows,
                            len(missing),
                        )
        unresolved = round_failures
        if not unresolved:
            break
        LOGGER.warning(
            "%s embedding round %d completed with %d failed batches.",
            prefix,
            retry_round,
            len(unresolved),
        )
    if unresolved:
        raise RuntimeError(
            "{} embedding generation left {} unresolved batches after {} rounds. "
            "Completed rows are cached; rerun to resume.".format(
                prefix, len(unresolved), retry_rounds
            )
        )
    if not bool(np.asarray(complete).all()):
        raise RuntimeError(
            "{} embedding cache is unexpectedly incomplete.".format(prefix)
        )
    return np.asarray(embeddings), keys


class TargetLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers, adjacency):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.adjacency = adjacency
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self):
        initial = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        outputs = [initial]
        current = initial
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            outputs.append(current)
        final = torch.stack(outputs, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]


def normalized_graph(num_users, num_items, edges, device):
    users = edges[:, 0].astype(np.int64) - 1
    items = edges[:, 1].astype(np.int64) - 1 + num_users
    row = np.concatenate([users, items])
    col = np.concatenate([items, users])
    degree = np.bincount(row, minlength=num_users + num_items).astype(np.float32)
    degree_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    degree_inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    values = degree_inv_sqrt[row] * degree_inv_sqrt[col]
    indices = torch.as_tensor(np.stack([row, col]), dtype=torch.long, device=device)
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, (num_users + num_items, num_users + num_items), device=device
    ).coalesce()


def interaction_sets(edges, num_users):
    positives = [set() for _ in range(num_users)]
    for user, item in edges:
        positives[int(user) - 1].add(int(item) - 1)
    return positives


def sample_negative_items(users_zero, positives, num_items, rng):
    negatives = rng.integers(0, num_items, size=len(users_zero), dtype=np.int64)
    for index, user in enumerate(users_zero.tolist()):
        forbidden = positives[user]
        if len(forbidden) >= num_items:
            raise RuntimeError(
                "Target user {} interacted with every target item.".format(user + 1)
            )
        while int(negatives[index]) in forbidden:
            negatives[index] = rng.integers(0, num_items)
    return negatives


def split_lightgcn_edges(edges, validation_ratio, seed):
    rng = np.random.default_rng(seed)
    by_user = defaultdict(list)
    for index, (user, _item) in enumerate(edges):
        by_user[int(user)].append(index)
    candidates = []
    for indices in by_user.values():
        if len(indices) >= 2:
            candidates.append(int(rng.choice(indices)))
    target_count = min(
        len(candidates), max(1, int(round(len(edges) * float(validation_ratio))))
    )
    validation_indices = set(
        rng.choice(np.asarray(candidates), size=target_count, replace=False).tolist()
    )
    train_mask = np.asarray(
        [index not in validation_indices for index in range(len(edges))], dtype=np.bool_
    )
    return edges[train_mask], edges[~train_mask]


def lightgcn_bpr(model, edge_sample, positives, num_items, rng, regularization):
    users = edge_sample[:, 0].astype(np.int64) - 1
    positive_items = edge_sample[:, 1].astype(np.int64) - 1
    negative_items = sample_negative_items(users, positives, num_items, rng)
    device = model.user_embedding.weight.device
    users_t = torch.as_tensor(users, dtype=torch.long, device=device)
    positives_t = torch.as_tensor(positive_items, dtype=torch.long, device=device)
    negatives_t = torch.as_tensor(negative_items, dtype=torch.long, device=device)
    user_graph, item_graph = model.propagate()
    positive_scores = (user_graph[users_t] * item_graph[positives_t]).sum(dim=-1)
    negative_scores = (user_graph[users_t] * item_graph[negatives_t]).sum(dim=-1)
    ranking_loss = -F.logsigmoid(positive_scores - negative_scores).mean()
    if regularization <= 0:
        return ranking_loss
    raw_norm = (
        model.user_embedding(users_t).pow(2).sum()
        + model.item_embedding(positives_t).pow(2).sum()
        + model.item_embedding(negatives_t).pow(2).sum()
    ) / (2.0 * len(users))
    return ranking_loss + regularization * raw_norm


def train_lightgcn_once(
    config,
    train_edges,
    validation_edges,
    all_positives,
    num_users,
    num_items,
    epochs,
    early_stopping,
    seed,
):
    settings = config["lightgcn"]
    device = torch.device(config["device"])
    set_seed(seed)
    adjacency = normalized_graph(num_users, num_items, train_edges, device)
    model = TargetLightGCN(
        num_users,
        num_items,
        int(settings["embedding_dim"]),
        int(settings["layers"]),
        adjacency,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=0.0,
    )
    rng = np.random.default_rng(seed)
    max_samples = int(settings.get("samples_per_epoch", 200000))
    max_eval = int(settings.get("validation_samples", 50000))
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(int(epochs)):
        model.train()
        if len(train_edges) > max_samples:
            indices = rng.choice(len(train_edges), size=max_samples, replace=False)
            sampled_edges = train_edges[indices]
        else:
            sampled_edges = train_edges
        optimizer.zero_grad()
        loss = lightgcn_bpr(
            model,
            sampled_edges,
            all_positives,
            num_items,
            rng,
            float(settings.get("weight_decay", 0.0)),
        )
        loss.backward()
        optimizer.step()

        model.eval()
        evaluation_edges = validation_edges if len(validation_edges) else sampled_edges
        if len(evaluation_edges) > max_eval:
            evaluation_edges = evaluation_edges[:max_eval]
        with torch.no_grad():
            validation_loss = float(
                lightgcn_bpr(
                    model,
                    evaluation_edges,
                    all_positives,
                    num_items,
                    rng,
                    0.0,
                ).cpu()
            )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            LOGGER.info(
                "LightGCN epoch %d: train=%.6f validation=%.6f best_epoch=%d",
                epoch + 1,
                float(loss.detach().cpu()),
                validation_loss,
                best_epoch,
            )
        if early_stopping and stale >= int(settings["patience"]):
            break
    if best_state is None:
        raise RuntimeError("LightGCN did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def create_lightgcn(config, output_dir, train_tgt, id_mapping, force=False):
    user_embedding_path = os.path.join(
        output_dir, "target_lightgcn_user_embeddings.npy"
    )
    embedding_path = os.path.join(output_dir, "target_lightgcn_item_embeddings.npy")
    checkpoint_path = os.path.join(output_dir, "target_lightgcn.pt")
    if force:
        for path in (user_embedding_path, embedding_path, checkpoint_path):
            if os.path.isfile(path):
                os.remove(path)

    edges = np.asarray(train_tgt[["user", "item"]], dtype=np.int64)
    edges = np.unique(edges, axis=0)
    num_users = len(id_mapping["tgt"]["id2user"]) - 1
    num_items = len(id_mapping["tgt"]["id2item"]) - 1
    if os.path.isfile(embedding_path) and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not os.path.isfile(user_embedding_path):
            device = torch.device(config["device"])
            adjacency = normalized_graph(num_users, num_items, edges, device)
            model = TargetLightGCN(
                num_users,
                num_items,
                int(checkpoint["embedding_dim"]),
                int(checkpoint["layers"]),
                adjacency,
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                users, _items = model.propagate()
                users = F.normalize(users, dim=-1).cpu().numpy().astype(np.float32)
            padded_users = np.zeros(
                (num_users + 1, users.shape[1]), dtype=np.float32
            )
            padded_users[1:] = users
            np.save(user_embedding_path, padded_users)
            LOGGER.info(
                "Materialized %d frozen target LightGCN user embeddings.", num_users
            )
        return (
            np.load(user_embedding_path),
            np.load(embedding_path),
            checkpoint,
        )

    positives = interaction_sets(edges, num_users)
    settings = config["lightgcn"]
    train_edges, validation_edges = split_lightgcn_edges(
        edges, settings["validation_ratio"], int(config["seed"])
    )
    LOGGER.info(
        "Training target-only LightGCN with %d train and %d validation edges.",
        len(train_edges),
        len(validation_edges),
    )
    _selection_model, best_epoch, best_loss = train_lightgcn_once(
        config,
        train_edges,
        validation_edges,
        positives,
        num_users,
        num_items,
        int(settings["max_epochs"]),
        True,
        int(config["seed"]),
    )
    LOGGER.info(
        "Refitting target-only LightGCN on all target training edges for %d epochs.",
        best_epoch,
    )
    final_model, _unused_epoch, _unused_loss = train_lightgcn_once(
        config,
        edges,
        np.empty((0, 2), dtype=np.int64),
        positives,
        num_users,
        num_items,
        best_epoch,
        False,
        int(config["seed"]),
    )
    final_model.eval()
    with torch.no_grad():
        users, items = final_model.propagate()
        users = F.normalize(users, dim=-1).cpu().numpy().astype(np.float32)
        items = F.normalize(items, dim=-1).cpu().numpy().astype(np.float32)
    padded_users = np.zeros((num_users + 1, users.shape[1]), dtype=np.float32)
    padded_users[1:] = users
    padded_items = np.zeros((num_items + 1, items.shape[1]), dtype=np.float32)
    padded_items[1:] = items
    np.save(user_embedding_path, padded_users)
    np.save(embedding_path, padded_items)
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": int(settings["embedding_dim"]),
        "layers": int(settings["layers"]),
        "selected_epochs": best_epoch,
        "selection_validation_bpr": best_loss,
        "train_interactions": len(edges),
        "seed": int(config["seed"]),
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    return padded_users, padded_items, checkpoint


def build_user_bridge_examples(train_tgt, semantic_items, settings, seed):
    """Create deterministic target-user semantic queries and held-out positives."""
    num_users = int(train_tgt["user"].max())
    histories = interaction_sets(
        np.asarray(train_tgt[["user", "item"]], dtype=np.int64), num_users
    )
    eligible_users = np.asarray(
        [user for user, items in enumerate(histories) if len(items) >= 2],
        dtype=np.int64,
    )
    if len(eligible_users) < 2:
        raise ValueError("Not enough target users to train a user-level bridge.")

    rng = np.random.default_rng(seed)
    shuffled_users = rng.permutation(eligible_users)
    requested_validation = int(
        round(len(shuffled_users) * float(settings["validation_ratio"]))
    )
    validation_count = min(
        max(1, requested_validation), int(settings.get("validation_users", 1024))
    )
    validation_users = shuffled_users[:validation_count]
    training_users = shuffled_users[validation_count:]
    ordered_users = np.concatenate([training_users, validation_users])
    max_context_items = int(settings.get("max_context_items", 32))
    queries = np.empty(
        (len(ordered_users), semantic_items.shape[1]), dtype=np.float16
    )
    positives = np.empty(len(ordered_users), dtype=np.int64)
    contexts = []

    for row, user in enumerate(ordered_users.tolist()):
        items = np.asarray(sorted(histories[user]), dtype=np.int64)
        held_out_position = int(rng.integers(0, len(items)))
        positives[row] = int(items[held_out_position])
        context = np.delete(items, held_out_position)
        if len(context) > max_context_items:
            context = np.sort(
                rng.choice(context, size=max_context_items, replace=False)
            )
        query = np.asarray(semantic_items[context], dtype=np.float32).mean(axis=0)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        queries[row] = query.astype(np.float16)
        contexts.append(context)
        if (row + 1) % 25000 == 0 or row + 1 == len(ordered_users):
            LOGGER.info(
                "Prepared target-user semantic queries: %d/%d.",
                row + 1,
                len(ordered_users),
            )

    split = len(training_users)
    return {
        "training_users": ordered_users[:split],
        "training_queries": queries[:split],
        "training_positives": positives[:split],
        "validation_users": ordered_users[split:],
        "validation_queries": queries[split:],
        "validation_positives": positives[split:],
        "validation_contexts": contexts[split:],
        "histories": histories,
    }


def sample_user_bridge_negatives(users, histories, num_items, count, rng):
    negatives = np.empty((len(users), count), dtype=np.int64)
    for row, user in enumerate(users.tolist()):
        forbidden = histories[int(user)]
        for column in range(count):
            while True:
                candidate = int(rng.integers(0, num_items))
                if candidate not in forbidden:
                    negatives[row, column] = candidate
                    break
    return negatives


def user_bridge_batch_loss(
    model,
    semantic_queries,
    collaborative_users,
    positive_items,
    negative_items,
    settings,
):
    predicted = model(semantic_queries)
    collaborative_users = F.normalize(collaborative_users, dim=-1)
    positive_items = F.normalize(positive_items, dim=-1)
    negative_items = F.normalize(negative_items, dim=-1)
    ranking_temperature = float(settings.get("ranking_temperature", 0.2))
    positive_scores = (predicted * positive_items).sum(dim=-1, keepdim=True)
    negative_scores = (predicted.unsqueeze(1) * negative_items).sum(dim=-1)
    ranking = F.softplus(
        -(positive_scores - negative_scores) / ranking_temperature
    ).mean()
    alignment = (1.0 - (predicted * collaborative_users).sum(dim=-1)).mean()

    contrastive_weight = float(settings.get("contrastive_weight", 0.0))
    if contrastive_weight > 0:
        temperature = float(settings.get("temperature", 0.07))
        logits = torch.matmul(predicted, collaborative_users.t()) / temperature
        labels = torch.arange(len(predicted), device=predicted.device)
        contrastive = F.cross_entropy(logits, labels)
    else:
        contrastive = predicted.new_zeros(())

    minimum_std = float(settings.get("minimum_output_std", 0.0))
    if minimum_std > 0 and len(predicted) > 1:
        variance = F.relu(
            minimum_std - predicted.std(dim=0, unbiased=False)
        ).mean()
    else:
        variance = predicted.new_zeros(())

    loss = (
        ranking
        + float(settings.get("alignment_weight", 0.2)) * alignment
        + contrastive_weight * contrastive
        + float(settings.get("variance_weight", 0.0)) * variance
    )
    return loss, ranking, alignment, contrastive, variance


def evaluate_user_bridge(
    model,
    examples,
    collaborative_users,
    collaborative_items,
    settings,
    device,
):
    model.eval()
    queries = examples["validation_queries"]
    users = examples["validation_users"]
    positives = examples["validation_positives"]
    contexts = examples["validation_contexts"]
    top_k = min(int(settings.get("validation_top_k", 20)), len(collaborative_items))
    batch_size = int(settings.get("validation_batch_size", 64))
    item_tensor = torch.as_tensor(
        collaborative_items, dtype=torch.float32, device=device
    )
    hits = 0
    ndcg = 0.0
    alignment_total = 0.0

    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            stop = min(start + batch_size, len(users))
            source = torch.as_tensor(
                queries[start:stop], dtype=torch.float32, device=device
            )
            predicted = model(source)
            target_users = torch.as_tensor(
                collaborative_users[users[start:stop]],
                dtype=torch.float32,
                device=device,
            )
            target_users = F.normalize(target_users, dim=-1)
            alignment_total += float(
                (1.0 - (predicted * target_users).sum(dim=-1)).sum().cpu()
            )
            scores = torch.matmul(predicted, item_tensor.t())
            for row, context in enumerate(contexts[start:stop]):
                if len(context):
                    context_tensor = torch.as_tensor(
                        context, dtype=torch.long, device=device
                    )
                    scores[row, context_tensor] = -float("inf")
            selected = torch.topk(scores, k=top_k, dim=1).indices.cpu().numpy()
            for row, positive in enumerate(positives[start:stop].tolist()):
                positions = np.flatnonzero(selected[row] == int(positive))
                if len(positions):
                    hits += 1
                    ndcg += 1.0 / np.log2(float(positions[0]) + 2.0)

    count = max(1, len(users))
    return {
        "recall": hits / float(count),
        "ndcg": ndcg / float(count),
        "alignment": alignment_total / float(count),
        "top_k": top_k,
    }


def create_bridge(
    config,
    output_dir,
    semantic_items,
    collaborative_users,
    collaborative_items,
    train_tgt,
    force=False,
):
    checkpoint_path = os.path.join(
        output_dir, "user_semantic_to_collaborative_bridge.pt"
    )
    if force and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    settings = config["bridge"]
    input_dim = semantic_items.shape[1]
    output_dim = collaborative_items.shape[1]
    device = torch.device(config["device"])
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("bridge_type") != BRIDGE_TYPE:
            raise ValueError(
                "Cached bridge has type {!r}, expected {!r}.".format(
                    checkpoint.get("bridge_type"), BRIDGE_TYPE
                )
            )
        model = TargetPatternBridge(
            checkpoint["input_dim"],
            checkpoint["hidden_dim"],
            checkpoint["output_dim"],
            checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device), checkpoint

    if len(semantic_items) != len(collaborative_items):
        raise ValueError(
            "Target semantic and collaborative item arrays are misaligned."
        )
    if int(train_tgt["user"].max()) != len(collaborative_users):
        raise ValueError(
            "Target interactions and collaborative user rows are misaligned."
        )

    examples = build_user_bridge_examples(
        train_tgt, semantic_items, settings, int(config["seed"])
    )
    LOGGER.info(
        "Training user-level semantic bridge with %d training and %d validation users.",
        len(examples["training_users"]),
        len(examples["validation_users"]),
    )
    set_seed(int(config["seed"]))
    model = TargetPatternBridge(
        input_dim,
        int(settings["hidden_dim"]),
        output_dim,
        float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["seed"]) + 1009)
    batch_size = int(settings["batch_size"])
    users_per_epoch = min(
        len(examples["training_users"]),
        int(settings.get("users_per_epoch", len(examples["training_users"]))),
    )
    negative_count = int(settings.get("negatives_per_positive", 4))
    best_state = None
    best_epoch = 0
    best_metrics = None
    best_key = None
    stale = 0

    for epoch in range(int(settings["max_epochs"])):
        model.train()
        if users_per_epoch < len(examples["training_users"]):
            epoch_indices = rng.choice(
                len(examples["training_users"]),
                size=users_per_epoch,
                replace=False,
            )
        else:
            epoch_indices = rng.permutation(len(examples["training_users"]))
        totals = np.zeros(5, dtype=np.float64)
        count = 0

        for start in range(0, len(epoch_indices), batch_size):
            batch = epoch_indices[start : start + batch_size]
            if len(batch) < 2:
                continue
            users = examples["training_users"][batch]
            positives = examples["training_positives"][batch]
            negatives = sample_user_bridge_negatives(
                users,
                examples["histories"],
                len(collaborative_items),
                negative_count,
                rng,
            )
            source = torch.as_tensor(
                examples["training_queries"][batch],
                dtype=torch.float32,
                device=device,
            )
            user_target = torch.as_tensor(
                collaborative_users[users], dtype=torch.float32, device=device
            )
            positive_target = torch.as_tensor(
                collaborative_items[positives], dtype=torch.float32, device=device
            )
            negative_target = torch.as_tensor(
                collaborative_items[negatives], dtype=torch.float32, device=device
            )
            optimizer.zero_grad()
            losses = user_bridge_batch_loss(
                model,
                source,
                user_target,
                positive_target,
                negative_target,
                settings,
            )
            losses[0].backward()
            optimizer.step()
            totals += np.asarray(
                [float(value.detach().cpu()) for value in losses], dtype=np.float64
            ) * len(batch)
            count += len(batch)

        metrics = evaluate_user_bridge(
            model,
            examples,
            collaborative_users,
            collaborative_items,
            settings,
            device,
        )
        averages = totals / max(1, count)
        selection_key = (
            round(float(metrics["recall"]), 12),
            round(float(metrics["ndcg"]), 12),
            -round(float(metrics["alignment"]), 12),
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch + 1
            best_metrics = dict(metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        LOGGER.info(
            "User bridge epoch %d: loss=%.6f rank=%.6f align=%.6f "
            "R@%d=%.6f N@%d=%.6f best_epoch=%d",
            epoch + 1,
            averages[0],
            averages[1],
            averages[2],
            metrics["top_k"],
            metrics["recall"],
            metrics["top_k"],
            metrics["ndcg"],
            best_epoch,
        )
        if stale >= int(settings["patience"]):
            break

    if best_state is None:
        raise RuntimeError("User bridge training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    checkpoint = {
        "bridge_type": BRIDGE_TYPE,
        "state_dict": best_state,
        "input_dim": input_dim,
        "hidden_dim": int(settings["hidden_dim"]),
        "output_dim": output_dim,
        "dropout": float(settings["dropout"]),
        "best_epoch": best_epoch,
        "validation_recall": float(best_metrics["recall"]),
        "validation_ndcg": float(best_metrics["ndcg"]),
        "validation_top_k": int(best_metrics["top_k"]),
        "validation_alignment": float(best_metrics["alignment"]),
        "training_users": int(len(examples["training_users"])),
        "validation_users": int(len(examples["validation_users"])),
        "max_context_items": int(settings.get("max_context_items", 32)),
        "seed": int(config["seed"]),
    }
    atomic_torch_save(checkpoint_path, checkpoint)
    return model, checkpoint


def load_frozen_transfer_components(config, output_dir, id_mapping):
    embedding_path = os.path.join(output_dir, "target_lightgcn_item_embeddings.npy")
    lightgcn_path = os.path.join(output_dir, "target_lightgcn.pt")
    bridge_path = os.path.join(
        output_dir, "user_semantic_to_collaborative_bridge.pt"
    )
    missing = [
        path
        for path in (embedding_path, lightgcn_path, bridge_path)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Overlap retrieval requires frozen cold-pipeline artifacts; missing {}."
            .format(", ".join(missing))
        )

    collaborative_items = np.load(embedding_path)
    lightgcn_checkpoint = torch.load(lightgcn_path, map_location="cpu")
    bridge_checkpoint = torch.load(bridge_path, map_location="cpu")
    if bridge_checkpoint.get("bridge_type") != BRIDGE_TYPE:
        raise ValueError(
            "Frozen bridge has type {!r}, expected {!r}.".format(
                bridge_checkpoint.get("bridge_type"), BRIDGE_TYPE
            )
        )
    expected_items = len(id_mapping["tgt"]["id2item"])
    if len(collaborative_items) != expected_items:
        raise ValueError(
            "Frozen LightGCN item rows do not match the active target item mapping."
        )
    if int(lightgcn_checkpoint["num_items"]) + 1 != expected_items:
        raise ValueError(
            "Frozen LightGCN checkpoint does not match the active target item mapping."
        )
    if collaborative_items.shape[1] != int(bridge_checkpoint["output_dim"]):
        raise ValueError(
            "Frozen bridge output dimension does not match LightGCN item embeddings."
        )

    bridge = TargetPatternBridge(
        bridge_checkpoint["input_dim"],
        bridge_checkpoint["hidden_dim"],
        bridge_checkpoint["output_dim"],
        bridge_checkpoint["dropout"],
    )
    bridge.load_state_dict(bridge_checkpoint["state_dict"])
    bridge = bridge.to(torch.device(config["device"]))
    bridge.eval()
    LOGGER.info(
        "Loaded frozen target LightGCN and user-level semantic bridge for overlap retrieval."
    )
    return collaborative_items, lightgcn_checkpoint, bridge, bridge_checkpoint


def stable_top_k(scores, k):
    if k >= len(scores):
        candidates = np.arange(len(scores))
    else:
        candidates = np.argpartition(scores, -k)[-k:]
    order = np.lexsort((candidates, -scores[candidates]))
    selected = candidates[order[:k]]
    return selected, scores[selected]


def retrieve_pseudo_rankings(
    config,
    output_dir,
    user_interests,
    interest_embeddings,
    interest_keys,
    bridge,
    collaborative_items,
    id_mapping,
    all_users,
    lightgcn_checkpoint,
    bridge_checkpoint,
    user_scope="cold",
    force=False,
):
    if user_scope not in ("cold", "overlap"):
        raise ValueError("Unsupported user scope: {}".format(user_scope))
    ranking_name = (
        "pseudo_rankings.npz"
        if user_scope == "cold"
        else "overlap_pseudo_rankings.npz"
    )
    ranking_path = os.path.join(output_dir, ranking_name)
    manifest_path = (
        os.path.join(output_dir, "manifest.json") if user_scope == "cold" else None
    )
    if force:
        for path in (ranking_path, manifest_path):
            if path and os.path.isfile(path):
                os.remove(path)
    if os.path.isfile(ranking_path) and (
        user_scope == "overlap" or os.path.isfile(manifest_path)
    ):
        return ranking_path, manifest_path

    key_to_row = {key: index for index, key in enumerate(interest_keys)}
    device = torch.device(config["device"])
    bridge.eval()
    target_items = np.asarray(collaborative_items[1:], dtype=np.float32)
    target_items /= np.maximum(
        np.linalg.norm(target_items, axis=1, keepdims=True), 1e-12
    )
    target_item_tensor = torch.as_tensor(
        target_items, dtype=torch.float32, device=device
    )

    users = sorted(int(user) for user in user_interests)
    official_k = int(config["retrieval"]["official_top_k"])
    cached_k = min(int(config["retrieval"]["cached_top_k"]), len(target_items))
    if official_k > cached_k:
        raise ValueError(
            "retrieval.official_top_k cannot exceed retrieval.cached_top_k."
        )
    ranked_items = np.empty((len(users), cached_k), dtype=np.int64)
    ranked_scores = np.empty((len(users), cached_k), dtype=np.float32)
    user_batch_size = int(config["retrieval"].get("user_batch_size", 128))
    with torch.no_grad():
        for start in range(0, len(users), user_batch_size):
            batch_users = users[start : start + user_batch_size]
            semantic_queries = np.zeros(
                (len(batch_users), interest_embeddings.shape[1]), dtype=np.float32
            )
            for batch_row, user in enumerate(batch_users):
                record = user_interests[str(user)]
                for interest_index, interest in enumerate(record["interests"]):
                    key = "{}:{}".format(user, interest_index)
                    if key not in key_to_row:
                        raise KeyError("Missing embedded interest {}".format(key))
                    semantic_queries[batch_row] += float(interest["weight"]) * np.asarray(
                        interest_embeddings[key_to_row[key]], dtype=np.float32
                    )
            semantic_queries /= np.maximum(
                np.linalg.norm(semantic_queries, axis=1, keepdims=True), 1e-12
            )
            source = torch.as_tensor(
                semantic_queries, dtype=torch.float32, device=device
            )
            mapped_queries = bridge(source)
            score_matrix = torch.matmul(mapped_queries, target_item_tensor.t())
            values, selected = torch.topk(
                score_matrix, k=cached_k, dim=1, largest=True, sorted=True
            )
            stop = start + len(batch_users)
            ranked_items[start:stop] = selected.cpu().numpy().astype(np.int64) + 1
            ranked_scores[start:stop] = values.cpu().numpy().astype(np.float32)
            completed = min(stop, len(users))
            if completed % 1000 < len(batch_users) or completed == len(users):
                LOGGER.info("Pseudo retrieval: %d/%d users.", completed, len(users))

    atomic_npz(
        ranking_path,
        user_ids=np.asarray(users, dtype=np.int64),
        item_ids=ranked_items,
        scores=ranked_scores,
    )
    if user_scope == "overlap":
        LOGGER.info(
            "Saved overlap-user pseudo cache: %d users x Top-%d (%s).",
            len(users),
            cached_k,
            ranking_path,
        )
        return ranking_path, None

    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset": config["dataset"],
        "domains": list(config["domains"]),
        "num_users_overlap": len(all_users["overlap_users"]),
        "num_users_src": len(id_mapping["src"]["id2user"]) - 1,
        "num_items_tgt": len(id_mapping["tgt"]["id2item"]) - 1,
        "mapping_sha256": mapping_fingerprint(id_mapping),
        "official_top_k": official_k,
        "cached_top_k": cached_k,
        "cold_source_users": len(users),
        "pseudo_pairs": len(users) * official_k,
        "user_interest_schema_version": USER_INTEREST_SCHEMA_VERSION,
        "interest_generation": "aligned_source_target_single_call",
        "target_context_examples": int(
            config["llm"].get("target_context_examples", 24)
        ),
        "llm_model": config["llm"]["model"],
        "embedding_model": config["embedding"]["model"],
        "embedding_dimensions": int(config["embedding"]["dimensions"]),
        "bridge_type": bridge_checkpoint["bridge_type"],
        "lightgcn_selected_epochs": lightgcn_checkpoint["selected_epochs"],
        "bridge_best_epoch": bridge_checkpoint["best_epoch"],
        "bridge_validation_recall": bridge_checkpoint["validation_recall"],
        "bridge_validation_ndcg": bridge_checkpoint["validation_ndcg"],
        "bridge_validation_top_k": bridge_checkpoint["validation_top_k"],
        "seed": int(config["seed"]),
        "created_at": datetime.now().astimezone().isoformat(),
        "rankings_sha256": file_sha256(ranking_path),
    }
    atomic_json(manifest_path, manifest)
    LOGGER.info(
        "Saved one shared pseudo cache: %d users x Top-%d (%s).",
        len(users),
        official_k,
        ranking_path,
    )
    return ranking_path, manifest_path


def aggregate_overlap_metrics(ranked_items, user_ids, ground_truth, cutoffs):
    totals = {
        cutoff: {
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "hits": 0,
            "ground_truth": 0,
        }
        for cutoff in cutoffs
    }
    for row, user in enumerate(user_ids.tolist()):
        relevant = ground_truth[int(user)]
        relevant_count = len(relevant)
        for cutoff in cutoffs:
            predictions = ranked_items[row, :cutoff]
            hit_mask = np.asarray(
                [int(item) in relevant for item in predictions], dtype=np.float64
            )
            hits = int(hit_mask.sum())
            discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
            dcg = float(np.dot(hit_mask, discounts))
            ideal_count = min(relevant_count, cutoff)
            ideal_dcg = float(discounts[:ideal_count].sum())
            totals[cutoff]["precision"] += hits / float(cutoff)
            totals[cutoff]["recall"] += hits / float(relevant_count)
            totals[cutoff]["hit_rate"] += float(hits > 0)
            totals[cutoff]["ndcg"] += dcg / max(ideal_dcg, 1e-12)
            totals[cutoff]["hits"] += hits
            totals[cutoff]["ground_truth"] += relevant_count

    user_count = len(user_ids)
    metrics = {}
    for cutoff in cutoffs:
        values = totals[cutoff]
        metrics[str(cutoff)] = {
            "precision": values["precision"] / user_count,
            "recall": values["recall"] / user_count,
            "hit_rate": values["hit_rate"] / user_count,
            "ndcg": values["ndcg"] / user_count,
            "mean_hits": values["hits"] / float(user_count),
            "micro_recall": values["hits"] / float(values["ground_truth"]),
        }
    return metrics


def evaluate_overlap_rankings(
    config,
    output_dir,
    ranking_path,
    train_tgt,
    all_users,
    id_mapping,
    force=False,
):
    quality_path = os.path.join(output_dir, "overlap_pseudo_quality.json")
    if force and os.path.isfile(quality_path):
        os.remove(quality_path)
    if os.path.isfile(quality_path):
        return quality_path

    cache = np.load(ranking_path, allow_pickle=False)
    user_ids = np.asarray(cache["user_ids"], dtype=np.int64)
    ranked_items = np.asarray(cache["item_ids"], dtype=np.int64)
    overlap_count = len(all_users["overlap_users"])
    expected_users = np.arange(1, overlap_count + 1, dtype=np.int64)
    if not np.array_equal(user_ids, expected_users):
        raise ValueError(
            "Overlap pseudo rankings are not aligned to contiguous overlap user IDs."
        )

    ground_truth = defaultdict(set)
    for user, item in train_tgt[["user", "item"]].itertuples(
        index=False, name=None
    ):
        user = int(user)
        if 0 < user <= overlap_count:
            ground_truth[user].add(int(item))
    missing_truth = [user for user in expected_users if not ground_truth[int(user)]]
    if missing_truth:
        raise ValueError(
            "{} overlap users have no target-domain training interactions."
            .format(len(missing_truth))
        )

    width = ranked_items.shape[1]
    cutoffs = [value for value in (1, 5, 10, 20, 50, 100) if value <= width]
    num_items = len(id_mapping["tgt"]["id2item"]) - 1
    candidates = np.arange(1, num_items + 1, dtype=np.int64)
    popularity = np.bincount(
        np.asarray(train_tgt["item"], dtype=np.int64), minlength=num_items + 1
    )[1:]
    popularity_order = np.lexsort((candidates, -popularity))
    popularity_items = np.tile(
        candidates[popularity_order[:width]], (len(user_ids), 1)
    )
    rng = np.random.default_rng(int(config["seed"]))
    random_items = np.empty_like(ranked_items)
    for row in range(len(user_ids)):
        random_items[row] = rng.choice(candidates, size=width, replace=False)

    relevant_counts = np.asarray(
        [len(ground_truth[int(user)]) for user in user_ids], dtype=np.int64
    )
    result = {
        "format_version": CACHE_FORMAT_VERSION,
        "dataset": config["dataset"],
        "domains": list(config["domains"]),
        "evaluation_scope": "overlap_users_train_tgt_diagnostic",
        "source_prompt_uses_target_interactions": False,
        "caveat": (
            "Warm validation/test sets are empty. Target train interactions are used "
            "only as diagnostic relevance labels; these are not independent test metrics."
        ),
        "num_overlap_users": int(len(user_ids)),
        "num_target_items": int(num_items),
        "ranking_width": int(width),
        "total_ground_truth_interactions": int(relevant_counts.sum()),
        "mean_ground_truth_items_per_user": float(relevant_counts.mean()),
        "cutoffs": cutoffs,
        "metrics": {
            "target_pattern": aggregate_overlap_metrics(
                ranked_items, user_ids, ground_truth, cutoffs
            ),
            "popularity": aggregate_overlap_metrics(
                popularity_items, user_ids, ground_truth, cutoffs
            ),
            "random": aggregate_overlap_metrics(
                random_items, user_ids, ground_truth, cutoffs
            ),
        },
        "seed": int(config["seed"]),
        "rankings_sha256": file_sha256(ranking_path),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(quality_path, result)
    LOGGER.info("Saved overlap pseudo-quality diagnostics to %s.", quality_path)
    return quality_path


def records_for_embeddings(target_patterns, user_interests):
    target_records = [
        (str(record["item_id"]), record["target_item_pattern"])
        for record in sorted(target_patterns.values(), key=lambda row: row["item_id"])
    ]
    interest_records = []
    for user in sorted(int(value) for value in user_interests):
        record = user_interests[str(user)]
        for interest_index, interest in enumerate(record["interests"]):
            interest_records.append(
                (
                    "{}:{}".format(user, interest_index),
                    interest["target_item_pattern"],
                )
            )
    return target_records, interest_records


def phase_index(phase):
    return len(PHASES) - 1 if phase == "all" else PHASES.index(phase)


def main():
    args = parse_args()
    config = load_config(args.config, args.dataset, args.domains, args.device)
    set_seed(int(config["seed"]))
    split_dir, all_users, id_mapping, train_src, train_tgt = load_split(config)
    output_dir = os.path.join(split_dir, config["output_subdir"])
    configure_logging(output_dir)
    LOGGER.info(
        "Dataset=%s domains=%s device=%s user_scope=%s",
        args.dataset,
        "+".join(args.domains),
        config["device"],
        args.user_scope,
    )
    LOGGER.info("Shared output directory: %s", output_dir)
    last_phase = phase_index(args.phase)

    metadata_path = os.path.join(
        output_dir,
        "metadata.json" if args.user_scope == "cold" else "overlap_metadata.json",
    )
    if (
        args.force_phase
        and args.phase in ("all", "metadata")
        and os.path.isfile(metadata_path)
    ):
        os.remove(metadata_path)
    artifact = build_metadata_artifacts(
        config,
        output_dir,
        all_users,
        id_mapping,
        train_src,
        user_scope=args.user_scope,
    )
    if last_phase == 0:
        return

    target_patterns = create_target_patterns(
        config,
        output_dir,
        artifact,
        id_mapping,
        force=args.force_phase and args.phase in ("all", "target-patterns"),
    )
    if last_phase == 1:
        return

    user_interests = create_user_interests(
        config,
        output_dir,
        artifact,
        target_patterns,
        all_users,
        id_mapping,
        train_src,
        user_scope=args.user_scope,
        force=args.force_phase and args.phase in ("all", "user-interests"),
    )
    if last_phase == 2:
        return

    target_records, interest_records = records_for_embeddings(
        target_patterns, user_interests
    )
    force_embeddings = args.force_phase and args.phase in ("all", "embeddings")
    target_embeddings, target_keys = create_embeddings(
        config,
        output_dir,
        "target_item_semantic_embeddings",
        target_records,
        force=force_embeddings and args.user_scope == "cold",
    )
    expected_target_keys = [
        str(index) for index in range(1, len(id_mapping["tgt"]["id2item"]))
    ]
    if target_keys != expected_target_keys:
        raise ValueError(
            "Target semantic embeddings are not aligned to target item IDs."
        )
    interest_embeddings, interest_keys = create_embeddings(
        config,
        output_dir,
        (
            "source_interest_semantic_embeddings"
            if args.user_scope == "cold"
            else "overlap_interest_semantic_embeddings"
        ),
        interest_records,
        force=force_embeddings,
    )
    if last_phase == 3:
        return

    if args.user_scope == "overlap":
        (
            collaborative_items,
            lightgcn_checkpoint,
            bridge,
            bridge_checkpoint,
        ) = load_frozen_transfer_components(config, output_dir, id_mapping)
        if last_phase in (4, 5):
            return
    else:
        (
            collaborative_users,
            collaborative_items,
            lightgcn_checkpoint,
        ) = create_lightgcn(
            config,
            output_dir,
            train_tgt,
            id_mapping,
            force=args.force_phase and args.phase in ("all", "lightgcn"),
        )
        if last_phase == 4:
            return

        bridge, bridge_checkpoint = create_bridge(
            config,
            output_dir,
            target_embeddings,
            collaborative_users[1:],
            collaborative_items[1:],
            train_tgt,
            force=args.force_phase and args.phase in ("all", "bridge"),
        )
        if last_phase == 5:
            return

    ranking_path, _manifest_path = retrieve_pseudo_rankings(
        config,
        output_dir,
        user_interests,
        interest_embeddings,
        interest_keys,
        bridge,
        collaborative_items,
        id_mapping,
        all_users,
        lightgcn_checkpoint,
        bridge_checkpoint,
        user_scope=args.user_scope,
        force=args.force_phase and args.phase in ("all", "retrieve"),
    )
    if args.user_scope == "overlap":
        evaluate_overlap_rankings(
            config,
            output_dir,
            ranking_path,
            train_tgt,
            all_users,
            id_mapping,
            force=args.force_phase and args.phase in ("all", "retrieve"),
        )


if __name__ == "__main__":
    main()
