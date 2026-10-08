import atexit
import csv
import json
import logging
import os
import time
from copy import deepcopy

import torch

from models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban import (
    CROSSDOMAIN_USER_SYSTEM_PROMPT,
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban,
    TGT_ITEM_SYSTEM_PROMPT,
    _build_all_user_prompt_strings,
    _build_item_prompt_string,
    _collect_domain_items,
    _collect_domain_user_items,
    _user_item_profile_to_string,
)
from models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban_end2end import (
    _ModalityAliasDataloader,
    _embed_text_batches,
    _get_src_only_users,
    _run_prompt_summarization_multiprocess,
    generate_src_only_tgt_only_embs,
    generate_src_only_tgt_only_final_embs,
    run_until_no_prompt_errors,
)


logger = logging.getLogger()

MODALITY_NAME = "CrossDomain_semantics_both_tag_douban_runtime_src_only_tgt_only"
DEFAULT_RUNTIME_DIR = "runtime_records/lsdcdr_book_music_runtime"


def _runtime_dir(config):
    path = config.get("runtime_record_dir", DEFAULT_RUNTIME_DIR)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def _runtime_json_path(config):
    return os.path.join(_runtime_dir(config), "runtime_summary.json")


def _runtime_csv_path(config):
    return os.path.join(_runtime_dir(config), "runtime_summary.csv")


def _read_runtime(config):
    path = _runtime_json_path(config)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _write_runtime(config, data):
    path = _runtime_json_path(config)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _reset_runtime(config, metadata):
    data = deepcopy(metadata)
    data.setdefault("llm_tag_extraction_sec", 0.0)
    data.setdefault("text_embedding_sec", 0.0)
    data.setdefault("pseudo_retrieval_sec", 0.0)
    data.setdefault("training_wall_sec", 0.0)
    data.setdefault("self_evolving_update_sec", 0.0)
    data["total_start_perf"] = time.perf_counter()
    data["total_start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_runtime(config, data)


def _update_runtime(config, updates):
    data = _read_runtime(config)
    data.update(updates)
    _write_runtime(config, data)
    return data


def _add_runtime(config, key, elapsed):
    data = _read_runtime(config)
    data[key] = float(data.get(key, 0.0)) + float(elapsed)
    _write_runtime(config, data)
    return data[key]


def _write_runtime_csv(config, data):
    columns = [
        ("llm_tag_extraction_sec", "LLM tag extraction"),
        ("text_embedding_sec", "Text embedding"),
        ("pseudo_retrieval_sec", "Pseudo retrieval"),
        ("diffusion_training_sec", "Diffusion training"),
        ("self_evolving_update_sec", "Self-evolving update"),
        ("total_sec", "Total"),
    ]
    path = _runtime_csv_path(config)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "seconds", "minutes"])
        for key, label in columns:
            seconds = float(data.get(key, 0.0))
            writer.writerow([label, f"{seconds:.6f}", f"{seconds / 60.0:.6f}"])


def _runtime_embed_text_batches(config, modality, texts):
    start = time.perf_counter()
    embs = _embed_text_batches(config, modality, texts)
    elapsed = time.perf_counter() - start
    batch_size = int(modality["emb_batch_size"])
    _add_runtime(config, "text_embedding_sec", elapsed)
    _update_runtime(
        config,
        {
            "text_embedding_model": modality["emb_model"],
            "text_embedding_num_texts": len(texts),
            "text_embedding_batches": (len(texts) + batch_size - 1) // batch_size,
            "text_embedding_batch_size": batch_size,
            "text_embedding_workers": int(modality.get("embedding_num_workers", 1)),
            "text_embedding_dim": int(embs.shape[1]),
        },
    )
    logger.info("[RUNTIME] Text embedding finished in %.2fs.", elapsed)
    return embs


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban_Runtime(
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban
):
    """Book->Music LSDCDR runtime profiler with src-only/tgt-only semantic generation."""

    def __init__(self, config, dataloader):
        self._runtime_finalized = False
        if "total_start_perf" not in _read_runtime(config):
            _update_runtime(
                config,
                {
                    "total_start_perf": time.perf_counter(),
                    "total_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "Semantic cache was likely reused; LLM/tag and embedding time may be zero.",
                },
            )
        super().__init__(config, _ModalityAliasDataloader(dataloader, MODALITY_NAME))
        atexit.register(self.write_runtime_summary)

    @torch.no_grad()
    def _build_src_only_pseudo_positives(self):
        start = time.perf_counter()
        result = super()._build_src_only_pseudo_positives()
        elapsed = time.perf_counter() - start
        _add_runtime(self.config, "pseudo_retrieval_sec", elapsed)
        _update_runtime(
            self.config,
            {
                "pseudo_retrieval_top_k": int(self.config["top_k_pos"]),
                "pseudo_retrieval_num_users": len(result),
                "pseudo_retrieval_num_pairs": sum(len(v) for v in result.values()),
            },
        )
        logger.info("[RUNTIME] Initial pseudo retrieval finished in %.2fs.", elapsed)
        return result

    @torch.no_grad()
    def rebuild_src_only_pseudo_with_ema(self):
        start = time.perf_counter()
        result = super().rebuild_src_only_pseudo_with_ema()
        elapsed = time.perf_counter() - start
        _add_runtime(self.config, "self_evolving_update_sec", elapsed)
        data = _read_runtime(self.config)
        _update_runtime(
            self.config,
            {
                "self_evolving_update_count": int(data.get("self_evolving_update_count", 0)) + 1,
                "self_evolving_update_last_sec": elapsed,
            },
        )
        return result

    def add_runtime_training_time(self, elapsed):
        _add_runtime(self.config, "training_wall_sec", elapsed)

    def write_runtime_summary(self):
        if self._runtime_finalized:
            return
        self._runtime_finalized = True
        data = _read_runtime(self.config)
        training_wall = float(data.get("training_wall_sec", 0.0))
        self_evolving = float(data.get("self_evolving_update_sec", 0.0))
        diffusion_training = max(0.0, training_wall - self_evolving)
        total_start_perf = data.get("total_start_perf")
        if total_start_perf is not None:
            total_sec = max(0.0, time.perf_counter() - float(total_start_perf))
        else:
            total_sec = (
                float(data.get("llm_tag_extraction_sec", 0.0))
                + float(data.get("text_embedding_sec", 0.0))
                + float(data.get("pseudo_retrieval_sec", 0.0))
                + diffusion_training
                + self_evolving
            )

        data.update(
            {
                "model": self.config.get("model"),
                "dataset": self.config.get("dataset"),
                "domains": self.config.get("domains"),
                "timesteps": self.config.get("timesteps"),
                "top_k_pos": self.config.get("top_k_pos"),
                "pseudo_start_epoch": self.config.get("pseudo_start_epoch"),
                "pseudo_update_interval": self.config.get("pseudo_update_interval"),
                "diffusion_training_sec": diffusion_training,
                "total_sec": total_sec,
                "total_end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        _write_runtime(self.config, data)
        _write_runtime_csv(self.config, data)
        logger.info("[RUNTIME] LSDCDR runtime summary saved to %s", _runtime_json_path(self.config))
        logger.info(
            "[RUNTIME] LLM tags %.2fs | Embedding %.2fs | Pseudo retrieval %.2fs | "
            "Diffusion training %.2fs | Self-evolving %.2fs | Total %.2fs",
            data.get("llm_tag_extraction_sec", 0.0),
            data.get("text_embedding_sec", 0.0),
            data.get("pseudo_retrieval_sec", 0.0),
            data.get("diffusion_training_sec", 0.0),
            data.get("self_evolving_update_sec", 0.0),
            data.get("total_sec", 0.0),
        )


def extract_CrossDomain_semantics_both_tag_douban_runtime_src_only_tgt_only_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    if config["dataset"] != "Douban":
        raise ValueError("This runtime LSDCDR variant is intended for Douban.")
    if modality.get("generation_scope", "src_only_tgt_only") != "src_only_tgt_only":
        raise ValueError("Expected generation_scope=src_only_tgt_only.")
    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for LLM tag generation.")

    _reset_runtime(
        config,
        {
            "model": config.get("model"),
            "dataset": config.get("dataset"),
            "domains": config.get("domains"),
            "modality": modality["name"],
            "llm_chat_model": modality.get("llm_chat_model", "gpt-4o-mini"),
            "embedding_model": modality["emb_model"],
            "generation_scope": "src_only_tgt_only",
        },
    )

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
    repair_max_rounds = int(modality.get("repair_max_rounds", 0))
    repair_sleep_seconds = float(modality.get("repair_sleep_seconds", 0.0))

    logger.info(
        "[RUNTIME] LSDCDR Book->Music runtime semantic generation: %d src-only users + %d target items.",
        len(user_prompts),
        len(item_prompts_tgt),
    )
    _update_runtime(
        config,
        {
            "profile_num_workers": profile_num_workers,
            "request_timeout": timeout,
            "num_src_only_users": len(user_prompts),
            "num_target_items": len(item_prompts_tgt),
        },
    )

    user_start = time.perf_counter()
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
    user_sec = time.perf_counter() - user_start

    item_start = time.perf_counter()
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
    item_sec = time.perf_counter() - item_start

    _update_runtime(
        config,
        {
            "llm_user_tag_extraction_sec": user_sec,
            "llm_item_tag_extraction_sec": item_sec,
            "llm_tag_extraction_sec": user_sec + item_sec,
            "num_failed_user_profiles": len(error_users),
            "num_failed_target_item_profiles": len(error_items_tgt),
        },
    )
    logger.info(
        "[RUNTIME] LLM tag extraction finished in %.2fs (users %.2fs, target items %.2fs).",
        user_sec + item_sec,
        user_sec,
        item_sec,
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


def generate_CrossDomain_semantics_both_tag_douban_runtime_src_only_tgt_only_embs(
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
        embed_text_batches=_runtime_embed_text_batches,
    )


def generate_CrossDomain_semantics_both_tag_douban_runtime_src_only_tgt_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    return generate_src_only_tgt_only_final_embs(config, modality, interaction, id_mapping, modality_embs)
