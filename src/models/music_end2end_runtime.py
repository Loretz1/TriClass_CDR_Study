import atexit
import csv
import json
import logging
import os
import re
import threading
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

import common.trainer as _common_trainer

from models.music import (
    MuSiC,
    _collect_music_user_text,
    _get_music_user_prompt,
    generate_CrossDomain_semantics_music_final_embs,
)
from models.music_end2end import (
    _ModalityAliasDataloader,
    _embed_text_batches,
    _id2raw,
    run_until_no_prompt_errors,
)


logger = logging.getLogger()

MODALITY_NAME = "CrossDomain_semantics_music_end2end_runtime_users_only"
DEFAULT_RUNTIME_DIR = "runtime_records/music_book_music_runtime"

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_STATES = {}
_RUNTIME_WALL_STARTS = {}
_RUNTIME_OFFLINE_STARTS = {}
_RUNTIME_CONFIGS = {}
_RUNTIME_OFFLINE_HANDLERS = {}
_RUNTIME_FINALIZED = set()

_BASE_TRAINER_CLASS = _common_trainer.Trainer

_PRIMARY_TIMING_KEYS = (
    "user_text_construction_sec",
    "source_profile_llm_sec",
    "target_profile_llm_sec",
    "user_profile_embedding_sec",
    "offline_preparation_sec",
    "training_core_sec",
    "validation_sec",
    "test_sec",
    "fit_wall_sec",
    "wall_clock_total_sec",
)


def _runtime_dir(config):
    path = config.get("runtime_record_dir", DEFAULT_RUNTIME_DIR)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def _runtime_json_path_from_dir(runtime_dir):
    return os.path.join(runtime_dir, "runtime_summary.json")


def _runtime_csv_path_from_dir(runtime_dir):
    return os.path.join(runtime_dir, "runtime_summary.csv")


def _write_runtime_locked(runtime_dir, data):
    path = _runtime_json_path_from_dir(runtime_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _write_runtime_csv_locked(runtime_dir, data):
    rows = (
        ("user_text_construction_sec", "User text construction"),
        ("source_profile_llm_sec", "Source-user profile LLM"),
        ("target_profile_llm_sec", "Target-user profile LLM"),
        ("user_profile_embedding_sec", "User-profile text embedding"),
        ("offline_other_sec", "Offline preparation (other)"),
        ("offline_preparation_sec", "Offline preparation (total)"),
        ("training_core_sec", "Training core"),
        ("training_stage_other_sec", "Model-training stage (other)"),
        ("model_training_sec", "Model training (excluding evaluation)"),
        ("validation_sec", "Validation"),
        ("test_sec", "Test"),
        ("evaluation_sec", "Evaluation (validation + test)"),
        ("fit_wall_sec", "Trainer fit wall-clock (including evaluation)"),
        ("end_to_end_other_sec", "End-to-end pipeline (other)"),
        ("total_excluding_evaluation_sec", "Total excluding evaluation"),
        ("wall_clock_total_sec", "Wall-clock total (including evaluation)"),
    )
    path = _runtime_csv_path_from_dir(runtime_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "seconds", "minutes"])
        for key, label in rows:
            seconds = float(data.get(key, 0.0))
            writer.writerow([label, f"{seconds:.6f}", f"{seconds / 60.0:.6f}"])
    os.replace(tmp_path, path)


def _reset_runtime(config, metadata):
    runtime_dir = _runtime_dir(config)
    now_perf = time.perf_counter()
    data = dict(metadata)
    for key in _PRIMARY_TIMING_KEYS:
        data[key] = 0.0
    data.update(
        {
            "offline_other_sec": 0.0,
            "evaluation_sec": 0.0,
            "training_stage_other_sec": 0.0,
            "model_training_sec": 0.0,
            "end_to_end_other_sec": 0.0,
            "total_excluding_evaluation_sec": 0.0,
            "wall_clock_start_perf": now_perf,
            "wall_clock_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cold_runtime_started": True,
            "embedding_generated_this_run": False,
            "fit_completed": False,
            "runtime_complete": False,
            "timing_protocol": {
                "wall_clock_start": "runtime semantic-extraction handler entry",
                "wall_clock_end": "process atexit, matching the existing LSDCDR total boundary",
                "offline_preparation_end": (
                    "processor modality-total log emitted after final semantic .npy persistence; "
                    "runtime model construction is the fallback boundary"
                ),
                "training_core": "sum of GPU-synchronized Trainer._train_epoch calls",
                "evaluation": "identity-separated, GPU-synchronized Trainer._valid_epoch calls",
                "model_training": "Trainer.fit wall-clock minus validation and test",
                "experimental_closure": (
                    "wall_clock_total_sec = offline_preparation_sec + fit_wall_sec + "
                    "end_to_end_other_sec"
                ),
                "total_excluding_evaluation": (
                    "wall_clock_total_sec - validation_sec - test_sec"
                ),
                "paper_aggregation": (
                    "The experimental timing partitions close independently; paper-level "
                    "component aggregation is performed separately."
                ),
            },
        }
    )
    with _RUNTIME_LOCK:
        _RUNTIME_STATES[runtime_dir] = data
        _RUNTIME_WALL_STARTS[runtime_dir] = now_perf
        _RUNTIME_OFFLINE_STARTS[runtime_dir] = now_perf
        _RUNTIME_CONFIGS[runtime_dir] = config
        _RUNTIME_FINALIZED.discard(runtime_dir)
        _write_runtime_locked(runtime_dir, data)
    _install_offline_boundary_handler(runtime_dir, config)
    return runtime_dir


def _ensure_runtime(config, metadata):
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        state = _RUNTIME_STATES.get(runtime_dir)
        if state:
            if config.get("require_fresh_runtime_cache", True) and not state.get("cold_runtime_started"):
                raise RuntimeError("The MuSiC runtime pipeline did not start from a cold cache.")
            return runtime_dir
    metadata = dict(metadata)
    metadata.update(
        {
            "semantic_cache_reused": True,
            "note": (
                "The runtime modality cache was reused, so offline preparation timings "
                "are zero. Remove all files whose names begin with the runtime modality "
                "name before a fresh end-to-end timing run."
            ),
        }
    )
    if config.get("require_fresh_runtime_cache", True):
        raise RuntimeError(
            f"Cached final embeddings were reused for {MODALITY_NAME}. Remove every cache file "
            f"whose name begins with '{MODALITY_NAME}' before the timed run."
        )
    runtime_dir = _reset_runtime(config, metadata)
    with _RUNTIME_LOCK:
        _RUNTIME_OFFLINE_STARTS.pop(runtime_dir, None)
    _remove_offline_boundary_handler(runtime_dir)
    return runtime_dir


def _require_active_cold_runtime(config, stage):
    if not config.get("require_fresh_runtime_cache", True):
        return
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        state = _RUNTIME_STATES.get(runtime_dir)
        if not state or not state.get("cold_runtime_started"):
            raise RuntimeError(
                f"Cannot time MuSiC {stage}: an earlier runtime modality cache was reused. "
                f"Delete every cache file beginning with '{MODALITY_NAME}' and rerun."
            )


def _require_embedding_generated_this_run(config, stage):
    if not config.get("require_fresh_runtime_cache", True):
        return
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        state = _RUNTIME_STATES.get(runtime_dir)
        generated = bool(state and state.get("embedding_generated_this_run"))
    if not generated:
        raise RuntimeError(
            f"Cannot continue MuSiC {stage}: the user-profile embedding cache was reused "
            "instead of being generated in this cold runtime. Delete every metadata, raw "
            f"embedding, and final-embedding cache beginning with '{MODALITY_NAME}' and rerun."
        )


def _update_runtime(config, updates):
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        data = _RUNTIME_STATES.setdefault(runtime_dir, {})
        data.update(updates)
        _write_runtime_locked(runtime_dir, data)
        return dict(data)


def _finish_offline_preparation(config):
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        start = _RUNTIME_OFFLINE_STARTS.pop(runtime_dir, None)
        if start is None:
            return float(_RUNTIME_STATES.get(runtime_dir, {}).get("offline_preparation_sec", 0.0))
        elapsed = max(0.0, time.perf_counter() - start)
        data = _RUNTIME_STATES[runtime_dir]
        data["offline_preparation_sec"] = elapsed
        data["offline_preparation_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_runtime_locked(runtime_dir, data)
        return elapsed


class _MuSiCOfflineBoundaryHandler(logging.Handler):
    """Close the offline interval on the processor's post-persistence log record."""

    _music_runtime_offline_boundary_handler = True
    _prefix = f"[DATASET][TIME] Modality {MODALITY_NAME} total preparation time:"

    def __init__(self, runtime_dir, config):
        super().__init__(level=logging.INFO)
        self.runtime_dir = runtime_dir
        self.config = config

    def emit(self, record):
        try:
            message = record.getMessage()
            if not message.startswith(self._prefix):
                return
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)s", message[len(self._prefix):])
            reported_sec = float(match.group(1)) if match else None
            with _RUNTIME_LOCK:
                state = _RUNTIME_STATES.get(self.runtime_dir)
                active = (
                    self.runtime_dir in _RUNTIME_OFFLINE_STARTS
                    and _RUNTIME_CONFIGS.get(self.runtime_dir) is self.config
                    and state is not None
                    and state.get("cold_runtime_started")
                    and state.get("modality") == MODALITY_NAME
                )
            if not active:
                return

            offline_sec = _finish_offline_preparation(self.config)
            updates = {
                "offline_boundary_source": "processor_modality_total_log",
                "offline_preparation_sec": offline_sec,
            }
            if reported_sec is not None:
                updates["processor_reported_modality_preparation_sec"] = reported_sec
            _update_runtime(self.config, updates)
            _remove_offline_boundary_handler(self.runtime_dir, expected_handler=self)
        except Exception:
            self.handleError(record)


def _remove_offline_boundary_handler(runtime_dir, expected_handler=None):
    root_logger = logging.getLogger()
    with _RUNTIME_LOCK:
        handler = _RUNTIME_OFFLINE_HANDLERS.get(runtime_dir)
        if handler is None or (expected_handler is not None and handler is not expected_handler):
            return
        _RUNTIME_OFFLINE_HANDLERS.pop(runtime_dir, None)
    root_logger.removeHandler(handler)


def _install_offline_boundary_handler(runtime_dir, config):
    _remove_offline_boundary_handler(runtime_dir)
    handler = _MuSiCOfflineBoundaryHandler(runtime_dir, config)
    with _RUNTIME_LOCK:
        _RUNTIME_OFFLINE_HANDLERS[runtime_dir] = handler
    logging.getLogger().addHandler(handler)


def _derive_runtime_fields(data, wall_clock_total):
    user_text = float(data.get("user_text_construction_sec", 0.0))
    source_llm = float(data.get("source_profile_llm_sec", 0.0))
    target_llm = float(data.get("target_profile_llm_sec", 0.0))
    embedding = float(data.get("user_profile_embedding_sec", 0.0))
    offline = float(data.get("offline_preparation_sec", 0.0))
    training_core = float(data.get("training_core_sec", 0.0))
    validation = float(data.get("validation_sec", 0.0))
    test = float(data.get("test_sec", 0.0))
    fit_wall = float(data.get("fit_wall_sec", 0.0))
    evaluation = validation + test
    model_training = fit_wall - evaluation
    end_to_end_other = wall_clock_total - offline - fit_wall
    total_excluding_evaluation = wall_clock_total - evaluation

    data.update(
        {
            "offline_other_sec": max(
                0.0,
                offline - user_text - source_llm - target_llm - embedding,
            ),
            "evaluation_sec": evaluation,
            "training_stage_other_sec": model_training - training_core,
            "model_training_sec": model_training,
            "end_to_end_other_sec": end_to_end_other,
            "wall_clock_total_sec": wall_clock_total,
            "total_sec": wall_clock_total,
            "total_excluding_evaluation_sec": total_excluding_evaluation,
            "experimental_closure_residual_sec": (
                wall_clock_total - offline - fit_wall - end_to_end_other
            ),
        }
    )


def _finalize_runtime_dir(runtime_dir):
    _remove_offline_boundary_handler(runtime_dir)
    with _RUNTIME_LOCK:
        if runtime_dir in _RUNTIME_FINALIZED:
            return
        data = _RUNTIME_STATES.get(runtime_dir)
        if not data:
            return

        offline_start = _RUNTIME_OFFLINE_STARTS.pop(runtime_dir, None)
        if offline_start is not None and not data.get("offline_preparation_sec"):
            data["offline_preparation_sec"] = max(0.0, time.perf_counter() - offline_start)
            data["offline_preparation_incomplete"] = True

        wall_start = _RUNTIME_WALL_STARTS.get(runtime_dir)
        if wall_start is None:
            wall_clock_total = sum(
                float(data.get(key, 0.0))
                for key in (
                    "offline_preparation_sec",
                    "fit_wall_sec",
                )
            )
        else:
            wall_clock_total = max(0.0, time.perf_counter() - wall_start)

        _derive_runtime_fields(data, wall_clock_total)
        fit_completed = bool(data.get("fit_completed", False))
        data.update(
            {
                "wall_clock_end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runtime_complete": fit_completed,
                "runtime_status": "complete" if fit_completed else "incomplete",
            }
        )
        _write_runtime_locked(runtime_dir, data)
        _write_runtime_csv_locked(runtime_dir, data)
        _RUNTIME_FINALIZED.add(runtime_dir)

        logger.info("[RUNTIME] MuSiC runtime summary saved to %s", _runtime_json_path_from_dir(runtime_dir))
        logger.info(
            "[RUNTIME] User text %.2fs | Source LLM %.2fs | Target LLM %.2fs | "
            "User embedding %.2fs | Offline %.2fs | Training core %.2fs | "
            "Validation %.2fs | Test %.2fs | Fit wall %.2fs | E2E other %.2fs | "
            "Wall total %.2fs",
            data.get("user_text_construction_sec", 0.0),
            data.get("source_profile_llm_sec", 0.0),
            data.get("target_profile_llm_sec", 0.0),
            data.get("user_profile_embedding_sec", 0.0),
            data.get("offline_preparation_sec", 0.0),
            data.get("training_core_sec", 0.0),
            data.get("validation_sec", 0.0),
            data.get("test_sec", 0.0),
            data.get("fit_wall_sec", 0.0),
            data.get("end_to_end_other_sec", 0.0),
            data.get("wall_clock_total_sec", 0.0),
        )


def _finalize_all_runtimes():
    with _RUNTIME_LOCK:
        runtime_dirs = list(_RUNTIME_STATES)
    for runtime_dir in runtime_dirs:
        try:
            _remove_offline_boundary_handler(runtime_dir)
            _finalize_runtime_dir(runtime_dir)
        except Exception:
            logger.exception("[RUNTIME] Failed to finalize MuSiC runtime summary for %s", runtime_dir)


atexit.register(_finalize_all_runtimes)


def _synchronize_runtime_cuda(device):
    if not _runtime_cuda_active(device):
        return False
    device = torch.device(device)
    torch.cuda.synchronize(device)
    return True


def _runtime_cuda_active(device):
    if not torch.cuda.is_available() or device is None:
        return False
    return torch.device(device).type == "cuda"


class MuSiC_End2End_RuntimeTrainer(_BASE_TRAINER_CLASS):
    """Process-local timing wrapper used only by MuSiC_End2End_Runtime."""

    _music_end2end_runtime_trainer = True

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, writer=None):
        if not getattr(self, "_music_runtime_enabled", False):
            return super()._train_epoch(train_data, epoch_idx, loss_func=loss_func, writer=writer)
        _synchronize_runtime_cuda(self.device)
        start = time.perf_counter()
        try:
            return super()._train_epoch(
                train_data,
                epoch_idx,
                loss_func=loss_func,
                writer=writer,
            )
        finally:
            _synchronize_runtime_cuda(self.device)
            self._music_runtime_training_core += time.perf_counter() - start

    def _valid_epoch(self, valid_data):
        if not getattr(self, "_music_runtime_enabled", False):
            return super()._valid_epoch(valid_data)
        _synchronize_runtime_cuda(self.device)
        start = time.perf_counter()
        try:
            return super()._valid_epoch(valid_data)
        finally:
            _synchronize_runtime_cuda(self.device)
            elapsed = time.perf_counter() - start
            if valid_data is self._music_runtime_validation_data:
                self._music_runtime_validation += elapsed
            elif valid_data is self._music_runtime_test_data:
                self._music_runtime_test += elapsed
            else:
                self._music_runtime_unclassified_evaluation += elapsed

    def fit(
        self,
        stage_id,
        train_data,
        valid_data=None,
        test_data=None,
        saved=False,
        verbose=True,
        writer=None,
    ):
        self._music_runtime_enabled = self.model.__class__.__name__ == "MuSiC_End2End_Runtime"
        if not self._music_runtime_enabled:
            return super().fit(
                stage_id,
                train_data,
                valid_data=valid_data,
                test_data=test_data,
                saved=saved,
                verbose=verbose,
                writer=writer,
            )

        self._music_runtime_validation_data = valid_data
        self._music_runtime_test_data = test_data
        self._music_runtime_training_core = 0.0
        self._music_runtime_validation = 0.0
        self._music_runtime_test = 0.0
        self._music_runtime_unclassified_evaluation = 0.0

        _synchronize_runtime_cuda(self.device)
        fit_start = time.perf_counter()
        fit_completed = False
        try:
            result = super().fit(
                stage_id,
                train_data,
                valid_data=valid_data,
                test_data=test_data,
                saved=saved,
                verbose=verbose,
                writer=writer,
            )
            fit_completed = True
        finally:
            _synchronize_runtime_cuda(self.device)
            fit_wall = time.perf_counter() - fit_start
            _update_runtime(
                self.model.config,
                {
                    "training_core_sec": self._music_runtime_training_core,
                    "validation_sec": self._music_runtime_validation,
                    "test_sec": self._music_runtime_test,
                    "fit_wall_sec": fit_wall,
                    "unclassified_evaluation_sec": (
                        self._music_runtime_unclassified_evaluation
                    ),
                    "runtime_trainer": self.__class__.__name__,
                    "runtime_trainer_gpu_synchronized": _runtime_cuda_active(self.device),
                    "fit_completed": fit_completed,
                },
            )
        return result


def _runtime_music_llm_worker(args):
    raw_uid, text, request = args
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=request["api_key"],
            base_url=request["base_url"] if request["base_url"] else None,
            timeout=request["request_timeout"],
            max_retries=request["openai_max_retries"],
        )
        response = client.chat.completions.create(
            model=request["model"],
            messages=[
                {"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        return raw_uid, response.choices[0].message.content.strip()
    except Exception as error:
        return raw_uid, f"ERROR: {type(error).__name__}: {error}"


def _run_runtime_music_llm_parallel(user_text_dict, config, modality, num_workers):
    request = {
        "api_key": config["openai_api_key"],
        "base_url": modality.get("llm_base_url") or config.get("openai_base_url"),
        "request_timeout": float(modality.get("request_timeout", 60)),
        "openai_max_retries": int(modality.get("openai_max_retries", 0)),
        "model": config.get("music_llm_model", "gpt-4o-mini"),
        "system_prompt": _get_music_user_prompt(config),
    }
    tasks = [(raw_uid, text, request) for raw_uid, text in user_text_dict.items()]
    if not tasks:
        return {}

    results = {}
    with Pool(processes=min(num_workers, len(tasks))) as pool:
        for raw_uid, summary in pool.imap_unordered(_runtime_music_llm_worker, tasks):
            results[raw_uid] = summary
    return results


def _runtime_music_profile_once(user_text_dict, config, modality, num_workers):
    results = _run_runtime_music_llm_parallel(user_text_dict, config, modality, num_workers)
    ok, errors = {}, {}
    for raw_uid in user_text_dict:
        summary = results.get(raw_uid)
        if isinstance(summary, str) and summary.strip() and not summary.strip().startswith("ERROR:"):
            ok[raw_uid] = summary.strip()
        else:
            errors[raw_uid] = summary
    return ok, errors


def _truncate_runtime_profile_text(text, max_chars):
    text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "\n[...truncated for API repair...]\n"
    available = max(1, max_chars - len(marker))
    head_chars = available // 2
    tail_chars = available - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


def _make_runtime_profile_repair_runner(
    config,
    modality,
    num_workers,
    label,
    max_rounds,
    char_limits,
    fallback_text,
    fallback_max_users,
):
    state = {
        "rounds": 0,
        "failed_users": {},
        "fallback_user_ids": [],
    }

    def run_once(pending):
        state["rounds"] += 1
        round_idx = state["rounds"]
        limit = char_limits[min(round_idx - 1, len(char_limits) - 1)]
        repaired_prompts = {
            raw_uid: _truncate_runtime_profile_text(text, limit)
            for raw_uid, text in pending.items()
        }
        ok, errors = _runtime_music_profile_once(
            repaired_prompts,
            config,
            modality,
            num_workers,
        )

        for raw_uid, error in errors.items():
            error_text = str(error)
            state["failed_users"][str(raw_uid)] = error_text
            logger.warning(
                "[DATASET][REPAIR] %s user=%s failed in round %d "
                "(input_chars=%d): %s",
                label,
                raw_uid,
                round_idx,
                len(repaired_prompts[raw_uid]),
                error_text[:1000],
            )

        if errors and round_idx >= max_rounds and len(errors) <= fallback_max_users:
            fallback_ids = list(errors)
            logger.warning(
                "[DATASET][REPAIR] %s reached %d rounds; using a safe fallback "
                "profile for %d user(s): %s",
                label,
                max_rounds,
                len(fallback_ids),
                fallback_ids,
            )
            for raw_uid in fallback_ids:
                ok[raw_uid] = fallback_text
            state["fallback_user_ids"].extend(fallback_ids)
            errors = {}
        elif errors and round_idx >= max_rounds:
            logger.error(
                "[DATASET][REPAIR] %s still has %d failed users after %d rounds, "
                "which exceeds profile_fallback_max_users=%d; aborting instead "
                "of masking a systemic API failure.",
                label,
                len(errors),
                max_rounds,
                fallback_max_users,
            )

        return ok, errors

    return run_once, state


def _embed_runtime_user_profiles(config, modality, texts):
    try:
        return _embed_text_batches(config, modality, texts)
    except RuntimeError as error:
        fallback_workers = int(modality.get("embedding_fallback_num_workers", 1))
        fallback_max_retry = int(modality.get("embedding_fallback_max_retry", 12))
        if fallback_workers < 1:
            raise ValueError("embedding_fallback_num_workers must be at least 1.")
        if fallback_max_retry < 1:
            raise ValueError("embedding_fallback_max_retry must be at least 1.")

        logger.warning(
            "[DATASET][REPAIR] Concurrent user-profile embedding failed: %s. "
            "Retrying all embedding batches with workers=%d and max_retry=%d.",
            error,
            fallback_workers,
            fallback_max_retry,
        )
        _update_runtime(
            config,
            {
                "embedding_fallback_used": True,
                "embedding_initial_error": str(error),
                "embedding_fallback_num_workers": fallback_workers,
                "embedding_fallback_max_retry": fallback_max_retry,
            },
        )
        fallback_modality = dict(modality)
        fallback_modality["embedding_num_workers"] = fallback_workers
        fallback_modality["embedding_max_retry"] = fallback_max_retry
        return _embed_text_batches(config, fallback_modality, texts)


class MuSiC_End2End_Runtime(MuSiC):
    """Fresh users-only MuSiC end-to-end profiler for Douban Book -> Music."""

    def __init__(self, config, dataloader):
        runtime_dir = _ensure_runtime(
            config,
            {
                "model": config.get("model"),
                "dataset": config.get("dataset"),
                "domains": config.get("domains"),
                "modality": MODALITY_NAME,
                "llm_chat_model": config.get("music_llm_model", "gpt-4o-mini"),
            },
        )
        self._runtime_dir = runtime_dir
        with _RUNTIME_LOCK:
            needs_offline_fallback = runtime_dir in _RUNTIME_OFFLINE_STARTS
        if needs_offline_fallback:
            offline_sec = _finish_offline_preparation(config)
            _remove_offline_boundary_handler(runtime_dir)
            _update_runtime(
                config,
                {
                    "offline_boundary_source": "runtime_model_init_fallback",
                    "offline_preparation_sec": offline_sec,
                },
            )
        _require_embedding_generated_this_run(config, "runtime model initialization")
        stage = config["training_stages"][0]
        _update_runtime(
            config,
            {
                "seed": int(config["seed"]),
                "device": str(config["device"]),
                "epochs": int(stage["epochs"]),
                "train_batch_size": int(stage["train_batch_size"]),
                "eval_batch_size": int(config["eval_batch_size"]),
                "eval_step": int(config["eval_step"]),
                "stopping_step": int(config["stopping_step"]),
                "learning_rate": float(stage["learning_rate"]),
                "diff_weight": float(config["diff_weight"]),
                "lambda_user_emb": float(config["lambda_user_emb"]),
                "beta_sche": str(config["beta_sche"]),
                "aggregator": str(config["aggregator"]),
                "history_len": int(config["history_len"]),
                "mask_rate": float(config["mask_rate"]),
            },
        )
        super().__init__(
            config,
            _ModalityAliasDataloader(dataloader, MODALITY_NAME, "CrossDomain_semantics_music"),
        )

    def write_runtime_summary(self):
        _finalize_runtime_dir(self._runtime_dir)


def extract_CrossDomain_semantics_music_end2end_runtime_users_only_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    if config["dataset"] != "Douban":
        raise ValueError("MuSiC_End2End_Runtime is intended for the Douban runtime comparison.")
    if list(config["domains"]) != ["Book", "Music"]:
        raise ValueError("MuSiC_End2End_Runtime requires domains=['Book', 'Music'].")
    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for MuSiC user-profile generation.")

    requested_workers = int(modality.get("profile_num_workers", 20))
    profile_num_workers = max(1, min(requested_workers, 20, cpu_count()))
    request_timeout = float(modality.get("request_timeout", 60))
    openai_max_retries = int(modality.get("openai_max_retries", 0))
    repair_max_rounds = int(modality.get("repair_max_rounds", 0))
    repair_sleep_seconds = float(modality.get("repair_sleep_seconds", 0.0))
    repair_char_limits = modality.get("profile_repair_char_limits", [0, 12000, 4000])
    if not isinstance(repair_char_limits, (list, tuple)) or not repair_char_limits:
        raise ValueError("profile_repair_char_limits must be a non-empty list.")
    repair_char_limits = [int(limit) for limit in repair_char_limits]
    if any(limit < 0 for limit in repair_char_limits):
        raise ValueError("profile_repair_char_limits cannot contain negative values.")
    if repair_max_rounds <= 0:
        repair_max_rounds = len(repair_char_limits)
    if len(repair_char_limits) < repair_max_rounds:
        repair_char_limits.extend(
            [repair_char_limits[-1]] * (repair_max_rounds - len(repair_char_limits))
        )
    fallback_text = str(
        modality.get(
            "profile_failure_fallback_text",
            "User preference information is unavailable.",
        )
    ).strip()
    if not fallback_text:
        raise ValueError("profile_failure_fallback_text cannot be empty.")
    fallback_max_users = int(modality.get("profile_fallback_max_users", 10))
    if fallback_max_users < 0:
        raise ValueError("profile_fallback_max_users cannot be negative.")

    _reset_runtime(
        config,
        {
            "model": config.get("model"),
            "dataset": config.get("dataset"),
            "domains": config.get("domains"),
            "modality": modality["name"],
            "semantic_scope": "all_source_and_target_users_only",
            "item_semantic_embeddings_generated": False,
            "llm_chat_model": config.get("music_llm_model", "gpt-4o-mini"),
            "embedding_model": modality["emb_model"],
            "profile_num_workers_requested": requested_workers,
            "profile_num_workers": profile_num_workers,
            "profile_num_workers_effective": profile_num_workers,
            "request_timeout": request_timeout,
            "openai_max_retries": openai_max_retries,
            "profile_repair_max_rounds": repair_max_rounds,
            "profile_repair_sleep_seconds": repair_sleep_seconds,
            "profile_repair_char_limits": repair_char_limits,
            "profile_failure_fallback_text": fallback_text,
            "profile_fallback_max_users": fallback_max_users,
            "semantic_cache_reused": False,
        },
    )

    reviews = raw_data_list[0]
    text_start = time.perf_counter()
    user_text_src = {}
    for uid in range(1, len(id_mapping["src"]["id2user"])):
        raw_uid = _id2raw(id_mapping["src"]["id2user"], uid)
        user_text_src[raw_uid] = (
            _collect_music_user_text(uid, interaction["src"], id_mapping, "src", reviews) or "None"
        )

    user_text_tgt = {}
    for uid in range(1, len(id_mapping["tgt"]["id2user"])):
        raw_uid = _id2raw(id_mapping["tgt"]["id2user"], uid)
        user_text_tgt[raw_uid] = (
            _collect_music_user_text(uid, interaction["tgt"], id_mapping, "tgt", reviews) or "None"
        )
    user_text_sec = time.perf_counter() - text_start
    _update_runtime(
        config,
        {
            "user_text_construction_sec": user_text_sec,
            "num_source_user_profiles": len(user_text_src),
            "num_target_user_profiles": len(user_text_tgt),
        },
    )

    logger.info(
        "[RUNTIME] MuSiC users-only semantic generation: %d source users + %d target users; "
        "profile workers=%d, timeout=%.1fs, OpenAI max_retries=%d.",
        len(user_text_src),
        len(user_text_tgt),
        profile_num_workers,
        request_timeout,
        openai_max_retries,
    )

    source_start = time.perf_counter()
    source_label = f"{modality['name']} source user profiles"
    source_run_once, source_repair_state = _make_runtime_profile_repair_runner(
        config=config,
        modality=modality,
        num_workers=profile_num_workers,
        label=source_label,
        max_rounds=repair_max_rounds,
        char_limits=repair_char_limits,
        fallback_text=fallback_text,
        fallback_max_users=fallback_max_users,
    )
    source_profiles, source_errors = run_until_no_prompt_errors(
        prompts=user_text_src,
        run_once_fn=source_run_once,
        label=source_label,
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )
    source_sec = time.perf_counter() - source_start
    _update_runtime(
        config,
        {
            "source_profile_llm_sec": source_sec,
            "num_failed_source_profiles": len(source_errors),
            "source_profile_repair_rounds": source_repair_state["rounds"],
            "source_profile_repair_errors": source_repair_state["failed_users"],
            "source_profile_fallback_user_ids": source_repair_state["fallback_user_ids"],
            "num_fallback_source_profiles": len(
                source_repair_state["fallback_user_ids"]
            ),
        },
    )

    target_start = time.perf_counter()
    target_label = f"{modality['name']} target user profiles"
    target_run_once, target_repair_state = _make_runtime_profile_repair_runner(
        config=config,
        modality=modality,
        num_workers=profile_num_workers,
        label=target_label,
        max_rounds=repair_max_rounds,
        char_limits=repair_char_limits,
        fallback_text=fallback_text,
        fallback_max_users=fallback_max_users,
    )
    target_profiles, target_errors = run_until_no_prompt_errors(
        prompts=user_text_tgt,
        run_once_fn=target_run_once,
        label=target_label,
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )
    target_sec = time.perf_counter() - target_start
    _update_runtime(
        config,
        {
            "target_profile_llm_sec": target_sec,
            "num_failed_target_profiles": len(target_errors),
            "target_profile_repair_rounds": target_repair_state["rounds"],
            "target_profile_repair_errors": target_repair_state["failed_users"],
            "target_profile_fallback_user_ids": target_repair_state["fallback_user_ids"],
            "num_fallback_target_profiles": len(
                target_repair_state["fallback_user_ids"]
            ),
        },
    )
    logger.info(
        "[RUNTIME] MuSiC user-profile LLM finished in %.2fs (source %.2fs, target %.2fs).",
        source_sec + target_sec,
        source_sec,
        target_sec,
    )

    return {
        "src_user": source_profiles,
        "tgt_user": target_profiles,
        "error_src_user": source_errors,
        "error_tgt_user": target_errors,
    }


def generate_CrossDomain_semantics_music_end2end_runtime_users_only_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    _require_active_cold_runtime(config, "user-profile embedding")
    src_users = id_mapping["src"]["id2user"][1:]
    tgt_users = id_mapping["tgt"]["id2user"][1:]
    src_texts = [modality_data["src_user"][_id2raw(src_users, idx)] for idx in range(len(src_users))]
    tgt_texts = [modality_data["tgt_user"][_id2raw(tgt_users, idx)] for idx in range(len(tgt_users))]

    embedding_start = time.perf_counter()
    # One combined request queue lets the shared 10-thread limit apply to all
    # user-profile batches instead of creating two smaller sequential queues.
    result = _embed_runtime_user_profiles(config, modality, src_texts + tgt_texts)
    embedding_sec = time.perf_counter() - embedding_start

    batch_size = int(modality["emb_batch_size"])
    _update_runtime(
        config,
        {
            "user_profile_embedding_sec": embedding_sec,
            "embedding_num_texts": len(src_texts) + len(tgt_texts),
            "embedding_num_source_texts": len(src_texts),
            "embedding_num_target_texts": len(tgt_texts),
            "embedding_batches": (len(src_texts) + len(tgt_texts) + batch_size - 1) // batch_size,
            "embedding_batch_size": batch_size,
            "embedding_num_workers": int(modality.get("embedding_num_workers", 1)),
            "embedding_dim": int(result.shape[1]),
            "embedding_generated_this_run": True,
            "item_semantic_embeddings_generated": False,
        },
    )
    logger.info(
        "[RUNTIME] MuSiC embedded %d user profiles in %.2fs; no item semantic embeddings were generated.",
        len(src_texts) + len(tgt_texts),
        embedding_sec,
    )
    return result


def generate_CrossDomain_semantics_music_end2end_runtime_users_only_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    _require_active_cold_runtime(config, "final-embedding generation")
    _require_embedding_generated_this_run(config, "final-embedding generation")
    return generate_CrossDomain_semantics_music_final_embs(
        config,
        modality,
        interaction,
        id_mapping,
        modality_embs,
    )


def _inject_runtime_trainer():
    # quick_start imported get_trainer into its own module namespace before the
    # model module was loaded. Replace that process-local reference instead of
    # rebinding common.trainer.Trainer: the base Trainer.__init__ uses the
    # module-level Trainer name in super(), so rebinding that class recurses.
    import utils.quick_start as quick_start_module

    quick_start_module.get_trainer = lambda: MuSiC_End2End_RuntimeTrainer


_inject_runtime_trainer()
