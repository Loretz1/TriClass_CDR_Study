import atexit
import csv
import json
import logging
import os
import threading
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import common.trainer as _trainer_module

from models.llm4cdsr import (
    LLM4CDSR,
    _build_overall_prompt_from_summaries,
    _build_subseq_prompt_from_titles,
    _build_user_mixed_sequences,
    _compute_counts_from_id_mapping,
    _get_title,
    _raw_item_to_global_index,
    generate_CrossDomain_semantics_LLM4CDSR_final_embs,
)
from models.llm4cdsr_end2end import (
    _ModalityAliasDataloader,
    _embed_text_batches,
    _item_emb_cache_path,
    _ordered_item_texts,
    _raw_metadata,
    run_until_no_prompt_errors,
)


logger = logging.getLogger()

MODALITY_NAME = "CrossDomain_semantics_LLM4CDSR_end2end_runtime"
DEFAULT_RUNTIME_DIR = "runtime_records/llm4cdsr_book_music_runtime"

_RUNTIME_LOCK = threading.RLock()
_RUNTIME_STATES = {}
_RUNTIME_WALL_STARTS = {}
_RUNTIME_OFFLINE_STARTS = {}
_RUNTIME_FINALIZED = set()

_IMPORTED_TRAINER = _trainer_module.Trainer

_TIMING_KEYS = (
    "item_text_construction_sec",
    "item_embedding_sec",
    "kmeans_prompt_construction_sec",
    "subsequence_profile_llm_sec",
    "overall_profile_llm_sec",
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


def _runtime_json_path(runtime_dir):
    return os.path.join(runtime_dir, "runtime_summary.json")


def _runtime_csv_path(runtime_dir):
    return os.path.join(runtime_dir, "runtime_summary.csv")


def _write_runtime_locked(runtime_dir, data):
    path = _runtime_json_path(runtime_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _write_runtime_csv_locked(runtime_dir, data):
    rows = (
        ("item_text_construction_sec", "Item text construction"),
        ("item_embedding_sec", "Item text embedding"),
        ("kmeans_prompt_construction_sec", "KMeans and prompt construction"),
        ("subsequence_profile_llm_sec", "Subsequence-profile LLM generation"),
        ("overall_profile_llm_sec", "Overall-profile LLM generation"),
        ("user_profile_embedding_sec", "User-profile text embedding"),
        ("offline_other_sec", "Offline preparation (other)"),
        ("offline_preparation_sec", "Offline preparation (total)"),
        ("training_core_sec", "Training core"),
        ("training_other_sec", "Training (other, excluding evaluation)"),
        ("model_training_sec", "Model training (excluding evaluation)"),
        ("validation_sec", "Validation"),
        ("test_sec", "Test"),
        ("evaluation_sec", "Evaluation (validation + test)"),
        ("fit_wall_sec", "Trainer fit wall clock (including evaluation)"),
        ("end_to_end_other_sec", "End-to-end overhead outside offline preparation and fit"),
        ("total_excluding_evaluation_sec", "Total excluding evaluation"),
        ("wall_clock_total_sec", "Wall-clock total (including evaluation)"),
    )
    path = _runtime_csv_path(runtime_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["component", "seconds", "minutes"])
        for key, label in rows:
            seconds = float(data.get(key, 0.0))
            writer.writerow([label, f"{seconds:.6f}", f"{seconds / 60.0:.6f}"])
    os.replace(tmp_path, path)


def _reset_runtime(config, metadata):
    runtime_dir = _runtime_dir(config)
    now_perf = time.perf_counter()
    data = dict(metadata)
    for key in _TIMING_KEYS:
        data[key] = 0.0
    data.update(
        {
            "offline_other_sec": 0.0,
            "training_other_sec": 0.0,
            "training_stage_other_sec": 0.0,
            "evaluation_sec": 0.0,
            "model_training_sec": 0.0,
            "end_to_end_other_sec": 0.0,
            "total_excluding_evaluation_sec": 0.0,
            "embedding_generated_this_run": False,
            "fit_completed": False,
            "wall_clock_start_perf": now_perf,
            "wall_clock_start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cold_runtime_started": True,
            "runtime_complete": False,
            "timing_protocol": {
                "experimental_partition": (
                    "wall_clock_total_sec = offline_preparation_sec + fit_wall_sec + "
                    "end_to_end_other_sec"
                ),
                "wall_clock_start": "runtime semantic-extraction handler entry",
                "wall_clock_end": "process atexit, matching the LSDCDR wall-clock-total boundary",
                "offline_preparation_end": (
                    "processor modality-total log emitted after final semantic-embedding .npy persisted"
                ),
                "training_core": "sum of target-model Trainer._train_epoch wall times",
                "evaluation": "sum of target-model Trainer._valid_epoch wall times, split by loader identity",
                "fit_wall": "target-model Trainer.fit wall time including training and evaluation",
                "model_training": "fit wall time minus validation and test",
                "training_other": "model training minus summed Trainer._train_epoch time",
                "end_to_end_other": "wall-clock total minus offline preparation and fit wall time",
                "total_excluding_evaluation": "wall-clock total minus validation and test",
                "paper_aggregation": (
                    "Offline/model-training/evaluation values are derived after the experiment from the "
                    "closed wall-clock partition; they are not independent wall-clock intervals."
                ),
            },
        }
    )
    with _RUNTIME_LOCK:
        _RUNTIME_STATES[runtime_dir] = data
        _RUNTIME_WALL_STARTS[runtime_dir] = now_perf
        _RUNTIME_OFFLINE_STARTS[runtime_dir] = now_perf
        _RUNTIME_FINALIZED.discard(runtime_dir)
        _write_runtime_locked(runtime_dir, data)
    _install_offline_log_handler(runtime_dir)
    return runtime_dir


def _ensure_runtime(config, metadata):
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        state = _RUNTIME_STATES.get(runtime_dir)
        if state:
            if config.get("require_fresh_runtime_cache", True) and not state.get("cold_runtime_started"):
                raise RuntimeError("The LLM4CDSR runtime pipeline did not start from a cold cache.")
            return runtime_dir
    metadata = dict(metadata)
    metadata.update(
        {
            "semantic_cache_reused": True,
            "note": (
                "The runtime modality cache was reused, so offline timings are zero. "
                "Remove all cache files beginning with the runtime modality name before "
                "a fresh end-to-end timing run."
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
    _remove_offline_log_handlers()
    return runtime_dir


def _require_active_cold_runtime(config, stage):
    if not config.get("require_fresh_runtime_cache", True):
        return
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        state = _RUNTIME_STATES.get(runtime_dir)
        if not state or not state.get("cold_runtime_started"):
            raise RuntimeError(
                f"Cannot time LLM4CDSR {stage}: an earlier runtime modality cache was reused. "
                f"Delete every cache file beginning with '{MODALITY_NAME}' and rerun."
            )


def _update_runtime(config, updates):
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        data = _RUNTIME_STATES.setdefault(runtime_dir, {})
        data.update(updates)
        _write_runtime_locked(runtime_dir, data)
        return dict(data)


def _add_runtime(config, key, elapsed):
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        data = _RUNTIME_STATES.setdefault(runtime_dir, {})
        data[key] = float(data.get(key, 0.0)) + float(elapsed)
        _write_runtime_locked(runtime_dir, data)
        return data[key]


def _finish_offline_preparation_dir(runtime_dir, source, fallback=False):
    with _RUNTIME_LOCK:
        start = _RUNTIME_OFFLINE_STARTS.pop(runtime_dir, None)
        if start is None:
            return float(_RUNTIME_STATES.get(runtime_dir, {}).get("offline_preparation_sec", 0.0))
        elapsed = max(0.0, time.perf_counter() - start)
        data = _RUNTIME_STATES[runtime_dir]
        data["offline_preparation_sec"] = elapsed
        data["offline_preparation_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["offline_completion_source"] = source
        data["offline_completion_fallback"] = bool(fallback)
        final_embs_path = data.get("final_embs_path")
        if final_embs_path and os.path.exists(final_embs_path):
            data["final_embs_file_bytes"] = os.path.getsize(final_embs_path)
        _write_runtime_locked(runtime_dir, data)
        return elapsed


def _finish_offline_preparation(config, source, fallback=False):
    return _finish_offline_preparation_dir(
        _runtime_dir(config),
        source=source,
        fallback=fallback,
    )


class _OfflinePreparationLogHandler(logging.Handler):
    """Ends offline timing on the processor's post-save modality-total log."""

    _llm4cdsr_runtime_handler = True
    _prefix = f"[DATASET][TIME] Modality {MODALITY_NAME} total preparation time:"

    def __init__(self, runtime_dir, level=logging.NOTSET):
        super().__init__(level=level)
        self.runtime_dir = runtime_dir

    def emit(self, record):
        try:
            if not record.getMessage().startswith(self._prefix):
                return
            with _RUNTIME_LOCK:
                data = _RUNTIME_STATES.get(self.runtime_dir)
                active = (
                    data is not None
                    and self.runtime_dir in _RUNTIME_OFFLINE_STARTS
                    and data.get("modality") == MODALITY_NAME
                    and data.get("cold_runtime_started") is True
                    and self.runtime_dir not in _RUNTIME_FINALIZED
                )
            if active:
                _finish_offline_preparation_dir(
                    self.runtime_dir,
                    source="processor_modality_total_log",
                    fallback=False,
                )
                logger.removeHandler(self)
                self.close()
        except Exception:
            self.handleError(record)


def _remove_offline_log_handlers():
    for handler in list(logger.handlers):
        if getattr(handler, "_llm4cdsr_runtime_handler", False):
            logger.removeHandler(handler)
            handler.close()


def _install_offline_log_handler(runtime_dir):
    # A reload should replace, rather than retain, a handler bound to stale module globals.
    _remove_offline_log_handlers()
    logger.addHandler(_OfflinePreparationLogHandler(runtime_dir, level=logging.INFO))


def _derive_runtime_fields(data, wall_clock_total):
    offline_detail = sum(
        float(data.get(key, 0.0))
        for key in (
            "item_text_construction_sec",
            "item_embedding_sec",
            "kmeans_prompt_construction_sec",
            "subsequence_profile_llm_sec",
            "overall_profile_llm_sec",
            "user_profile_embedding_sec",
        )
    )
    offline = float(data.get("offline_preparation_sec", 0.0))
    training_core = float(data.get("training_core_sec", 0.0))
    validation = float(data.get("validation_sec", 0.0))
    test = float(data.get("test_sec", 0.0))
    fit_wall = float(data.get("fit_wall_sec", 0.0))
    evaluation = validation + test
    model_training = max(0.0, fit_wall - evaluation)
    training_other = max(0.0, model_training - training_core)
    # Keep the experimental partition algebraically closed. A negative value is
    # intentionally preserved because it signals overlapping or inconsistent timers.
    end_to_end_other = wall_clock_total - offline - fit_wall
    total_excluding_evaluation = max(0.0, wall_clock_total - evaluation)

    data.update(
        {
            "offline_other_sec": max(0.0, offline - offline_detail),
            "evaluation_sec": evaluation,
            "training_other_sec": training_other,
            # Backward-compatible alias retained for existing runtime-summary consumers.
            "training_stage_other_sec": training_other,
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
                "runtime_finalized": True,
                "runtime_complete": fit_completed,
                "fit_failed_or_incomplete": not fit_completed,
            }
        )
        _write_runtime_locked(runtime_dir, data)
        _write_runtime_csv_locked(runtime_dir, data)
        _RUNTIME_FINALIZED.add(runtime_dir)

        logger.info("[RUNTIME] LLM4CDSR runtime summary saved to %s", _runtime_json_path(runtime_dir))
        logger.info(
            "[RUNTIME] Item text %.2fs | Item embedding %.2fs | KMeans/prompts %.2fs | "
            "Sub-profile LLM %.2fs | Overall-profile LLM %.2fs | User embedding %.2fs | "
            "Offline %.2fs | Training core %.2fs | Validation %.2fs | Test %.2fs | Total %.2fs",
            data.get("item_text_construction_sec", 0.0),
            data.get("item_embedding_sec", 0.0),
            data.get("kmeans_prompt_construction_sec", 0.0),
            data.get("subsequence_profile_llm_sec", 0.0),
            data.get("overall_profile_llm_sec", 0.0),
            data.get("user_profile_embedding_sec", 0.0),
            data.get("offline_preparation_sec", 0.0),
            data.get("training_core_sec", 0.0),
            data.get("validation_sec", 0.0),
            data.get("test_sec", 0.0),
            data.get("wall_clock_total_sec", 0.0),
        )


def _finalize_all_runtimes():
    with _RUNTIME_LOCK:
        runtime_dirs = list(_RUNTIME_STATES)
    for runtime_dir in runtime_dirs:
        try:
            _finalize_runtime_dir(runtime_dir)
        except Exception:
            logger.exception("[RUNTIME] Failed to finalize LLM4CDSR runtime summary for %s", runtime_dir)


atexit.register(_finalize_all_runtimes)


def _runtime_prompt_worker(args):
    key, prompt, request = args
    from openai import OpenAI

    client = OpenAI(
        api_key=request["api_key"],
        base_url=request["base_url"] if request["base_url"] else None,
        timeout=request["request_timeout"],
        max_retries=request["openai_max_retries"],
    )
    last_error = None
    for _ in range(max(1, request["max_retry"])):
        try:
            response = client.chat.completions.create(
                model=request["model"],
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return key, text, True
        except Exception as error:
            last_error = error
    return key, f"{type(last_error).__name__}: {last_error}", False


def _run_runtime_prompts(prompts, config, modality, num_workers):
    request = {
        "api_key": config["openai_api_key"],
        "base_url": modality.get("llm_base_url") or config.get("openai_base_url"),
        "request_timeout": float(modality.get("request_timeout", 60)),
        "openai_max_retries": int(modality.get("openai_max_retries", 0)),
        "model": modality.get("llm_chat_model", "gpt-4o-mini"),
        "max_retry": int(modality.get("max_retry", 2)),
    }
    tasks = [(key, prompt, request) for key, prompt in prompts.items()]
    if not tasks:
        return {}, {}

    ok, errors = {}, {}
    with Pool(processes=min(num_workers, len(tasks))) as pool:
        for key, output, success in pool.imap_unordered(_runtime_prompt_worker, tasks):
            if success:
                ok[key] = output
            else:
                errors[key] = output
    return ok, errors


def _truncate_runtime_prompt(prompt, max_chars):
    prompt = str(prompt)
    if max_chars <= 0 or len(prompt) <= max_chars:
        return prompt

    marker = "\n[...truncated for API repair...]\n"
    available = max(1, max_chars - len(marker))
    head_chars = available // 2
    tail_chars = available - head_chars
    return prompt[:head_chars] + marker + prompt[-tail_chars:]


def _make_runtime_prompt_repair_runner(
    config,
    modality,
    num_workers,
    label,
    max_rounds,
    char_limits,
    fallback_text,
    fallback_max_prompts,
):
    state = {
        "rounds": 0,
        "failed_prompts": {},
        "fallback_prompt_ids": [],
    }

    def run_once(pending):
        state["rounds"] += 1
        round_idx = state["rounds"]
        limit = char_limits[min(round_idx - 1, len(char_limits) - 1)]
        repaired_prompts = {
            key: _truncate_runtime_prompt(prompt, limit)
            for key, prompt in pending.items()
        }
        ok, errors = _run_runtime_prompts(
            repaired_prompts,
            config,
            modality,
            num_workers,
        )

        for key, error in errors.items():
            error_text = str(error)
            state["failed_prompts"][str(key)] = error_text
            logger.warning(
                "[DATASET][REPAIR] %s prompt=%s failed in round %d "
                "(input_chars=%d): %s",
                label,
                key,
                round_idx,
                len(repaired_prompts[key]),
                error_text[:1000],
            )

        if errors and round_idx >= max_rounds and len(errors) <= fallback_max_prompts:
            fallback_ids = list(errors)
            logger.warning(
                "[DATASET][REPAIR] %s reached %d rounds; using a safe fallback "
                "for %d prompt(s): %s",
                label,
                max_rounds,
                len(fallback_ids),
                fallback_ids,
            )
            for key in fallback_ids:
                ok[key] = fallback_text
            state["fallback_prompt_ids"].extend(fallback_ids)
            errors = {}
        elif errors and round_idx >= max_rounds:
            logger.error(
                "[DATASET][REPAIR] %s still has %d failed prompts after %d rounds, "
                "which exceeds profile_fallback_max_prompts=%d; aborting instead "
                "of masking a systemic API failure.",
                label,
                len(errors),
                max_rounds,
                fallback_max_prompts,
            )

        return ok, errors

    return run_once, state


def _embed_runtime_texts(config, modality, texts, label):
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
            "[DATASET][REPAIR] Concurrent %s embedding failed: %s. "
            "Retrying all batches with workers=%d and max_retry=%d.",
            label,
            error,
            fallback_workers,
            fallback_max_retry,
        )
        _update_runtime(
            config,
            {
                f"{label}_embedding_fallback_used": True,
                f"{label}_embedding_initial_error": str(error),
                f"{label}_embedding_fallback_num_workers": fallback_workers,
                f"{label}_embedding_fallback_max_retry": fallback_max_retry,
            },
        )
        fallback_modality = dict(modality)
        fallback_modality["embedding_num_workers"] = fallback_workers
        fallback_modality["embedding_max_retry"] = fallback_max_retry
        return _embed_text_batches(config, fallback_modality, texts)


def _sync_runtime_device(model):
    device = getattr(model, "device", None)
    if device is None or not str(device).startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _record_fit_timings(
    config,
    training_core,
    validation,
    test,
    fit_wall,
    fit_completed,
    unclassified_evaluation=0.0,
):
    """Persist one fit call atomically, avoiding per-epoch runtime-file I/O."""
    runtime_dir = _runtime_dir(config)
    with _RUNTIME_LOCK:
        data = _RUNTIME_STATES.setdefault(runtime_dir, {})
        data["training_core_sec"] = float(data.get("training_core_sec", 0.0)) + float(training_core)
        data["validation_sec"] = float(data.get("validation_sec", 0.0)) + float(validation)
        data["test_sec"] = float(data.get("test_sec", 0.0)) + float(test)
        data["fit_wall_sec"] = float(data.get("fit_wall_sec", 0.0)) + float(fit_wall)
        data["unclassified_evaluation_sec"] = (
            float(data.get("unclassified_evaluation_sec", 0.0))
            + float(unclassified_evaluation)
        )
        previous_fit_calls = int(data.get("fit_call_count", 0))
        if previous_fit_calls == 0:
            data["fit_completed"] = bool(fit_completed)
        else:
            data["fit_completed"] = bool(data.get("fit_completed", False)) and bool(
                fit_completed
            )
        data["fit_call_count"] = previous_fit_calls + 1
        _write_runtime_locked(runtime_dir, data)


class LLM4CDSREnd2EndRuntimeTrainer(_IMPORTED_TRAINER):
    """Runtime-only Trainer wrapper; non-target models retain base behavior."""

    def _runtime_target_enabled(self):
        return isinstance(self.model, LLM4CDSR_End2End_Runtime)

    def _train_epoch(self, *args, **kwargs):
        if not self._runtime_target_enabled() or not getattr(self, "_runtime_fit_active", False):
            return super()._train_epoch(*args, **kwargs)

        _sync_runtime_device(self.model)
        start = time.perf_counter()
        try:
            return super()._train_epoch(*args, **kwargs)
        finally:
            _sync_runtime_device(self.model)
            self._runtime_training_core_sec += max(0.0, time.perf_counter() - start)

    def _valid_epoch(self, eval_data):
        if not self._runtime_target_enabled() or not getattr(self, "_runtime_fit_active", False):
            return super()._valid_epoch(eval_data)

        if eval_data is self._runtime_valid_data:
            timing_attr = "_runtime_validation_sec"
        elif eval_data is self._runtime_test_data:
            timing_attr = "_runtime_test_sec"
        else:
            timing_attr = "_runtime_unclassified_evaluation_sec"

        _sync_runtime_device(self.model)
        start = time.perf_counter()
        try:
            return super()._valid_epoch(eval_data)
        finally:
            _sync_runtime_device(self.model)
            elapsed = max(0.0, time.perf_counter() - start)
            setattr(self, timing_attr, float(getattr(self, timing_attr, 0.0)) + elapsed)

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
        if not self._runtime_target_enabled():
            return super().fit(
                stage_id,
                train_data,
                valid_data=valid_data,
                test_data=test_data,
                saved=saved,
                verbose=verbose,
                writer=writer,
            )

        self._runtime_fit_active = True
        self._runtime_valid_data = valid_data
        self._runtime_test_data = test_data
        self._runtime_training_core_sec = 0.0
        self._runtime_validation_sec = 0.0
        self._runtime_test_sec = 0.0
        self._runtime_unclassified_evaluation_sec = 0.0

        _sync_runtime_device(self.model)
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
            return result
        finally:
            _sync_runtime_device(self.model)
            fit_wall_sec = max(0.0, time.perf_counter() - fit_start)
            _record_fit_timings(
                self.model.config,
                training_core=self._runtime_training_core_sec,
                validation=self._runtime_validation_sec,
                test=self._runtime_test_sec,
                fit_wall=fit_wall_sec,
                fit_completed=fit_completed,
                unclassified_evaluation=self._runtime_unclassified_evaluation_sec,
            )
            self._runtime_fit_active = False
            self._runtime_valid_data = None
            self._runtime_test_data = None


class LLM4CDSR_End2End_Runtime(LLM4CDSR):
    """Fresh LLM4CDSR end-to-end profiler for Douban Book -> Music."""

    def __init__(self, config, dataloader):
        self._runtime_dir = _ensure_runtime(
            config,
            {
                "model": config.get("model"),
                "dataset": config.get("dataset"),
                "domains": config.get("domains"),
                "modality": MODALITY_NAME,
            },
        )
        with _RUNTIME_LOCK:
            embedding_generated_this_run = bool(
                _RUNTIME_STATES[self._runtime_dir].get("embedding_generated_this_run", False)
            )
        if config.get("require_fresh_runtime_cache", True) and not embedding_generated_this_run:
            _remove_offline_log_handlers()
            raise RuntimeError(
                "Fresh LLM4CDSR metadata was prepared, but the runtime user/item modality "
                "embeddings were reused from an intermediate cache. Delete every cache file "
                f"beginning with '{MODALITY_NAME}' before the timed run."
            )
        with _RUNTIME_LOCK:
            offline_log_was_seen = self._runtime_dir not in _RUNTIME_OFFLINE_STARTS
        if not offline_log_was_seen:
            offline_sec = _finish_offline_preparation(
                config,
                source="model_init_fallback",
                fallback=True,
            )
            logger.warning(
                "[RUNTIME] Processor completion log was not observed; "
                "offline timing ended at model initialization after %.2fs.",
                offline_sec,
            )
            _remove_offline_log_handlers()
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
                "alpha": float(config["alpha"]),
                "beta": float(config["beta"]),
                "history_len": int(config["history_len"]),
                "user_llm_emb_w": float(config["user_llm_emb_w"]),
                "item_llm_emb_w": float(config["item_llm_emb_w"]),
            },
        )
        super().__init__(
            config,
            _ModalityAliasDataloader(
                dataloader,
                MODALITY_NAME,
                "CrossDomain_semantics_LLM4CDSR",
            ),
        )

    def write_runtime_summary(self):
        _finalize_runtime_dir(self._runtime_dir)


def extract_CrossDomain_semantics_LLM4CDSR_end2end_runtime_modality_data(
    config,
    modality,
    interaction,
    id_mapping,
    raw_data_list,
):
    if config["dataset"] != "Douban":
        raise ValueError("LLM4CDSR_End2End_Runtime is intended for the Douban runtime comparison.")
    if list(config["domains"]) != ["Book", "Music"]:
        raise ValueError("LLM4CDSR_End2End_Runtime requires domains=['Book', 'Music'].")
    if not config.get("openai_api_key"):
        raise ValueError("openai_api_key is required for LLM4CDSR profile generation.")

    requested_workers = int(modality.get("profile_num_workers", 20))
    profile_num_workers = max(1, min(requested_workers, 20, cpu_count()))
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
            "No reliable preference summary is available.",
        )
    ).strip()
    if not fallback_text:
        raise ValueError("profile_failure_fallback_text cannot be empty.")
    fallback_max_prompts = int(modality.get("profile_fallback_max_prompts", 10))
    if fallback_max_prompts < 0:
        raise ValueError("profile_fallback_max_prompts cannot be negative.")

    _reset_runtime(
        config,
        {
            "model": config.get("model"),
            "dataset": config.get("dataset"),
            "domains": config.get("domains"),
            "modality": modality["name"],
            "llm_chat_model": modality.get("llm_chat_model", "gpt-4o-mini"),
            "embedding_model": modality["emb_model"],
            "profile_num_workers_requested": requested_workers,
            "profile_num_workers": profile_num_workers,
            "profile_num_workers_effective": profile_num_workers,
            "request_timeout": float(modality.get("request_timeout", 60)),
            "openai_max_retries": int(modality.get("openai_max_retries", 0)),
            "profile_repair_max_rounds": repair_max_rounds,
            "profile_repair_sleep_seconds": repair_sleep_seconds,
            "profile_repair_char_limits": repair_char_limits,
            "profile_failure_fallback_text": fallback_text,
            "profile_fallback_max_prompts": fallback_max_prompts,
            "semantic_cache_reused": False,
        },
    )

    metadata = _raw_metadata(config, raw_data_list)
    counts = _compute_counts_from_id_mapping(id_mapping)
    item_emb_path = _item_emb_cache_path(config, modality)

    if os.path.exists(item_emb_path):
        if config.get("require_fresh_runtime_cache", True):
            raise RuntimeError(
                f"Cached LLM4CDSR runtime item embeddings found at {item_emb_path}. "
                f"Delete every cache file beginning with '{MODALITY_NAME}' before the timed run."
            )
        item_embs = np.load(item_emb_path).astype(np.float32)
        _update_runtime(
            config,
            {
                "item_embedding_cache_reused": True,
                "semantic_cache_reused": True,
                "item_emb_path": item_emb_path,
            },
        )
    else:
        text_start = time.perf_counter()
        _, src_texts = _ordered_item_texts(config, id_mapping, metadata, "src")
        _, tgt_texts = _ordered_item_texts(config, id_mapping, metadata, "tgt")
        item_text_sec = time.perf_counter() - text_start

        embedding_start = time.perf_counter()
        item_embs = _embed_runtime_texts(
            config,
            modality,
            src_texts + tgt_texts,
            "item",
        ).astype(np.float32)
        item_embedding_sec = time.perf_counter() - embedding_start
        np.save(item_emb_path, item_embs)
        _update_runtime(
            config,
            {
                "item_text_construction_sec": item_text_sec,
                "item_embedding_sec": item_embedding_sec,
                "item_embedding_cache_reused": False,
                "item_embedding_num_texts": len(src_texts) + len(tgt_texts),
                "item_embedding_num_source_texts": len(src_texts),
                "item_embedding_num_target_texts": len(tgt_texts),
                "item_emb_path": item_emb_path,
            },
        )

    expected_items = counts["num_src_items"] + counts["num_tgt_items"]
    if item_embs.shape[0] != expected_items:
        raise ValueError(f"Item embedding row mismatch: got {item_embs.shape[0]}, expected {expected_items}")

    from sklearn.cluster import KMeans

    construction_start = time.perf_counter()
    cluster_k = int(modality.get("cluster_k", 2))
    kmeans = KMeans(n_clusters=cluster_k, random_state=999, n_init="auto")
    cluster_ids = kmeans.fit_predict(item_embs)
    user2seq = _build_user_mixed_sequences(interaction, id_mapping)
    subsequence_prompts = {}
    for raw_user, sequence in user2seq.items():
        buckets = [[] for _ in range(cluster_k)]
        for domain, raw_item in sequence:
            index = _raw_item_to_global_index(domain, raw_item, id_mapping, counts)
            cluster_id = int(cluster_ids[index])
            buckets[cluster_id].append(_get_title(raw_item, metadata[domain], config["dataset"]))
        for cluster_id in range(cluster_k):
            subsequence_prompts[f"{raw_user}__sub{cluster_id}"] = _build_subseq_prompt_from_titles(
                buckets[cluster_id]
            )
    construction_sec = time.perf_counter() - construction_start
    _add_runtime(config, "kmeans_prompt_construction_sec", construction_sec)

    logger.info(
        "[RUNTIME] LLM4CDSR semantic generation: %d items, %d users, K=%d, profile workers=%d.",
        expected_items,
        len(user2seq),
        cluster_k,
        profile_num_workers,
    )

    subsequence_start = time.perf_counter()
    subsequence_label = f"{modality['name']} subsequence profiles"
    subsequence_run_once, subsequence_repair_state = _make_runtime_prompt_repair_runner(
        config=config,
        modality=modality,
        num_workers=profile_num_workers,
        label=subsequence_label,
        max_rounds=repair_max_rounds,
        char_limits=repair_char_limits,
        fallback_text=fallback_text,
        fallback_max_prompts=fallback_max_prompts,
    )
    subsequence_profiles, subsequence_errors = run_until_no_prompt_errors(
        prompts=subsequence_prompts,
        run_once_fn=subsequence_run_once,
        label=subsequence_label,
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )
    subsequence_sec = time.perf_counter() - subsequence_start
    _update_runtime(
        config,
        {
            "subsequence_profile_llm_sec": subsequence_sec,
            "num_subsequence_profile_prompts": len(subsequence_prompts),
            "num_failed_subsequence_profiles": len(subsequence_errors),
            "subsequence_profile_repair_rounds": subsequence_repair_state["rounds"],
            "subsequence_profile_repair_errors": subsequence_repair_state["failed_prompts"],
            "subsequence_profile_fallback_prompt_ids": subsequence_repair_state[
                "fallback_prompt_ids"
            ],
            "num_fallback_subsequence_profiles": len(
                subsequence_repair_state["fallback_prompt_ids"]
            ),
        },
    )

    overall_prompt_start = time.perf_counter()
    overall_prompts = {}
    for raw_user in user2seq:
        summaries = [
            subsequence_profiles.get(f"{raw_user}__sub{cluster_id}", "None")
            for cluster_id in range(cluster_k)
        ]
        overall_prompts[raw_user] = _build_overall_prompt_from_summaries(summaries)
    _add_runtime(
        config,
        "kmeans_prompt_construction_sec",
        time.perf_counter() - overall_prompt_start,
    )

    overall_start = time.perf_counter()
    overall_label = f"{modality['name']} overall user profiles"
    overall_run_once, overall_repair_state = _make_runtime_prompt_repair_runner(
        config=config,
        modality=modality,
        num_workers=profile_num_workers,
        label=overall_label,
        max_rounds=repair_max_rounds,
        char_limits=repair_char_limits,
        fallback_text=fallback_text,
        fallback_max_prompts=fallback_max_prompts,
    )
    overall_profiles, overall_errors = run_until_no_prompt_errors(
        prompts=overall_prompts,
        run_once_fn=overall_run_once,
        label=overall_label,
        max_rounds=repair_max_rounds,
        sleep_seconds=repair_sleep_seconds,
    )
    overall_sec = time.perf_counter() - overall_start
    _update_runtime(
        config,
        {
            "overall_profile_llm_sec": overall_sec,
            "num_overall_profile_prompts": len(overall_prompts),
            "num_failed_overall_profiles": len(overall_errors),
            "overall_profile_repair_rounds": overall_repair_state["rounds"],
            "overall_profile_repair_errors": overall_repair_state["failed_prompts"],
            "overall_profile_fallback_prompt_ids": overall_repair_state[
                "fallback_prompt_ids"
            ],
            "num_fallback_overall_profiles": len(
                overall_repair_state["fallback_prompt_ids"]
            ),
            "cluster_k": cluster_k,
        },
    )

    return {
        "counts": counts,
        "cluster_k": cluster_k,
        "item_emb_path": item_emb_path,
        "data_subseq": subsequence_profiles,
        "error_subseq": subsequence_errors,
        "data_users": overall_profiles,
        "error_users": overall_errors,
    }


def generate_CrossDomain_semantics_LLM4CDSR_end2end_runtime_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    _require_active_cold_runtime(config, "user-profile embedding")
    counts = modality_data["counts"]
    src_users = id_mapping["src"]["id2user"][1:]
    src_user_set = set(src_users)
    tgt_users = id_mapping["tgt"]["id2user"][1:]
    ordered_users = list(src_users) + [user for user in tgt_users if user not in src_user_set]
    if len(ordered_users) != counts["num_users"]:
        raise ValueError("Union user size mismatch")

    user_texts = [modality_data["data_users"].get(raw_user, "None") for raw_user in ordered_users]
    embedding_start = time.perf_counter()
    user_embeddings = _embed_runtime_texts(
        config,
        modality,
        user_texts,
        "user_profile",
    ).astype(np.float32)
    embedding_sec = time.perf_counter() - embedding_start
    _update_runtime(
        config,
        {
            "user_profile_embedding_sec": embedding_sec,
            "embedding_generated_this_run": True,
            "user_profile_embedding_num_texts": len(user_texts),
            "embedding_batch_size": int(modality["emb_batch_size"]),
            "embedding_num_workers": int(modality.get("embedding_num_workers", 1)),
            "embedding_dim": int(user_embeddings.shape[1]),
        },
    )

    item_embeddings = np.load(modality_data["item_emb_path"]).astype(np.float32)
    num_src_items = counts["num_src_items"]
    num_tgt_items = counts["num_tgt_items"]
    expected_items = num_src_items + num_tgt_items
    if item_embeddings.shape[0] != expected_items:
        raise ValueError(
            f"Item embedding row mismatch: got {item_embeddings.shape[0]}, expected {expected_items}"
        )
    return np.concatenate(
        [
            user_embeddings,
            item_embeddings[:num_src_items],
            item_embeddings[num_src_items:num_src_items + num_tgt_items],
        ],
        axis=0,
    )


def generate_CrossDomain_semantics_LLM4CDSR_end2end_runtime_final_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_embs,
):
    _require_active_cold_runtime(config, "final-embedding generation")
    final_embeddings = generate_CrossDomain_semantics_LLM4CDSR_final_embs(
        config,
        modality,
        interaction,
        id_mapping,
        modality_embs,
    )
    final_embs_path = os.path.join(
        os.path.dirname(_item_emb_cache_path(config, modality)),
        f"{modality['name']}_final_emb_{modality['emb_pca']}.npy",
    )
    _update_runtime(
        config,
        {
            "final_embs_path": final_embs_path,
        },
    )
    return final_embeddings


def _inject_runtime_trainer():
    import utils.quick_start as quick_start_module

    quick_start_module.get_trainer = lambda: LLM4CDSREnd2EndRuntimeTrainer


_inject_runtime_trainer()
