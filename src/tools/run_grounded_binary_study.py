"""Build four binary caches, then run sixteen fixed-setting backbone ablations."""

import argparse
import ast
import csv
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.grounded_binary_pseudo import (
    DEFAULT_CONFIG, VARIANTS, build, cache_directory, load_experiment, read_json,
)
from models.target_pattern_pseudo import atomic_json, file_sha256
from tools.run_grounded_triclass_queue import available_ram_mib, gpu_memory


LOGGER = logging.getLogger("grounded-binary-study")


def test_metrics(text):
    if "Test:" not in text:
        return None
    section = text.split("Test:", 1)[1]
    values = [re.search(name + r"\s*:.*?@20=([0-9.]+)", section, re.IGNORECASE)
              for name in ("recall", "ndcg")]
    return tuple(float(match.group(1)) for match in values) if all(values) else None


def reference_results(path, seed, weight, eta):
    text = path.read_text(encoding="utf-8")
    if "All Over" not in text or "Best Parameters:" not in text:
        raise ValueError("Reference search did not finish: " + str(path))
    recap = text.split("All Over", 1)[1]
    pattern = r"(?m)^.*?INFO Parameters:\s*(\[[^\n]+\])\s*=\s*(\([^\n]+\))\s*$"
    matches = list(re.finditer(pattern, recap))
    results = {}
    for i, match in enumerate(matches):
        params = dict(zip(ast.literal_eval(match.group(1)), ast.literal_eval(match.group(2))))
        if params.get("seed") == seed and params.get("pseudo_loss_weight") == weight:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(recap)
            results[float(params["near_support_weight"])] = test_metrics(recap[match.end():end])
    if results.get(eta) is None or results.get(0.0) is None:
        raise ValueError("Missing matched TriClass configurations in " + str(path))
    return results[eta], results[0.0]


def train(experiment, task, model, variant):
    import utils.quick_start as quick_start_module
    from models.target_pattern_pseudo import apply_model_domain_profile
    from utils.configurator import Config

    directory = cache_directory(experiment, task, variant)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("variant") != variant or manifest.get("classifier_output_dim") != 2:
        raise ValueError("Expected a completed two-output binary cache.")
    weight, original_eta = experiment["tasks"][task]["backbones"][model]

    class BinaryStudyConfig(Config):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_model_domain_profile(self)
            self["domain_profiles"] = {}
            self["seed"] = [int(experiment["seed"])]
            self["triclass_cache_path"] = str(directory)
            self["pseudo_loss_weight"] = [float(weight)]
            self["near_support_weight"] = [0.0]
            self["pseudo_sampler_seed"] = int(experiment["seed"])
            self["ablation_classifier"] = variant
            self["ablation_classifier_class_names"] = VARIANTS[variant]
            self["reference_triclass_near_support_weight"] = original_eta
            self["log_model_suffix"] = "binary_" + variant
            self["hyper_parameters"] = ["seed", "pseudo_loss_weight", "near_support_weight"]

    LOGGER.info("Training task=%s backbone=%s binary=%s weight=%s cache=%s",
                task, model, variant, weight, directory)
    with patch.object(quick_start_module, "Config", BinaryStudyConfig):
        quick_start_module.quick_start(model + "_Grounded_TriClass_Pseudo", "Amazon2014",
                                        experiment["tasks"][task]["domains"], save_model=False)


def create_jobs(experiment, config_path):
    jobs = []
    for variant in experiment["variants"]:
        for task in experiment["tasks"]:
            jobs.append({"id": f"build_{task}_{variant}", "task": task, "variant": variant,
                         "kind": "build", "status": "waiting", "dependency": None})
    for model in next(iter(experiment["tasks"].values()))["backbones"]:
        for task in experiment["tasks"]:
            for variant in experiment["variants"]:
                jobs.append({"id": f"train_{model}_{task}_{variant}", "task": task, "model": model,
                             "variant": variant, "kind": "train", "status": "waiting",
                             "dependency": f"build_{task}_{variant}"})
    for job in jobs:
        job["command"] = [sys.executable, "-u", str(Path(__file__).resolve()), job["kind"],
                          "--config", str(config_path), "--task", job["task"], "--variant", job["variant"]]
        if job["kind"] == "train":
            job["command"] += ["--model", job["model"]]
    return jobs


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(directory, experiment, state):
    rows = []
    for task, settings in experiment["tasks"].items():
        for model, (weight, eta) in settings["backbones"].items():
            source = SRC / experiment["reference_run"] / f"{model}_{task}.log"
            best, hit = reference_results(source, int(experiment["seed"]), weight, eta)
            for variant, metrics, value in (("triclass_reference", best, eta), ("triclass_eta0", hit, 0.0)):
                rows.append(dict(task=task, model=model, variant=variant, pseudo_loss_weight=weight,
                    near_support_weight=value, test_recall20=metrics[0], test_ndcg20=metrics[1],
                    status="reused", log=str(source)))
    for job in state["jobs"]:
        if job["kind"] != "train":
            continue
        metrics = None
        if job["status"] == "complete":
            text = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
            if "Best Parameters:" in text:
                metrics = test_metrics(text.rsplit("Best Parameters:", 1)[1])
        rows.append(dict(task=job["task"], model=job["model"], variant=job["variant"],
            pseudo_loss_weight=experiment["tasks"][job["task"]]["backbones"][job["model"]][0],
            near_support_weight=0.0, test_recall20=metrics[0] if metrics else "",
            test_ndcg20=metrics[1] if metrics else "", status=job["status"], log=job.get("log", "")))
    write_csv(directory / "performance.csv", list(rows[0]), rows)
    confidence = []
    for job in state["jobs"]:
        if job["kind"] != "build" or job["status"] != "complete":
            continue
        path = cache_directory(experiment, job["task"], job["variant"]) / "diagnostics.json"
        report = read_json(path)
        test = report["splits"]["test"]
        confidence.append(dict(task=job["task"], variant=job["variant"], best_epoch=report["best_epoch"],
            threshold=report["selection"]["0.0"]["threshold"],
            **{key: value for key, value in test.items() if key not in ("selection", "users")},
            **{"selected_" + key: test["selection"][key] for key in
               ("hit_precision", "near_fraction", "supported_precision", "user_coverage", "pairs", "mean_slot_weight")},
            cold_coverage=report["cold_selection"]["user_coverage"], cold_pairs=report["cold_selection"]["pairs"],
            cold_mean_slot_weight=report["cold_selection"]["mean_slot_weight"], diagnostics=str(path)))
    if confidence:
        write_csv(directory / "confidence.csv", list(confidence[0]), confidence)


def coordinate(experiment, config_path, directory, gpu_ids, max_per_gpu, interval):
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "coordinator.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = directory / "status.json"
        if status_path.exists():
            raise ValueError("Run directory already contains a queue; use another run directory.")
        state = dict(coordinator_pid=os.getpid(), started_at=time.strftime("%Y-%m-%d %H:%M:%S%z"),
            gpus=gpu_ids, max_active_per_gpu=max_per_gpu, jobs=create_jobs(experiment, config_path),
            protocol={"seed": experiment["seed"], "classifier_variants": VARIANTS,
                      "backbone_settings": experiment["tasks"], "api_calls": False,
                      "reference_run": experiment["reference_run"], "search": False,
                      "same_candidate_features_and_user_splits": True,
                      "binary_threshold_objective": "validation positive precision with original coverage floor",
                      "triclass_eta0": "reused three-class model at matched loss weight; not a binary classifier"},
            code_sha256={str(p.relative_to(SRC)): file_sha256(str(p)) for p in
                         (Path(__file__).resolve(), SRC / "models/grounded_binary_pseudo.py", config_path)})
        summarize(directory, experiment, state)
        active, by_id = {}, {job["id"]: job for job in state["jobs"]}
        while True:
            for job_id, (process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                job = by_id[job_id]
                job.update(returncode=code, finished_at=time.strftime("%Y-%m-%d %H:%M:%S%z"))
                job["status"] = "complete" if code == 0 else "failed"
                if code == 0 and job["kind"] == "train":
                    text = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
                    if "Best Parameters:" not in text or test_metrics(text.rsplit("Best Parameters:", 1)[1]) is None:
                        job.update(status="failed", reason="No final recommendation metrics.")
                LOGGER.info("Finished %s status=%s code=%d", job_id, job["status"], code)
                del active[job_id]
            for job in state["jobs"]:
                if job["status"] == "waiting" and job["dependency"] and by_id[job["dependency"]]["status"] == "failed":
                    job.update(status="blocked", reason="Binary cache build failed.")
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
            atomic_json(str(status_path), state)
            summarize(directory, experiment, state)
            if all(job["status"] in ("complete", "failed", "blocked") for job in state["jobs"]):
                LOGGER.info("Binary study queue finished")
                return 0 if all(job["status"] == "complete" for job in state["jobs"]) else 1
            try:
                memory, ram = gpu_memory(), available_ram_mib()
            except (OSError, subprocess.SubprocessError) as exc:
                LOGGER.warning("Resource probe failed: %s", exc)
                time.sleep(interval)
                continue
            started = False
            if ram >= 12000:
                for job in state["jobs"]:
                    if job["status"] != "waiting" or (job["dependency"] and by_id[job["dependency"]]["status"] != "complete"):
                        continue
                    eligible = []
                    for gpu in gpu_ids:
                        occupants = [j for j in state["jobs"] if j["status"] == "running" and j.get("gpu") == gpu]
                        if any(j["kind"] == "build" for j in occupants) or (job["kind"] == "build" and occupants):
                            continue
                        if len(occupants) < max_per_gpu and memory.get(gpu, 0) >= 8000:
                            eligible.append(gpu)
                    if not eligible:
                        continue
                    gpu = max(eligible, key=memory.get)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                               OPENBLAS_NUM_THREADS="4", NUMEXPR_MAX_THREADS="4", PYTHONUNBUFFERED="1")
                    log = directory / (job["id"] + ".log")
                    stream = log.open("x", buffering=1)
                    process = subprocess.Popen(job["command"], cwd=SRC, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status="running", pid=process.pid, gpu=gpu, log=str(log),
                               started_at=time.strftime("%Y-%m-%d %H:%M:%S%z"))
                    active[job["id"]] = process, stream
                    LOGGER.info("Started %s pid=%d gpu=%d", job["id"], process.pid, gpu)
                    atomic_json(str(status_path), state)
                    started = True
                    break
            time.sleep(5 if started else interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    coordinator = sub.add_parser("coordinate")
    coordinator.add_argument("--run-dir", type=Path, required=True)
    coordinator.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    coordinator.add_argument("--max-per-gpu", type=int, default=2)
    coordinator.add_argument("--interval", type=int, default=15)
    coordinator.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for command in ("build", "train"):
        child = sub.add_parser(command)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--task", choices=["ClothSport", "PhoneElec"], required=True)
        child.add_argument("--variant", choices=list(VARIANTS), required=True)
        if command == "train":
            child.add_argument("--model", choices=["EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB"], required=True)
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "coordinate":
        if args.max_per_gpu < 1 or args.interval < 1:
            parser.error("Queue limits must be positive.")
        return coordinate(experiment, args.config.resolve(), args.run_dir.resolve(), args.gpus,
                          args.max_per_gpu, args.interval)
    if args.command == "build":
        build(experiment, args.task, args.variant)
    else:
        train(experiment, args.task, args.model, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
