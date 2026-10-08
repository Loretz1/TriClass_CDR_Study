"""Run the TriClass overlap-ratio study without changing backbone models."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score


SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
RATIOS = (0.2, 0.4, 0.6, 0.8, 1.0)
TASKS = {
    "ClothSport": ("Clothing_Shoes_and_Jewelry", "Sports_and_Outdoors"),
    "PhoneElec": ("Cell_Phones_and_Accessories", "Electronics"),
}
MODELS = ("EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB")
BEST = {
    ("EMCDR", "ClothSport"): (0.1, 0.25),
    ("UniCDR", "ClothSport"): (0.1, 0.0),
    ("CD_CDR_Normal", "ClothSport"): (1.0, 0.0),
    ("CDRIB", "ClothSport"): (0.1, 0.25),
    ("EMCDR", "PhoneElec"): (0.1, 0.0),
    ("UniCDR", "PhoneElec"): (1.0, 0.5),
    ("CD_CDR_Normal", "PhoneElec"): (1.0, 0.25),
    ("CDRIB", "PhoneElec"): (1.0, 0.5),
}
BASELINE_LOGS = {
    (model, task): SRC / "triclass_runs" / "20260908" / f"{model}_{task}.log"
    for model in MODELS for task in TASKS
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ratio_tag(ratio):
    return f"r{int(round(100 * ratio)):03d}"


def cache_path(task, ratio):
    from models.target_pattern_pseudo import dataset_split_directory
    from models.grounded_static_confidence import load_config

    domains = TASKS[task]
    config = load_config(
        str(SRC / "configs/model/GroundedTriClass.yaml"),
        "Amazon2014",
        domains,
        "cpu",
    )
    subdir = "grounded_triclass" if ratio == 1.0 else (
        "grounded_triclass_overlap_ratio_" + ratio_tag(ratio)
    )
    return Path(dataset_split_directory(config)) / subdir


def fixed_split(num_overlap, ratios, seed, train_fraction):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Expected the original train/validation/test split ratios.")
    order = np.random.default_rng(seed).permutation(num_overlap)
    full_train = int(num_overlap * ratios[0])
    validation = int(num_overlap * ratios[1])
    retained = max(2, int(full_train * train_fraction))
    if retained > full_train:
        raise ValueError("The retained training population exceeds the baseline population.")
    codes = np.full(num_overlap, 3, dtype=np.int8)
    codes[order[:retained]] = 0
    codes[order[full_train:full_train + validation]] = 1
    codes[order[full_train + validation:]] = 2
    return codes


def hardlink_teachers(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        if not source_path.is_file():
            continue
        destination_path = destination / source_path.name
        if destination_path.exists():
            continue
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path)


def ece_score(probabilities, labels, bins=15):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index + 1 == bins:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def confidence_metrics(task, ratio, output_path):
    from models.grounded_triclass_pseudo import support_report

    directory = cache_path(task, ratio)
    with np.load(directory / "overlap_support.npz", allow_pickle=False) as data:
        items = data["item_ids"]
        probabilities = data["probabilities"]
        labels = data["labels"]
        split_codes = data["split_codes"]
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    test_users = split_codes == 2
    valid = test_users[:, None] & (items > 0) & (labels >= 0)
    y = labels[valid].astype(np.int64)
    p = probabilities[valid].astype(np.float64)
    selected = {}
    for eta, setting in manifest["selection"].items():
        selected[eta] = support_report(
            items[test_users], probabilities[test_users], labels[test_users],
            int(manifest["top_k"]), float(eta), float(setting["threshold"]),
        )
        selected[eta]["threshold"] = float(setting["threshold"])

    counts = np.bincount(split_codes, minlength=4)
    one_hot = np.eye(3, dtype=np.float64)[y]
    result = {
        "task": task,
        "domains": list(TASKS[task]),
        "requested_training_overlap_ratio": ratio,
        "training_users": int(counts[0]),
        "validation_users": int(counts[1]),
        "test_users": int(counts[2]),
        "unused_overlap_users": int(counts[3]),
        "total_overlap_users": int(len(split_codes)),
        "actual_fraction_of_all_overlap_users": float(counts[0] / len(split_codes)),
        "candidate_class_counts": np.bincount(y, minlength=3).tolist(),
        "test_multiclass_ce": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "test_multiclass_brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "test_ece_15": ece_score(p, y),
        "test_hit_auprc": float(average_precision_score(y == 0, p[:, 0])),
        "test_supported_auprc": float(average_precision_score(y < 2, p[:, 0] + p[:, 1])),
        "selection_test": selected,
        "cache": str(directory),
    }
    atomic_json(output_path, result)
    return result


def build_ratio(task, ratio, metrics_path):
    import models.grounded_triclass_pseudo as triclass
    from models.grounded_static_confidence import load_config

    if ratio == 1.0:
        return confidence_metrics(task, ratio, metrics_path)

    domains = TASKS[task]
    config = load_config(
        str(SRC / "configs/model/GroundedTriClass.yaml"),
        "Amazon2014",
        domains,
        "auto",
    )
    config["output_subdir"] = "grounded_triclass_overlap_ratio_" + ratio_tag(ratio)
    config["overlap_train_ratio"] = ratio
    config["overlap_ratio_protocol"] = "fixed_validation_test_nested_training_v1"
    config["ratio_runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = cache_path(task, ratio)
    base = cache_path(task, 1.0)
    hardlink_teachers(base / "target_only_teachers", destination / "target_only_teachers")

    original_split = triclass.split_overlap_users
    triclass.split_overlap_users = lambda n, ratios, seed: fixed_split(n, ratios, seed, ratio)
    try:
        triclass.build(config, "all")
    finally:
        triclass.split_overlap_users = original_split
    return confidence_metrics(task, ratio, metrics_path)


def train_ratio(model, task, ratio):
    import utils.quick_start as quick_start_module
    from models.target_pattern_pseudo import apply_model_domain_profile
    from utils.configurator import Config as BaseConfig

    pseudo_weight, near_weight = BEST[(model, task)]
    ratio_cache = str(cache_path(task, ratio))

    class StudyConfig(BaseConfig):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_model_domain_profile(self)
            self["domain_profiles"] = {}
            self["seed"] = [999]
            self["triclass_cache_path"] = ratio_cache
            self["pseudo_loss_weight"] = [pseudo_weight]
            self["near_support_weight"] = [near_weight]
            self["pseudo_sampler_seed"] = 999
            self["log_model_suffix"] = "triclass_overlap_ratio_" + ratio_tag(ratio)
            self["hyper_parameters"] = ["seed", "pseudo_loss_weight", "near_support_weight"]

    quick_start_module.Config = StudyConfig
    quick_start_module.quick_start(
        model + "_Grounded_TriClass_Pseudo",
        "Amazon2014",
        list(TASKS[task]),
        save_model=False,
    )


def parse_best_metrics(path):
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    positions = [match.start() for match in re.finditer(r"Best Parameters:", text)]
    if not positions:
        return None
    block = text[positions[-1]:]
    test_position = block.find("Test:")
    if test_position < 0:
        return None
    test = block[test_position:]
    recall = re.search(r"recall\s*:.*?@20=([0-9.]+)", test, re.IGNORECASE)
    ndcg = re.search(r"ndcg\s*:.*?@20=([0-9.]+)", test, re.IGNORECASE)
    if not recall or not ndcg:
        return None
    return float(recall.group(1)), float(ndcg.group(1))


def write_summaries(run_dir, state):
    performance_rows = []
    for model in MODELS:
        for task in TASKS:
            baseline = parse_best_metrics(BASELINE_LOGS[(model, task)])
            if baseline:
                performance_rows.append([task, model, 1.0, *BEST[(model, task)], *baseline, "reused"])
            for ratio in RATIOS[:-1]:
                job_id = f"train_{model}_{task}_{ratio_tag(ratio)}"
                job = next((item for item in state["jobs"] if item["id"] == job_id), None)
                metrics = parse_best_metrics(job["log"]) if job and job["status"] == "complete" else None
                performance_rows.append([
                    task, model, ratio, *BEST[(model, task)],
                    metrics[0] if metrics else "", metrics[1] if metrics else "",
                    job["status"] if job else "waiting",
                ])
    path = run_dir / "performance.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "model", "training_overlap_ratio", "pseudo_loss_weight",
                         "near_support_weight", "test_recall@20", "test_ndcg@20", "status"])
        writer.writerows(performance_rows)

    confidence_rows = []
    for task in TASKS:
        for ratio in RATIOS:
            path = run_dir / "confidence" / f"{task}_{ratio_tag(ratio)}.json"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            for eta, report in result["selection_test"].items():
                confidence_rows.append([
                    task, ratio, result["training_users"], result["actual_fraction_of_all_overlap_users"],
                    result["test_multiclass_ce"], result["test_multiclass_brier"], result["test_ece_15"],
                    result["test_hit_auprc"], result["test_supported_auprc"], eta,
                    report["threshold"], report["hit_precision"], report["supported_precision"],
                    report["near_fraction"], report["user_coverage"], report["pairs"],
                ])
    path = run_dir / "confidence.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "training_overlap_ratio", "training_users", "fraction_of_all_overlap_users",
                         "test_ce", "test_brier", "test_ece15", "test_hit_auprc",
                         "test_supported_auprc", "eta", "threshold", "selected_hit_precision",
                         "selected_supported_precision", "selected_near_fraction",
                         "selected_user_coverage", "selected_pairs"])
        writer.writerows(confidence_rows)


def create_jobs(run_dir):
    jobs = []
    for task in TASKS:
        for ratio in RATIOS:
            jobs.append({
                "id": f"build_{task}_{ratio_tag(ratio)}",
                "kind": "build",
                "status": "waiting",
                "dependency": None,
                "task": task,
                "ratio": ratio,
                "command": [sys.executable, "-u", str(Path(__file__)), "build",
                            "--task", task, "--ratio", str(ratio),
                            "--metrics", str(run_dir / "confidence" / f"{task}_{ratio_tag(ratio)}.json")],
            })
    for model in MODELS:
        for task in TASKS:
            for ratio in RATIOS[:-1]:
                jobs.append({
                    "id": f"train_{model}_{task}_{ratio_tag(ratio)}",
                    "kind": "train",
                    "status": "waiting",
                    "dependency": f"build_{task}_{ratio_tag(ratio)}",
                    "task": task,
                    "model": model,
                    "ratio": ratio,
                    "command": [sys.executable, "-u", str(Path(__file__)), "train",
                                "--task", task, "--ratio", str(ratio), "--model", model],
                })
    return jobs


def gpu_memory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True, timeout=15)
    return {int(parts[0]): int(parts[1]) for line in output.splitlines()
            if (parts := [part.strip() for part in line.split(",")])}


def available_ram_mib():
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("Cannot read available host memory.")


def coordinate(run_dir, gpu_ids, max_per_gpu, interval):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "confidence").mkdir(exist_ok=True)
    with (run_dir / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = run_dir / "status.json"
        if state_path.exists():
            raise RuntimeError("Queue state already exists; choose a new run directory.")
        state = {
            "coordinator_pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
            "protocol": {
                "ratios": RATIOS,
                "meaning": "fraction of the fixed baseline TriClass training-overlap population",
                "fixed_validation_test": True,
                "nested_training_subsets": True,
                "backbone_hyperparameters": "fixed full-overlap TriClass best settings",
                "full_overlap_results": "reused from src/triclass_runs/20260908",
            },
            "jobs": create_jobs(run_dir),
            "gpus": gpu_ids,
            "max_active_per_gpu": max_per_gpu,
        }
        active = {}
        by_id = {job["id"]: job for job in state["jobs"]}
        logger = logging.getLogger("triclass-overlap-ratio")
        while True:
            for job_id, (process, output) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                output.close()
                job = by_id[job_id]
                job.update(returncode=code, finished_at=time.strftime("%Y-%m-%d %H:%M:%S%z"))
                job["status"] = "complete" if code == 0 else "failed"
                logger.info("Finished %s status=%s code=%s", job_id, job["status"], code)
                del active[job_id]
            for job in state["jobs"]:
                dependency = job["dependency"]
                if job["status"] == "waiting" and dependency and by_id[dependency]["status"] == "failed":
                    job.update(status="blocked", reason="Required ratio cache failed.")
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
            atomic_json(state_path, state)
            write_summaries(run_dir, state)
            if all(job["status"] in ("complete", "failed", "blocked") for job in state["jobs"]):
                logger.info("Overlap-ratio study queue finished")
                return 0 if all(job["status"] == "complete" for job in state["jobs"]) else 1
            try:
                memory = gpu_memory()
                ram = available_ram_mib()
            except (OSError, subprocess.SubprocessError) as error:
                logger.warning("Resource probe failed: %s", error)
                time.sleep(interval)
                continue
            if ram < 12000:
                time.sleep(interval)
                continue
            started = False
            for job in state["jobs"]:
                if job["status"] != "waiting":
                    continue
                dependency = job["dependency"]
                if dependency and by_id[dependency]["status"] != "complete":
                    continue
                eligible = []
                for gpu in gpu_ids:
                    occupants = [item for item in state["jobs"]
                                 if item["status"] == "running" and item.get("gpu") == gpu]
                    if job["kind"] == "build" and occupants:
                        continue
                    if any(item["kind"] == "build" for item in occupants):
                        continue
                    required = 12000 if job["kind"] == "build" else 8000
                    if len(occupants) < max_per_gpu and memory.get(gpu, 0) >= required:
                        eligible.append(gpu)
                if not eligible:
                    continue
                gpu = max(eligible, key=lambda value: memory[value])
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                                   NUMEXPR_MAX_THREADS="4", PYTHONUNBUFFERED="1")
                log_path = run_dir / (job["id"] + ".log")
                output = log_path.open("x", buffering=1)
                process = subprocess.Popen(
                    job["command"], cwd=SRC, env=environment, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                )
                job.update(status="running", pid=process.pid, gpu=gpu, log=str(log_path),
                           started_at=time.strftime("%Y-%m-%d %H:%M:%S%z"))
                active[job["id"]] = (process, output)
                logger.info("Started %s pid=%d gpu=%d", job["id"], process.pid, gpu)
                atomic_json(state_path, state)
                started = True
                break
            time.sleep(5 if started else interval)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    coordinator = subparsers.add_parser("coordinate")
    coordinator.add_argument("--run-dir", required=True, type=Path)
    coordinator.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    coordinator.add_argument("--max-per-gpu", type=int, default=2)
    coordinator.add_argument("--interval", type=int, default=15)
    builder = subparsers.add_parser("build")
    builder.add_argument("--task", choices=TASKS, required=True)
    builder.add_argument("--ratio", choices=RATIOS, type=float, required=True)
    builder.add_argument("--metrics", type=Path, required=True)
    trainer = subparsers.add_parser("train")
    trainer.add_argument("--task", choices=TASKS, required=True)
    trainer.add_argument("--ratio", choices=RATIOS, type=float, required=True)
    trainer.add_argument("--model", choices=MODELS, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(SRC)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "coordinate":
        return coordinate(args.run_dir.resolve(), args.gpus, args.max_per_gpu, args.interval)
    if args.command == "build":
        build_ratio(args.task, args.ratio, args.metrics.resolve())
        return 0
    train_ratio(args.model, args.task, args.ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
