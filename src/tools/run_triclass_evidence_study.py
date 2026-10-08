"""Run fixed-setting evidence ablations and collect finished remote logs."""

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.grounded_triclass_evidence_ablation import (
    DEFAULT_CONFIG, VARIANTS, build, load_experiment, locations,
)
from models.target_pattern_pseudo import atomic_json, file_sha256
from tools.run_grounded_binary_study import reference_results, test_metrics, write_csv
from tools.run_grounded_triclass_queue import available_ram_mib, gpu_memory


LOGGER = logging.getLogger("triclass-evidence-study")


def train(experiment, task, model, variant, run_dir):
    import utils.quick_start as quick
    from models.target_pattern_pseudo import apply_model_domain_profile
    from utils.configurator import Config

    _, _, parent = locations(experiment, task, run_dir)
    cache = parent / variant
    with (cache / "manifest.json").open() as handle:
        manifest = json.load(handle)
    if manifest.get("variant") != variant or manifest.get("candidate_routes_unchanged") is not True:
        raise ValueError("Missing completed evidence-only ablation cache.")
    weight, eta = experiment["tasks"][task]["backbones"][model]

    class EvidenceConfig(Config):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            apply_model_domain_profile(self)
            self["domain_profiles"] = {}
            self["data_path"] = str(Path(experiment["data_path"]).resolve())
            self["seed"] = [int(experiment["seed"])]
            self["triclass_cache_path"] = str(cache.resolve())
            self["pseudo_loss_weight"] = [float(weight)]
            self["near_support_weight"] = [float(eta)]
            self["pseudo_sampler_seed"] = int(experiment["seed"])
            self["evidence_ablation"] = variant
            self["log_model_suffix"] = "triclass_" + variant
            self["log_dir"] = str(Path(run_dir) / "log")
            self["hyper_parameters"] = ["seed", "pseudo_loss_weight", "near_support_weight"]

    with patch.object(quick, "Config", EvidenceConfig):
        quick.quick_start(model + "_Grounded_TriClass_Pseudo", "Amazon2014",
                          experiment["tasks"][task]["domains"], save_model=False)


def create_jobs(experiment, config_path, run_dir):
    jobs = []
    for task in experiment["tasks"]:
        for variant in experiment["variants"]:
            jobs.append(dict(id=f"build_{task}_{variant}", kind="build", task=task,
                             variant=variant, dependency=None, status="waiting"))
    for model in next(iter(experiment["tasks"].values()))["backbones"]:
        for task in experiment["tasks"]:
            for variant in experiment["variants"]:
                jobs.append(dict(id=f"{model}_{task}_{variant}", kind="train", task=task, model=model,
                    variant=variant, dependency=f"build_{task}_{variant}", status="waiting"))
    for job in jobs:
        job["command"] = [sys.executable, "-u", str(Path(__file__).resolve()), job["kind"],
                          "--config", str(config_path), "--run-dir", str(run_dir),
                          "--task", job["task"], "--variant", job["variant"]]
        if job["kind"] == "train":
            job["command"] += ["--model", job["model"]]
    return jobs


def summarize(experiment, directory, state):
    rows, confidence = [], []
    for task, settings in experiment["tasks"].items():
        for model, (weight, eta) in settings["backbones"].items():
            path = SRC / experiment["reference_run"] / f"{model}_{task}.log"
            ref, _ = reference_results(path, int(experiment["seed"]), weight, eta)
            rows.append(dict(task=task, model=model, variant="full_triclass", pseudo_loss_weight=weight,
                near_support_weight=eta, recall20=ref[0], ndcg20=ref[1], delta_recall20=0,
                delta_ndcg20=0, status="reference", log=str(path)))
            for variant in experiment["variants"]:
                job = next(j for j in state["jobs"] if j["id"] == f"{model}_{task}_{variant}")
                result = job.get("metrics")
                rows.append(dict(task=task, model=model, variant=variant, pseudo_loss_weight=weight,
                    near_support_weight=eta, recall20=result[0] if result else "", ndcg20=result[1] if result else "",
                    delta_recall20=result[0]-ref[0] if result else "", delta_ndcg20=result[1]-ref[1] if result else "",
                    status=job["status"], log=job.get("log", "")))
    write_csv(directory / "performance.csv", list(rows[0]), rows)
    for job in state["jobs"]:
        if job["kind"] != "build" or job["status"] != "complete":
            continue
        path = directory / "cache" / job["task"] / job["variant"] / "diagnostics.json"
        with path.open() as handle:
            report = json.load(handle)
        metrics = report["splits"]["test"]
        for eta, selected in metrics["selection"].items():
            confidence.append(dict(task=job["task"], variant=job["variant"], eta=eta,
                **{k: metrics[k] for k in ("users", "hit_auprc", "supported_auprc", "brier", "ece15")},
                threshold=report["selection"][eta]["threshold"],
                **{"selected_"+k: selected[k] for k in ("hit_precision", "near_fraction", "supported_precision", "user_coverage", "pairs")},
                diagnostics=str(path)))
    if confidence:
        write_csv(directory / "confidence.csv", list(confidence[0]), confidence)


def coordinate(experiment, config_path, directory, gpus, max_per_gpu, interval):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / "status.json"
        if state_path.exists():
            raise ValueError("Queue already exists. Do not overwrite or duplicate it.")
        state = dict(coordinator_pid=os.getpid(), started_at=time.strftime("%Y-%m-%d %H:%M:%S%z"),
            gpus=gpus, max_active_per_gpu=max_per_gpu, jobs=create_jobs(experiment, config_path, directory),
            protocol=dict(seed=experiment["seed"], fixed_backbone_settings=experiment["tasks"],
                search=False, api_calls=False, candidate_routes_unchanged=True, labels_and_user_splits_unchanged=True,
                removal="constant zero before normalization in fit and inference; zero effective linear column",
                threshold_selection="original rule on original overlap validation users only"),
            code_sha256={str(p.relative_to(SRC)): file_sha256(str(p)) for p in
                (Path(__file__).resolve(), SRC / "models/grounded_triclass_evidence_ablation.py", config_path)})
        active = {}
        by_id = {j["id"]: j for j in state["jobs"]}
        while True:
            for key, (process, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                job = by_id[key]
                job.update(returncode=code, finished_at=time.strftime("%Y-%m-%d %H:%M:%S%z"),
                           status="complete" if code == 0 else "failed")
                if code == 0 and job["kind"] == "train":
                    content = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
                    result = test_metrics(content.rsplit("Best Parameters:", 1)[1]) if "Best Parameters:" in content else None
                    if not result or "All Over" not in content:
                        job.update(status="failed", reason="Missing final metrics.")
                    else:
                        job["metrics"] = result
                LOGGER.info("Finished %s status=%s", key, job["status"])
                del active[key]
            for job in state["jobs"]:
                if job["status"] == "waiting" and job["dependency"] and by_id[job["dependency"]]["status"] == "failed":
                    job.update(status="blocked", reason="Required evidence cache failed.")
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
            summarize(experiment, directory, state)
            done = all(j["status"] in ("complete", "failed", "blocked") for j in state["jobs"])
            if done:
                state["finished_at"] = state["updated_at"]
            atomic_json(str(state_path), state)
            if done:
                LOGGER.info("Evidence ablation queue finished")
                return 0 if all(j["status"] == "complete" for j in state["jobs"]) else 1
            try:
                memory, ram = gpu_memory(), available_ram_mib()
            except (OSError, subprocess.SubprocessError) as error:
                LOGGER.warning("Resource probe failed: %s", error)
                time.sleep(interval)
                continue
            started = False
            if ram >= 14000:
                for job in state["jobs"]:
                    if job["status"] != "waiting" or (job["dependency"] and by_id[job["dependency"]]["status"] != "complete"):
                        continue
                    eligible = []
                    for gpu in gpus:
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
                    output = directory / (job["id"] + ".log")
                    stream = output.open("x", buffering=1)
                    process = subprocess.Popen(job["command"], cwd=SRC, env=env, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status="running", pid=process.pid, gpu=gpu, log=str(output),
                               started_at=time.strftime("%Y-%m-%d %H:%M:%S%z"))
                    active[job["id"]] = process, stream
                    LOGGER.info("Started %s pid=%d gpu=%d", job["id"], process.pid, gpu)
                    atomic_json(str(state_path), state)
                    started = True
                    break
            time.sleep(5 if started else interval)


def ssh(host, command):
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                                    command], text=True, timeout=60)


def collect(host, remote, local, interval):
    """Copy completed-run logs and reports; verify hashes and retain the originals."""
    local.mkdir(parents=True, exist_ok=True)
    with (local / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                state = json.loads(ssh(host, "cat " + shlex.quote(str(remote / "status.json"))))
                if not state.get("finished_at"):
                    counts = {s: sum(j["status"] == s for j in state["jobs"]) for s in ("complete", "running", "waiting", "failed")}
                    atomic_json(str(local / "collector_state.json"), dict(status="waiting", remote=host,
                        updated_at=time.strftime("%Y-%m-%d %H:%M:%S%z"), jobs=counts))
                    time.sleep(interval)
                    continue
                # The queue has finished writing training logs. Transfer only its
                # outputs, never unrelated server files, inputs or checkpoints.
                subprocess.run(["rsync", "-az", "--checksum", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                    "--exclude=code/", "--exclude=inputs/", "--exclude=cache/", "--include=*/",
                    "--include=*.log", "--include=*.csv", "--include=*.json", "--exclude=*",
                    host + ":" + str(remote) + "/", str(local) + "/"], check=True, timeout=1200)
                code = ("import hashlib,json,pathlib; p=pathlib.Path(" + repr(str(remote)) + "); "
                    "files=[f for f in p.rglob('*') if f.is_file() and f.suffix in ('.log','.csv','.json') "
                    "and not set(f.relative_to(p).parts).intersection(('code','inputs','cache'))]; "
                    "print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))")
                hashes = json.loads(ssh(host, "python3 -c " + shlex.quote(code)))
                if any(file_sha256(str(local / name)) != digest for name, digest in hashes.items()):
                    raise ValueError("Transferred files changed during copying; retrying.")
                atomic_json(str(local / "collector_state.json"), dict(status="complete", remote=host,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S%z"), verified_sha256=hashes,
                    remote_logs_retained=True, remote_failed_jobs=[j["id"] for j in state["jobs"] if j["status"] != "complete"]))
                LOGGER.info("Collected and verified %d files into %s", len(hashes), local)
                return 0
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                LOGGER.warning("Collection not ready: %s", error)
                time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("coordinate", "build", "train"):
        child = sub.add_parser(name)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.add_argument("--run-dir", type=Path, required=True)
        if name == "coordinate":
            child.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
            child.add_argument("--max-per-gpu", type=int, default=2)
            child.add_argument("--interval", type=int, default=15)
        else:
            child.add_argument("--task", choices=["ClothSport", "PhoneElec"], required=True)
            child.add_argument("--variant", choices=list(VARIANTS), required=True)
            if name == "train":
                child.add_argument("--model", choices=["EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB"], required=True)
    collector = sub.add_parser("collect")
    collector.add_argument("--host", required=True)
    collector.add_argument("--remote-dir", type=Path, required=True)
    collector.add_argument("--local-dir", type=Path, required=True)
    collector.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "collect":
        return collect(args.host, args.remote_dir, args.local_dir.resolve(), args.interval)
    experiment = load_experiment(args.config)
    directory = args.run_dir.resolve()
    if args.command == "coordinate":
        if args.max_per_gpu < 1 or args.interval < 1:
            parser.error("Queue limits must be positive.")
        return coordinate(experiment, args.config.resolve(), directory, args.gpus, args.max_per_gpu, args.interval)
    if args.command == "build":
        build(experiment, args.task, args.variant, directory)
    else:
        train(experiment, args.task, args.model, args.variant, directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
