"""Build per-user count-matched candidates and run 40 fixed-setting trials."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from models import grounded_triclass_pseudo as triclass
from models.grounded_retrieval_budget import (
    DEFAULT_CONFIG, PROTOCOL, VARIANTS, FixedBudgetSupervisor, build, load_experiment, locations,
)
from models.target_pattern_pseudo import apply_model_domain_profile, atomic_json, file_sha256
from tools.run_grounded_binary_study import write_csv
from tools.run_triclass_paired_overlap import require_prepared


def configure(config, experiment, task, model, variant, run_dir):
    apply_model_domain_profile(config)
    _, _, _, cache = locations(experiment, task, run_dir)
    root = Path(experiment["data_path"])
    weight, eta = experiment["tasks"][task]["backbones"][model]
    config["domain_profiles"] = {}
    config["data_path"] = str((root if root.is_absolute() else SRC / root).resolve())
    config["seed"] = [experiment["seed"]]
    config["pseudo_loss_weight"] = [weight]
    config["near_support_weight"] = [eta]
    config["pseudo_sampler_seed"] = experiment["seed"]
    config["retrieval_cache_path"] = str(cache)
    config["retrieval_variant"] = variant
    config["log_dir"] = str(run_dir / "benchmark_logs")
    config["log_model_suffix"] = f"matched_retrieval_{task}_{variant}"
    config["hyper_parameters"] = ["seed", "pseudo_loss_weight", "near_support_weight"]
    config["reuse_pretrained_stage_cache"] = False
    config["openai_api_key"] = ""


def train(experiment, task, model, variant, run_dir):
    import torch
    import utils.quick_start as quick
    from common.trainer import Trainer
    from utils.configurator import Config

    torch.set_num_threads(experiment["num_threads"])
    if model == "CDRIB":
        from models.cdrib_grounded_triclass_deterministic import enable_determinism
        enable_determinism()
    model_name = experiment["cdrib_model"] if model == "CDRIB" else model + "_Grounded_TriClass_Pseudo"
    name = f"{model}_{task}_{variant}"
    if (run_dir / (name + ".result.json")).exists():
        raise ValueError("Do not overwrite a completed trial.")
    provenance = dict(protocol=PROTOCOL, task=task, model=model, variant=variant,
                      torch=torch.__version__, cuda=torch.version.cuda, deterministic=torch.are_deterministic_algorithms_enabled())

    class StudyConfig(Config):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            configure(self, experiment, task, model, variant, run_dir)
            provenance["config"] = json.loads(json.dumps(self.final_config_dict, default=str))
            path = Path(self["retrieval_cache_path"]) / "manifest.json"
            provenance["cache_manifest_sha256"] = file_sha256(str(path))

    class StudyTrainer(Trainer):
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            provenance.update(validation_and_test=result[0], stop_epoch=result[1][1])
            return result

    with patch.object(quick, "Config", StudyConfig), patch.object(quick, "get_trainer", lambda: StudyTrainer), \
            patch.object(quick, "check_and_prepare_dataset", require_prepared), \
            patch.object(triclass, "TriClassPseudoSupervisor", FixedBudgetSupervisor):
        quick.quick_start(model_name, "Amazon2014", experiment["tasks"][task]["domains"], save_model=False)
    atomic_json(str(run_dir / (name + ".result.json")), provenance)


def create_jobs(experiment, config_path, run_dir, skip_build=False):
    jobs = []
    for task in experiment["tasks"]:
        if not skip_build:
            jobs.append(dict(id="build_" + task, kind="build", task=task, dependency=None, status="waiting"))
    for model in next(iter(experiment["tasks"].values()))["backbones"]:
        for task in experiment["tasks"]:
            for variant in VARIANTS:
                jobs.append(dict(id=f"{model}_{task}_{variant}", kind="train", model=model, task=task,
                    variant=variant, dependency=None if skip_build else "build_" + task, status="waiting"))
    for job in jobs:
        job["command"] = [sys.executable, "-u", str(Path(__file__).resolve()), job["kind"],
            "--config", str(config_path), "--run-dir", str(run_dir), "--task", job["task"]]
        if job["kind"] == "train":
            job["command"] += ["--model", job["model"], "--variant", job["variant"]]
    return jobs


def summarize(experiment, run_dir, state):
    rows = []
    for job in state["jobs"]:
        if job["kind"] != "train":
            continue
        metrics = None
        if job["status"] == "complete":
            path = run_dir / (job["id"] + ".result.json")
            if path.exists():
                result = json.loads(path.read_text())["validation_and_test"][5]
                metrics = [result["recall@20"], result["ndcg@20"]]
            else:
                job.update(status="failed", reason="Missing final audited metrics.")
        rows.append(dict(task=job["task"], model=job["model"], variant=job["variant"],
            recall20=metrics[0] if metrics else "", ndcg20=metrics[1] if metrics else "",
            status=job["status"], log=job.get("log", "")))
    write_csv(run_dir / "performance.csv", list(rows[0]), rows)


def coordinate(args, experiment):
    from tools import run_grounded_triclass_queue as queue
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_build:
        for task in experiment["tasks"]:
            path = args.run_dir / "cache" / task / "manifest.json"
            if json.loads(path.read_text())["protocol"] != PROTOCOL:
                raise ValueError("Missing fully prepared matched-retrieval cache.")

    def save_state(path, state):
        summarize(experiment, args.run_dir, state)
        if all(job["status"] in ("complete", "failed", "blocked") for job in state["jobs"]):
            state.setdefault("finished_at", time.strftime("%Y-%m-%d %H:%M:%S%z"))
        state["protocol"] = dict(experiment, protocol=PROTOCOL, search=False, api_calls=False,
            original_three_route_candidates_unchanged=True, uniform_final_weights=True,
            common_negative_pool=True, cdrib_deterministic=True)
        atomic_json(str(path), state)

    with patch.object(queue, "jobs", lambda: create_jobs(experiment, args.config, args.run_dir, args.skip_build)), \
            patch.object(queue, "atomic_json", save_state):
        return queue.run(args.run_dir, args.gpus, args.max_per_gpu, 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "train", "coordinate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("ClothSport", "PhoneElec"))
    parser.add_argument("--model", choices=("EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-per-gpu", type=int, default=2)
    args = parser.parse_args()
    args.config, args.run_dir = args.config.resolve(), args.run_dir.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    experiment = load_experiment(args.config)
    if args.mode == "coordinate":
        return coordinate(args, experiment)
    if not args.task or (args.mode == "train" and (not args.model or not args.variant)):
        parser.error("Build needs task; train also needs model and variant.")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "build":
        build(experiment, args.task, args.run_dir)
    else:
        train(experiment, args.task, args.model, args.variant, args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
