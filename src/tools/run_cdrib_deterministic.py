"""Rerun paired CDRIB budgets using immutable inputs and audited repeats."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

# This must precede torch imports and CUDA context initialization.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from models.cdrib_grounded_triclass_deterministic import enable_determinism, weights_digest
from models.target_pattern_pseudo import atomic_json, file_sha256, dataset_split_directory
from tools import run_triclass_paired_overlap as paired
from tools.run_grounded_binary_study import write_csv

MODEL = "CDRIB_Grounded_TriClass_Deterministic"
DEFAULT_INPUT = SRC / "triclass_paired_overlap_runs/20260914"


def job_id(task, ratio, variant, repeat=0):
    return f"CDRIB_{task}_{paired.tag(ratio)}_{variant}_repeat{repeat}"


def source_files():
    files = list((SRC / "utils").rglob("*.py")) + list((SRC / "common").rglob("*.py"))
    files += [SRC / "models" / name for name in (
        "cdrib.py", "cdrib_grounded_triclass_pseudo.py", "cdrib_grounded_triclass_deterministic.py",
        "grounded_triclass_pseudo.py", "grounded_static_confidence.py", "direct_semantic_pseudo.py",
        "target_pattern_pseudo.py")]
    files += [SRC / "tools" / name for name in (
        "run_cdrib_deterministic.py", "run_triclass_paired_overlap.py", "run_grounded_triclass_queue.py",
        "run_grounded_binary_study.py", "run_triclass_overlap_ratio_study.py")]
    files += [SRC / "configs" / name for name in (
        "overall.yaml", "dataset/Amazon2014.yaml", "model/CDRIB_Grounded_TriClass_Deterministic.yaml",
        "model/GroundedTriClass.yaml", "model/TriClassPairedOverlap.yaml")]
    return sorted(set(files))


def verify_sources(run_dir):
    manifest = paired.read_json(run_dir / "source_manifest.json")
    for name, digest in manifest.items():
        if file_sha256(str(SRC / name)) != digest:
            raise RuntimeError("Source changed after queue launch: " + name)
    return manifest


def configure(config, experiment, task, ratio, variant, inputs, run_dir, repeat):
    config["model"] = "CDRIB_Grounded_TriClass_Pseudo"
    paired.training_config(config, experiment, task, ratio, variant, inputs)
    config["model"] = MODEL
    name = job_id(task, ratio, variant, repeat)
    config["log_dir"] = str(run_dir / "benchmark_logs")
    config["log_model_suffix"] = name
    config["repro_trace_path"] = str(run_dir / (name + ".trace.jsonl"))
    config["openai_api_key"] = ""


def verify_inputs(config, inputs, task, ratio):
    spec_path = inputs / "preparation" / (task + "_" + paired.tag(ratio) + ".json")
    spec = paired.read_json(spec_path)
    view = Path(dataset_split_directory(config))
    for name, digest in spec["view_sha256"].items():
        if file_sha256(str(view / name)) != digest:
            raise RuntimeError("Prepared data changed: " + str(view / name))
    cache = Path(config["triclass_cache_path"])
    manifest = paired.read_json(cache / "manifest.json")
    cold = cache / "cold_support.npz"
    if file_sha256(str(cold)) != manifest["artifacts"]["cold_support.npz"]:
        raise RuntimeError("Pseudo supervision cache changed.")
    return dict(data=spec["view_sha256"], preparation_sha256=file_sha256(str(spec_path)),
                cache_manifest_sha256=file_sha256(str(cache / "manifest.json")),
                cold_support_sha256=manifest["artifacts"]["cold_support.npz"])


def train(args):
    import utils.quick_start as quick
    from utils.configurator import Config
    from common.trainer import Trainer

    enable_determinism()
    torch.set_num_threads(4)
    experiment = paired.load_experiment(paired.DEFAULT_CONFIG)
    sources = verify_sources(args.run_dir)
    name = job_id(args.task, args.ratio, args.variant, args.repeat)
    output = args.run_dir / (name + ".result.json")
    if output.exists():
        raise RuntimeError("Refusing to overwrite a completed result.")
    provenance = {}

    class RunConfig(Config):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            configure(self, experiment, args.task, args.ratio, args.variant,
                      args.inputs, args.run_dir, args.repeat)
            provenance.update(verify_inputs(self, args.inputs, args.task, args.ratio))
            provenance["effective_config"] = json.loads(json.dumps(self.final_config_dict, default=str))

    class AuditedTrainer(Trainer):
        def fit(self, *positional, **keywords):
            result = super().fit(*positional, **keywords)
            provenance["validation_and_test"] = result[0]
            provenance["stop_epoch"] = result[1][1]
            provenance["final_weights_sha256"] = weights_digest(self.model)
            return result

    with patch.object(quick, "Config", RunConfig), \
            patch.object(quick, "check_and_prepare_dataset", paired.require_prepared), \
            patch.object(quick, "get_trainer", lambda: AuditedTrainer):
        quick.quick_start(MODEL, "Amazon2014", experiment["tasks"][args.task]["domains"], save_model=False)
    verify_sources(args.run_dir)
    provenance.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), gpu_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        num_threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"], source_sha256=sources,
        trace_sha256=file_sha256(str(args.run_dir / (name + ".trace.jsonl"))))
    atomic_json(str(output), provenance)


def check_repeat(run_dir, task, variant):
    paths = [run_dir / (job_id(task, 1.0, variant, r) + ".result.json") for r in (0, 1)]
    if not all(path.exists() for path in paths):
        return dict(status="waiting")
    a, b = [paired.read_json(path) for path in paths]
    keys = ("validation_and_test", "stop_epoch", "final_weights_sha256", "trace_sha256",
            "data", "cache_manifest_sha256", "cold_support_sha256", "source_sha256",
            "torch_version", "cuda_version", "gpu", "num_threads", "deterministic", "cublas_workspace")
    different = [key for key in keys if a[key] != b[key]]
    return dict(status="failed" if different else "passed", different=different,
                primary=str(paths[0]), repeat=str(paths[1]))


def make_jobs(run_dir, inputs):
    result = []
    for repeat in (0, 1):
        for task in ("ClothSport", "PhoneElec"):
            for variant in ("pure", "triclass"):
                result.append(dict(task=task, ratio=1.0, variant=variant, repeat=repeat,
                    dependency=job_id(task, 1.0, variant) if repeat else None))
    for ratio in (.2, .4, .6, .8):
        for task in ("ClothSport", "PhoneElec"):
            for variant in ("pure", "triclass"):
                result.append(dict(task=task, ratio=ratio, variant=variant, repeat=0,
                    dependency=job_id(task, 1.0, variant, 1)))
    for job in result:
        job.update(id=job_id(job["task"], job["ratio"], job["variant"], job["repeat"]),
                   kind="train", model="CDRIB", status="waiting")
        job["command"] = [sys.executable, "-u", str(Path(__file__).resolve()), "train",
            "--run-dir", str(run_dir), "--inputs", str(inputs), "--task", job["task"],
            "--ratio", str(job["ratio"]), "--variant", job["variant"], "--repeat", str(job["repeat"])]
    return result


def summarize(run_dir, state):
    checks = {(t, v): check_repeat(run_dir, t, v) for t in ("ClothSport", "PhoneElec")
              for v in ("pure", "triclass")}
    for job in state["jobs"]:
        if job["status"] == "complete":
            path = run_dir / (job["id"] + ".result.json")
            if not path.exists():
                job.update(status="failed", reason="Missing audited result.")
            elif job["repeat"] and checks[(job["task"], job["variant"])]["status"] != "passed":
                job.update(status="failed", reason="Full-run reproducibility check failed.")
    by_id = {job["id"]: job for job in state["jobs"]}
    changed = True
    while changed:
        changed = False
        for job in state["jobs"]:
            dependency = job["dependency"]
            if job["status"] == "waiting" and dependency and by_id[dependency]["status"] in ("failed", "blocked"):
                job.update(status="blocked", reason="A prerequisite failed validation.")
                changed = True
    atomic_json(str(run_dir / "reproducibility.json"), {t + "_" + v: c for (t, v), c in checks.items()})
    rows = []
    for job in state["jobs"]:
        if job["repeat"]:
            continue
        metrics = None
        if job["status"] == "complete":
            result = paired.read_json(run_dir / (job["id"] + ".result.json"))
            test = result["validation_and_test"][5]
            metrics = (test["recall@20"], test["ndcg@20"])
        rows.append(dict(task=job["task"], model="CDRIB", ratio=job["ratio"], variant=job["variant"],
            recall20=metrics[0] if metrics else "", ndcg20=metrics[1] if metrics else "",
            status=job["status"], log=job.get("log", "")))
    write_csv(run_dir / "performance.csv", list(rows[0]), rows)
    pairs = []
    for p in rows:
        if p["variant"] != "pure":
            continue
        t = next(r for r in rows if r["variant"] == "triclass" and r["task"] == p["task"] and r["ratio"] == p["ratio"])
        done = p["status"] == t["status"] == "complete"
        pairs.append(dict(task=p["task"], model="CDRIB", ratio=p["ratio"],
            pure_recall20=p["recall20"], pure_ndcg20=p["ndcg20"],
            triclass_recall20=t["recall20"], triclass_ndcg20=t["ndcg20"],
            delta_recall20=t["recall20"] - p["recall20"] if done else "",
            delta_ndcg20=t["ndcg20"] - p["ndcg20"] if done else "",
            pure_status=p["status"], triclass_status=t["status"]))
    write_csv(run_dir / "paired_results.csv", list(pairs[0]), pairs)


def coordinate(args):
    from tools import run_grounded_triclass_queue as queue
    args.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (args.run_dir / "status.json").exists() or (args.run_dir / "source_manifest.json").exists():
        raise RuntimeError("Use a new run directory; previous results are immutable.")
    files = source_files()
    atomic_json(str(args.run_dir / "source_manifest.json"),
                {str(path.relative_to(SRC)): file_sha256(str(path)) for path in files})
    with tarfile.open(args.run_dir / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(SRC)))
    (args.run_dir / "source_snapshot.tar.gz").chmod(0o600)
    handler = logging.FileHandler(args.run_dir / "coordinator.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    def save_state(path, state):
        summarize(args.run_dir, state)
        state["protocol"] = dict(seed=999, input_run=str(args.inputs),
            primary_runs=20, full_budget_repeats=4, sparse_operator="deterministic_sparse_bmm",
            low_budgets_require_passed_repeat=True, api_calls=False)
        atomic_json(str(path), state)

    with patch.object(queue, "jobs", lambda: make_jobs(args.run_dir, args.inputs)), \
            patch.object(queue, "atomic_json", save_state):
        return queue.run(args.run_dir, args.gpus, args.max_per_gpu, 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "train", "summarize"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--task", choices=("ClothSport", "PhoneElec"))
    parser.add_argument("--ratio", type=float, choices=(.2, .4, .6, .8, 1.0))
    parser.add_argument("--variant", choices=("pure", "triclass"))
    parser.add_argument("--repeat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-per-gpu", type=int, default=2)
    args = parser.parse_args()
    if args.max_per_gpu < 1:
        parser.error("max-per-gpu must be positive.")
    args.run_dir, args.inputs = args.run_dir.resolve(), args.inputs.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "queue":
        return coordinate(args)
    if args.mode == "train":
        if args.task is None or args.ratio is None or args.variant is None:
            parser.error("Training requires task, ratio and variant.")
        train(args)
    else:
        summarize(args.run_dir, paired.read_json(args.run_dir / "status.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
