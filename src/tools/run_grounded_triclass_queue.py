"""Run two shared offline builds followed by eight tri-class backbone searches."""

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time


SRC = Path(__file__).resolve().parents[1]
DOMAINS = {
    "ClothSport": ["Clothing_Shoes_and_Jewelry", "Sports_and_Outdoors"],
    "PhoneElec": ["Cell_Phones_and_Accessories", "Electronics"],
}
MODELS = ["EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB"]


def atomic_json(path, content):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def gpu_memory():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
    ], text=True, timeout=15)
    return {int(parts[0]): int(parts[1]) for line in output.splitlines()
            if (parts := line.split(","))}


def available_ram_mib():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("Cannot read available host memory.")


def jobs():
    result = []
    for task, domains in DOMAINS.items():
        result.append({"id": "build_" + task, "task": task, "kind": "build",
                       "status": "waiting", "dependency": None,
                       "command": [sys.executable, "-u", "-m", "models.grounded_triclass_pseudo",
                                   "--dataset", "Amazon2014", "--domains", *domains]})
    for model in MODELS:
        for task, domains in DOMAINS.items():
            result.append({"id": model + "_" + task, "task": task, "kind": "search",
                           "status": "waiting", "dependency": "build_" + task,
                           "combinations": 9, "model": model + "_Grounded_TriClass_Pseudo",
                           "command": [sys.executable, "-u", "main.py", "--model",
                                       model + "_Grounded_TriClass_Pseudo", "--dataset",
                                       "Amazon2014", "--domains", *domains]})
    return result


def run(directory, gpu_ids, max_per_gpu, interval):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / "status.json"
        if state_path.exists():
            raise RuntimeError("Queue state already exists. Inspect it before creating another run.")
        state = {"coordinator_pid": os.getpid(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
                 "cwd": str(SRC), "jobs": jobs(), "gpus": gpu_ids,
                 "max_active_per_gpu": max_per_gpu}
        active = {}
        logger = logging.getLogger("triclass-queue")
        by_id = {job["id"]: job for job in state["jobs"]}
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
                    job.update(status="blocked", reason="Offline build failed; no search was launched.")
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
            atomic_json(state_path, state)
            if all(job["status"] in ("complete", "failed", "blocked") for job in state["jobs"]):
                logger.info("Queue finished")
                return 0 if all(j["status"] == "complete" for j in state["jobs"]) else 1
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
                    occupants = [j for j in state["jobs"] if j["status"] == "running" and j["gpu"] == gpu]
                    # Builders reserve a GPU from new searches until their cache is finished.
                    if any(j["kind"] == "build" for j in occupants):
                        continue
                    if job["kind"] == "build" and occupants:
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
                log_path = directory / (job["id"] + ".log")
                output = log_path.open("x", buffering=1)
                process = subprocess.Popen(job["command"], cwd=SRC, env=environment,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True)
                job.update(status="running", pid=process.pid, gpu=gpu, log=str(log_path),
                           started_at=time.strftime("%Y-%m-%d %H:%M:%S%z"))
                active[job["id"]] = (process, output)
                logger.info("Started %s pid=%d gpu=%d log=%s", job["id"], process.pid, gpu, log_path)
                atomic_json(state_path, state)
                started = True
                # Wait for allocations to become visible before admitting another job.
                break
            time.sleep(5 if started else interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_per_gpu < 1 or args.interval < 1:
        parser.error("Queue limits and interval must be positive.")
    return run(args.run_dir.resolve(), args.gpus, args.max_per_gpu, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
