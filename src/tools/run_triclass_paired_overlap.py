"""Paired Pure/TriClass runs with genuinely reduced overlap target supervision."""

import argparse
import copy
import fcntl
import json
import logging
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.grounded_triclass_pseudo import (
    CLASS_NAMES, FEATURE_NAMES, VERSION, choose_thresholds, digest_json,
    exact_overlap_votes, fit_calibrator, histories, predict_probabilities,
    read_json, split_overlap_users, support_labels, support_report,
)
from models.grounded_static_confidence import load_config
from models.target_pattern_pseudo import (
    apply_model_domain_profile, atomic_json, atomic_npz, atomic_torch_save,
    dataset_split_directory, file_sha256, mapping_fingerprint,
)
from tools.run_grounded_binary_study import test_metrics, write_csv
from tools.run_triclass_overlap_ratio_study import ece_score


DEFAULT_CONFIG = SRC / "configs/model/TriClassPairedOverlap.yaml"
DATA_FILES = ("all_users.json", "id_mapping.json", "train_src.pkl", "train_tgt.pkl",
              "valid_cold_tgt.pkl", "test_cold_tgt.pkl", "valid_warm_tgt.pkl", "test_warm_tgt.pkl")
LOGGER = logging.getLogger("triclass-paired-overlap")


def load_experiment(path):
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    ratios = result["ratios"]
    if ratios != sorted(set(ratios)) or not 0 < min(ratios) <= max(ratios) == 1.0:
        raise ValueError("Require increasing unique ratios in (0, 1], including 1.")
    return result


def tag(ratio):
    return "r%03d" % round(ratio * 100)


def locations(experiment, task, ratio, run_dir, device="cpu"):
    config = load_config(str(SRC / experiment["reference_config"]), "Amazon2014",
                         experiment["tasks"][task]["domains"], device)
    base = Path(dataset_split_directory(config))
    if ratio != 1.0:
        config["data_path"] = str(run_dir / "data" / tag(ratio))
    view = Path(dataset_split_directory(config))
    return config, base, view, view / "grounded_triclass"


def nested_plan(num_overlap, ratio, seed, audit_fraction):
    # Keep classifier roles fixed across budgets and retain a common audit cohort.
    order = np.random.default_rng(seed).permutation(num_overlap) + 1
    a, b = int(num_overlap * 0.6), int(num_overlap * 0.2)
    groups = (order[:a], order[a:a + b], order[a + b:])
    if any(int(len(group) * ratio) < 2 for group in groups):
        raise ValueError("The budget is too small for disjoint calibration roles.")
    kept = np.sort(np.concatenate([g[:int(len(g) * ratio)] for g in groups]))
    original_codes = split_overlap_users(num_overlap, [0.6, 0.2, 0.2], seed)
    return {"kept_old_ids": kept.tolist(), "codes": original_codes[kept - 1].tolist(),
            "audit_old_ids": groups[2][:int(len(groups[2]) * audit_fraction)].tolist()}


def remap_data(all_users, mapping, frames, kept):
    n_overlap = len(all_users["overlap_users"])
    kept = np.asarray(sorted(kept), dtype=np.int64)
    if len(np.unique(kept)) != len(kept) or not np.isin(kept, np.arange(1, n_overlap + 1)).all():
        raise ValueError("Invalid retained overlap IDs.")
    removed = np.setdiff1d(np.arange(1, n_overlap + 1), kept)
    source_order = np.concatenate((kept, np.arange(n_overlap + 1, len(mapping["src"]["id2user"])), removed))
    target_order = np.concatenate((kept, np.arange(n_overlap + 1, len(mapping["tgt"]["id2user"]))))
    revised, transforms = copy.deepcopy(mapping), {}
    for domain, order in (("src", source_order), ("tgt", target_order)):
        old_names = mapping[domain]["id2user"]
        names = [old_names[0]] + [old_names[int(i)] for i in order]
        revised[domain]["id2user"] = names
        revised[domain]["user2id"] = {name: i for i, name in enumerate(names) if i}
        transform = np.zeros(len(old_names), dtype=np.int64)
        transform[order] = np.arange(1, len(order) + 1)
        transforms[domain] = transform
    users = {"overlap_users": list(range(1, len(kept) + 1))}
    for name in ("valid_cold_users", "test_cold_users", "src_only_users"):
        ids = all_users[name] + (removed.tolist() if name == "src_only_users" else [])
        users[name] = transforms["src"][ids].tolist()
    users["tgt_only_users"] = transforms["tgt"][all_users["tgt_only_users"]].tolist()
    result = {}
    for name, frame in frames.items():
        domain = "src" if name in ("train_src.pkl", "valid_cold_tgt.pkl", "test_cold_tgt.pkl") else "tgt"
        copy_frame = frame.copy()
        copy_frame["user"] = transforms[domain][frame.user.to_numpy(dtype=np.int64)]
        if domain == "tgt":
            copy_frame = copy_frame.loc[copy_frame.user > 0]
        result[name] = copy_frame.reset_index(drop=True)
    return users, revised, result, source_order


def raw_pairs(frame, mapping, user_domain):
    users = np.asarray(mapping[user_domain]["id2user"], dtype=object)[frame.user.to_numpy(dtype=np.int64)]
    return sorted(zip(users.tolist(), frame.item.tolist()))


def validate_remap(original, revised, original_mapping, revised_mapping, kept, n_overlap):
    for name in ("train_src.pkl", "valid_cold_tgt.pkl", "test_cold_tgt.pkl"):
        if raw_pairs(original[name], original_mapping, "src") != raw_pairs(revised[name], revised_mapping, "src"):
            raise ValueError("Source histories or cold evaluation changed: " + name)
    expected = original["train_tgt.pkl"]
    expected = expected.loc[(expected.user > n_overlap) | expected.user.isin(kept)]
    if raw_pairs(expected, original_mapping, "tgt") != raw_pairs(revised["train_tgt.pkl"], revised_mapping, "tgt"):
        raise ValueError("Retained target training interactions differ.")
    dropped_names = {original_mapping["src"]["id2user"][u]
                     for u in range(1, n_overlap + 1) if u not in set(kept)}
    if dropped_names.intersection(revised_mapping["tgt"]["user2id"]):
        raise ValueError("A removed overlap user's target identity is still visible.")
    for domain in ("src", "tgt"):
        if revised_mapping[domain]["item2id"] != original_mapping[domain]["item2id"]:
            raise ValueError("The item universe must be identical across budgets.")


def prepare_view(experiment, task, ratio, run_dir):
    config, base, view, cache = locations(experiment, task, ratio, run_dir)
    original_users, mapping = read_json(base / "all_users.json"), read_json(base / "id_mapping.json")
    n_overlap = len(original_users["overlap_users"])
    plan = nested_plan(n_overlap, ratio, experiment["seed"], min(experiment["ratios"]))
    hashes = {name: file_sha256(str(base / name)) for name in DATA_FILES}
    spec = {"protocol": experiment["protocol"], "task": task, "ratio": ratio,
            "original_overlap_users": n_overlap, "retained_overlap_users": len(plan["kept_old_ids"]),
            "actual_retained_ratio": len(plan["kept_old_ids"]) / n_overlap,
            "original_data": str(base), "original_data_sha256": hashes, **plan}
    marker = run_dir / "preparation" / (task + "_" + tag(ratio) + ".json")
    frames = {name: pd.read_pickle(base / name) for name in DATA_FILES if name.endswith(".pkl")}
    users, revised, changed, source_order = remap_data(original_users, mapping, frames, plan["kept_old_ids"])
    validate_remap(frames, changed, mapping, revised, plan["kept_old_ids"], n_overlap)
    spec["source_new_to_old"] = [0] + source_order.tolist()
    spec["retained_target_edges"] = len(changed["train_tgt.pkl"])
    spec["original_target_edges"] = len(frames["train_tgt.pkl"])
    spec["source_edges"] = len(changed["train_src.pkl"])
    spec["removed_overlap_users_become_source_only"] = True
    if marker.exists():
        previous = read_json(marker)
        if any(previous.get(k) != v for k, v in spec.items()):
            raise ValueError("Prepared view inputs changed; use a fresh run directory.")
        for name, digest in previous["view_sha256"].items():
            if file_sha256(str(view / name)) != digest:
                raise ValueError("Prepared view was modified: " + name)
        return previous
    if ratio != 1.0:
        view.mkdir(parents=True, exist_ok=True)
        atomic_json(str(view / "all_users.json"), users)
        atomic_json(str(view / "id_mapping.json"), revised)
        for name, frame in changed.items():
            temporary = view / (name + ".tmp")
            frame.to_pickle(temporary)
            temporary.replace(view / name)
    spec["view_sha256"] = {name: file_sha256(str(view / name)) for name in DATA_FILES}
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(str(marker), spec)
    return spec


def verify_base_cache(base):
    cache = base / "grounded_triclass"
    manifest = read_json(cache / "manifest.json")
    if manifest.get("teacher_target_users") != "target_only" or manifest.get("cold_evaluation_labels_used") is not False:
        raise ValueError("Base candidates do not satisfy the required teacher isolation.")
    for name, digest in manifest["artifacts"].items():
        if file_sha256(str(cache / name)) != digest:
            raise ValueError("Base cache artifact changed: " + name)
    for name in ("train_src", "train_tgt"):
        if file_sha256(str(base / (name + ".pkl"))) != manifest[name + "_sha256"]:
            raise ValueError("Base training split changed.")
    if mapping_fingerprint(read_json(base / "id_mapping.json")) != manifest["mapping_sha256"]:
        raise ValueError("Base mapping changed.")
    return manifest


def read_all_candidates(cache, order):
    pieces = []
    for scope in ("overlap", "cold"):
        with np.load(cache / (scope + "_candidates.npz"), allow_pickle=False) as data:
            pieces.append({k: data[k] for k in ("user_ids", "item_ids", "features", "sources")})
    full = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    if not np.array_equal(full["user_ids"], np.arange(1, len(order) + 1)):
        raise ValueError("Base candidate rows are not aligned with source users.")
    indices = np.asarray(order) - 1
    return tuple(full[k][indices].copy() for k in ("item_ids", "features", "sources"))


def confidence_report(cache, audit_old_ids, kept, output):
    with np.load(cache / "overlap_support.npz", allow_pickle=False) as data:
        items, p, labels = data["item_ids"], data["probabilities"], data["labels"]
        codes = data["split_codes"]
    audit = np.isin(kept, audit_old_ids)
    if int(audit.sum()) != len(audit_old_ids) or not (codes[audit] == 2).all():
        raise ValueError("The fixed confidence audit cohort leaked or changed.")
    valid = audit[:, None] & (items > 0) & (labels >= 0)
    y, prob = labels[valid], p[valid].astype(np.float64)
    manifest = read_json(cache / "manifest.json")
    result = {"audit_users": len(audit_old_ids), "training_users": int((codes == 0).sum()),
              "validation_users": int((codes == 1).sum()), "test_users": int((codes == 2).sum()),
              "hit_auprc": float(average_precision_score(y == 0, prob[:, 0])),
              "supported_auprc": float(average_precision_score(y < 2, prob[:, 0] + prob[:, 1])),
              "brier": float(((prob - np.eye(3)[y]) ** 2).sum(1).mean()),
              "ece15": ece_score(prob, y), "selection": {}}
    for eta, setting in manifest["selection"].items():
        result["selection"][eta] = support_report(items[audit], p[audit], labels[audit],
            manifest["top_k"], float(eta), setting["threshold"])
    atomic_json(str(output), result)
    return result


def build_budget(experiment, task, ratio, run_dir):
    config, base, view, cache = locations(experiment, task, ratio, run_dir, "auto")
    torch.set_num_threads(experiment["num_threads"])
    spec = prepare_view(experiment, task, ratio, run_dir)
    original = verify_base_cache(base)
    metrics = run_dir / ("confidence_" + task + "_" + tag(ratio) + ".json")
    if ratio == 1.0:
        return confidence_report(cache, spec["audit_old_ids"], spec["kept_old_ids"], metrics)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {"data_spec": spec, "classifier_config": config["calibrator"],
                    "selection_config": config["selection"], "seed": experiment["seed"],
                    "code_sha256": file_sha256(str(Path(__file__).resolve())),
                    "base_manifest_sha256": file_sha256(str(base / "grounded_triclass/manifest.json"))}
        identity["sha256"] = digest_json(identity)
        if (cache / "build_spec.json").exists() and read_json(cache / "build_spec.json") != identity:
            raise ValueError("Budget cache inputs changed; use a fresh run directory.")
        atomic_json(str(cache / "build_spec.json"), identity)
        if (cache / "manifest.json").exists():
            for name, digest in read_json(cache / "manifest.json")["artifacts"].items():
                if file_sha256(str(cache / name)) != digest:
                    raise ValueError("Completed budget cache changed: " + name)
            return confidence_report(cache, spec["audit_old_ids"], spec["kept_old_ids"], metrics)
        old_cache = base / "grounded_triclass"
        old_order = np.asarray(spec["source_new_to_old"][1:])
        kept = np.asarray(spec["kept_old_ids"])
        codes = np.asarray(spec["codes"], dtype=np.int8)
        n = len(kept)
        candidate_path = cache / "all_candidates.npz"
        if candidate_path.exists():
            with np.load(candidate_path, allow_pickle=False) as data:
                items, features, sources = data["item_ids"], data["features"], data["sources"]
        else:
            items, features, sources = read_all_candidates(old_cache, old_order)
            source_dir = base / "grounded_static_confidence"
            base_spec = read_json(old_cache / "build_spec.json")
            source_hashes = base_spec.get("reused_source_teacher_sha256")
            if source_hashes is None:
                source_dir = old_cache / "source_teacher"
                source_hashes = {"source_lightgcn_user_embeddings.npy": original["artifacts"]["source_teacher/source_lightgcn_user_embeddings.npy"]}
            path = source_dir / "source_lightgcn_user_embeddings.npy"
            if file_sha256(str(path)) != source_hashes["source_lightgcn_user_embeddings.npy"]:
                raise ValueError("Source teacher fingerprint changed.")
            source_users = np.load(path, mmap_mode="r", allow_pickle=False)
            target = pd.read_pickle(view / "train_tgt.pkl")
            target = target.loc[target.user <= n].copy()
            target["user"] = kept[target.user.to_numpy() - 1]
            observed = histories(target)
            references = kept[codes == 0]
            # Only this feature used overlap labels; overwrite it for every query.
            features[:, :, 2] = 0
            for start in range(0, len(old_order), 128):
                end = min(start + 128, len(old_order))
                features[start:end, :, 2] = exact_overlap_votes(old_order[start:end], items[start:end],
                    source_users, references, observed, config["candidates"]["neighbor_count"],
                    config["candidates"]["neighbor_temperature"])
                if start % 4096 == 0:
                    LOGGER.info("Rebuilding restricted votes %s %s: %d/%d references=%d",
                                task, tag(ratio), end, len(old_order), len(references))
            atomic_npz(str(candidate_path), user_ids=np.arange(1, len(old_order) + 1),
                       item_ids=items, features=features, sources=sources)
        target = pd.read_pickle(view / "train_tgt.pkl")
        observed = histories(target.loc[target.user <= n])
        neighbors = np.load(old_cache / "target_item_neighbors.npy", mmap_mode="r")
        labels = support_labels(np.arange(1, n + 1), items[:n], observed, neighbors)
        model, checkpoint = fit_calibrator(features[:n], labels, codes, config["calibrator"],
                                          experiment["seed"], config["device"])
        atomic_torch_save(str(cache / "calibrator.pt"), checkpoint)
        probabilities = predict_probabilities(model, features, items > 0, config["device"])
        selection = choose_thresholds(items[:n], probabilities[:n], labels, codes, config)
        atomic_npz(str(cache / "overlap_support.npz"), user_ids=np.arange(1, n + 1),
                   item_ids=items[:n], probabilities=probabilities[:n], labels=labels, split_codes=codes)
        atomic_npz(str(cache / "cold_support.npz"), user_ids=np.arange(n + 1, len(old_order) + 1),
                   item_ids=items[n:], probabilities=probabilities[n:])
        names = ("all_candidates.npz", "calibrator.pt", "overlap_support.npz", "cold_support.npz")
        mapping = read_json(view / "id_mapping.json")
        manifest = {"format_version": VERSION, "dataset": "Amazon2014", "domains": config["domains"],
                    "mapping_sha256": mapping_fingerprint(mapping), "num_users_overlap": n,
                    "num_users_src": len(old_order), "num_items_tgt": len(mapping["tgt"]["item2id"]),
                    "class_names": CLASS_NAMES, "feature_names": FEATURE_NAMES,
                    "teacher_target_users": "target_only", "cold_evaluation_labels_used": False,
                    "overlap_vote_users": "calibrator_train_excluding_query", "selection": selection,
                    "top_k": config["selection"]["top_k"], "protocol": experiment["protocol"],
                    "frozen_label_independent_candidates_from": str(old_cache),
                    "build_sha256": identity["sha256"],
                    "train_src_sha256": file_sha256(str(view / "train_src.pkl")),
                    "train_tgt_sha256": file_sha256(str(view / "train_tgt.pkl")),
                    "artifacts": {name: file_sha256(str(cache / name)) for name in names}}
        atomic_json(str(cache / "manifest.json"), manifest)
        return confidence_report(cache, spec["audit_old_ids"], kept, metrics)


def training_config(config, experiment, task, ratio, variant, run_dir):
    apply_model_domain_profile(config)
    data_config, _, view, cache = locations(experiment, task, ratio, run_dir)
    weight, eta = experiment["tasks"][task]["backbones"][config["model"].replace("_Grounded_TriClass_Pseudo", "")]
    config["domain_profiles"] = {}
    config["data_path"] = data_config["data_path"]
    config["seed"] = [experiment["seed"]]
    config["triclass_cache_path"] = str(cache)
    config["pseudo_loss_weight"] = [float(weight) if variant == "triclass" else 0.0]
    config["near_support_weight"] = [float(eta)]
    config["pseudo_sampler_seed"] = experiment["seed"]
    config["pseudo_start_epoch"] = 0
    config["hyper_parameters"] = ["seed", "pseudo_loss_weight", "near_support_weight"]
    config["log_model_suffix"] = "paired_overlap_" + tag(ratio) + "_" + variant
    config["log_dir"] = str(run_dir / "benchmark_logs")
    config["reuse_pretrained_stage_cache"] = False
    config["save_recommended_topk"] = False
    return config


def require_prepared(config):
    view = Path(dataset_split_directory(config))
    if not all((view / name).is_file() for name in DATA_FILES):
        raise ValueError("This study only accepts a fully prepared immutable dataset view.")


def train(experiment, task, ratio, model, variant, run_dir):
    import utils.quick_start as quick
    from utils.configurator import Config

    class PairedConfig(Config):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            training_config(self, experiment, task, ratio, variant, run_dir)

    prepare_view(experiment, task, ratio, run_dir)
    # Both arms run the identical model class; Pure only disables the extra loss.
    with patch.object(quick, "Config", PairedConfig), patch.object(quick, "check_and_prepare_dataset", require_prepared):
        quick.quick_start(model + "_Grounded_TriClass_Pseudo", "Amazon2014",
                          experiment["tasks"][task]["domains"], save_model=False)


def make_jobs(experiment, config_path, run_dir):
    jobs = []
    for ratio in experiment["ratios"]:
        for task in experiment["tasks"]:
            jobs.append(dict(id=f"build_{task}_{tag(ratio)}", kind="build", task=task, ratio=ratio,
                             dependency=None, status="waiting"))
    for model in next(iter(experiment["tasks"].values()))["backbones"]:
        for ratio in experiment["ratios"]:
            for task in experiment["tasks"]:
                for variant in ("pure", "triclass"):
                    jobs.append(dict(id=f"{model}_{task}_{tag(ratio)}_{variant}", kind="train",
                        model=model, task=task, ratio=ratio, variant=variant, status="waiting",
                        dependency=f"build_{task}_{tag(ratio)}"))
    for job in jobs:
        job["command"] = [sys.executable, "-u", str(Path(__file__).resolve()), job["kind"],
            "--config", str(config_path), "--run-dir", str(run_dir), "--task", job["task"],
            "--ratio", str(job["ratio"])]
        if job["kind"] == "train":
            job["command"] += ["--model", job["model"], "--variant", job["variant"]]
    return jobs


def summarize(directory, state):
    rows, quality = [], []
    for job in state["jobs"]:
        if job["kind"] == "build":
            path = directory / ("confidence_" + job["task"] + "_" + tag(job["ratio"]) + ".json")
            if path.exists() and job["status"] == "complete":
                record = read_json(path)
                quality.append(dict(task=job["task"], ratio=job["ratio"],
                    **{k: v for k, v in record.items() if k != "selection"}))
            continue
        metrics = None
        if job["status"] == "complete":
            content = Path(job["log"]).read_text(encoding="utf-8", errors="replace")
            if "Best Parameters:" in content:
                metrics = test_metrics(content.rsplit("Best Parameters:", 1)[1])
            if metrics is None:
                job.update(status="failed", reason="No final recommendation metrics.")
        rows.append(dict(task=job["task"], model=job["model"], ratio=job["ratio"], variant=job["variant"],
            recall20=metrics[0] if metrics else "", ndcg20=metrics[1] if metrics else "",
            status=job["status"], log=job.get("log", "")))
    write_csv(directory / "performance.csv", list(rows[0]), rows)
    if quality:
        write_csv(directory / "confidence.csv", list(quality[0]), quality)
    paired = []
    for row in rows:
        if row["variant"] != "pure":
            continue
        tri = next(r for r in rows if r["variant"] == "triclass" and
                   all(r[k] == row[k] for k in ("task", "model", "ratio")))
        done = row["status"] == tri["status"] == "complete"
        paired.append(dict(task=row["task"], model=row["model"], ratio=row["ratio"],
            pure_recall20=row["recall20"], pure_ndcg20=row["ndcg20"],
            triclass_recall20=tri["recall20"], triclass_ndcg20=tri["ndcg20"],
            delta_recall20=tri["recall20"] - row["recall20"] if done else "",
            delta_ndcg20=tri["ndcg20"] - row["ndcg20"] if done else "",
            pure_status=row["status"], triclass_status=tri["status"]))
    write_csv(directory / "paired_results.csv", list(paired[0]), paired)


def coordinate(experiment, config_path, run_dir, gpus, max_per_gpu):
    import tools.run_grounded_triclass_queue as queue
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "coordinator.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    protocol = {"experiment": experiment, "api_calls": False, "fresh_training_pairs": 40,
        "fixed_original_cold_evaluation": True, "removed_target_histories_visible": False,
        "frozen_target_only_teachers_and_label_independent_candidates": True,
        "rebuild_overlap_votes_and_classifier": True, "same_backbone_code_for_both_arms": True,
        "runner_sha256": file_sha256(str(Path(__file__).resolve())),
        "config_sha256": file_sha256(str(config_path))}

    def save_state(path, state):
        state["protocol"] = protocol
        summarize(run_dir, state)
        atomic_json(str(path), state)

    with patch.object(queue, "jobs", lambda: make_jobs(experiment, config_path, run_dir)), \
            patch.object(queue, "atomic_json", save_state):
        return queue.run(run_dir, gpus, max_per_gpu, 15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "build", "train", "queue", "summarize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("ClothSport", "PhoneElec"))
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--model", choices=("EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB"))
    parser.add_argument("--variant", choices=("pure", "triclass"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-per-gpu", type=int, default=2)
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    run_dir = args.run_dir.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "queue":
        return coordinate(experiment, args.config.resolve(), run_dir, args.gpus, args.max_per_gpu)
    if args.mode == "summarize":
        summarize(run_dir, read_json(run_dir / "status.json"))
        return 0
    if args.task is None or args.ratio not in experiment["ratios"]:
        parser.error("A configured task and ratio are required.")
    if args.mode == "prepare":
        spec = prepare_view(experiment, args.task, args.ratio, run_dir)
        LOGGER.info("Prepared %s %s: overlap=%d target_edges=%d", args.task, tag(args.ratio),
                    spec["retained_overlap_users"], spec["retained_target_edges"])
    elif args.mode == "build":
        build_budget(experiment, args.task, args.ratio, run_dir)
    else:
        if args.model is None or args.variant is None:
            parser.error("Training requires --model and --variant.")
        train(experiment, args.task, args.ratio, args.model, args.variant, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
