"""Leave-one-evidence-out calibration with immutable TriClass candidates."""

import copy
import fcntl
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from models.grounded_static_confidence import configure_logging, load_config
from models.grounded_triclass_pseudo import (
    CLASS_NAMES, FEATURE_NAMES, VERSION, choose_thresholds, digest_json,
    fit_calibrator, predict_probabilities, read_json, select_support, support_report,
)
from models.target_pattern_pseudo import (
    atomic_json, atomic_npz, atomic_torch_save, dataset_split_directory, file_sha256,
)


SRC = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SRC / "configs/model/GroundedTriClassEvidence.yaml"
VARIANTS = {"wo_semantic": 0, "wo_g": 1, "wo_overlap_vote": 2, "wo_popularity": 3}
LOGGER = logging.getLogger("triclass-evidence")


def load_experiment(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if set(result["variants"]) != set(VARIANTS):
        raise ValueError("Require all four predeclared evidence ablations.")
    return result


def locations(experiment, task, run_dir, device="cpu"):
    config = load_config(str(SRC / experiment["reference_config"]), "Amazon2014",
                         experiment["tasks"][task]["domains"], device)
    config["data_path"] = str(Path(experiment["data_path"]).resolve())
    base = Path(dataset_split_directory(config)) / config["output_subdir"]
    override = experiment.get("source_caches", {}).get(task)
    return config, Path(override).resolve() if override else base, Path(run_dir) / "cache" / task


def mask_evidence(features, variant):
    if variant not in VARIANTS or features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("Unknown ablation or invalid feature shape.")
    result = np.array(features, dtype=np.float32, copy=True)
    # Zero before train-only normalization, in both fitting and prediction. The
    # zero-initialized linear column stays zero: three effective input features.
    result[..., VARIANTS[variant]] = 0
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite retained evidence.")
    return result


def confidence_metrics(probabilities, labels):
    p = probabilities.astype(np.float64)
    confidence = p.max(-1)
    correct = p.argmax(-1) == labels
    ece = 0.0
    for index in range(15):
        mask = (confidence >= index / 15) & (confidence < (index + 1) / 15)
        if index == 14:
            mask = (confidence >= index / 15) & (confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {"hit_auprc": float(average_precision_score(labels == 0, p[:, 0])),
            "supported_auprc": float(average_precision_score(labels < 2, p[:, 0] + p[:, 1])),
            "brier": float(((p - np.eye(3)[labels]) ** 2).sum(-1).mean()),
            "ece15": float(ece)}


def build(experiment, task, variant, run_dir, device="auto"):
    if variant not in VARIANTS:
        raise ValueError("Unknown evidence ablation.")
    config, base, parent = locations(experiment, task, run_dir, device)
    output = parent / variant
    if output.resolve() == base.resolve():
        raise ValueError("Do not overwrite the original TriClass cache.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(experiment["num_threads"]))
    configure_logging(str(output))
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, spec = read_json(base / "manifest.json"), read_json(base / "build_spec.json")
        if (manifest["domains"] != config["domains"] or manifest["class_names"] != CLASS_NAMES
                or manifest["feature_names"] != FEATURE_NAMES
                or manifest["build_sha256"] != spec["build_sha256"]
                or manifest.get("cold_evaluation_labels_used") is not False
                or manifest.get("teacher_target_users") != "target_only"):
            raise ValueError("Source TriClass cache identity or isolation mismatch.")
        inputs = ("overlap_candidates.npz", "cold_candidates.npz", "overlap_support.npz")
        hashes = {name: file_sha256(str(base / name)) for name in inputs}
        if any(manifest["artifacts"].get(name) != value for name, value in hashes.items()):
            raise ValueError("Source candidate/label checksum mismatch.")
        if int(spec["config"]["seed"]) != int(experiment["seed"]):
            raise ValueError("Classifier seed must match the full TriClass reference.")
        settings = copy.deepcopy(spec["config"]["calibrator"])
        selection_config = {"selection": copy.deepcopy(spec["config"]["selection"])}
        identity = {"variant": variant, "removed_feature": FEATURE_NAMES[VARIANTS[variant]],
                    "source_manifest_sha256": file_sha256(str(base / "manifest.json")),
                    "source_files": hashes, "calibrator": settings, "selection": selection_config,
                    "seed": experiment["seed"], "builder_sha256": file_sha256(__file__),
                    "shared_calibrator_sha256": file_sha256(str(SRC / "models/grounded_triclass_pseudo.py"))}
        identity["build_sha256"] = digest_json(identity)
        if (output / "build_spec.json").exists() and read_json(output / "build_spec.json") != identity:
            raise ValueError("Build inputs changed; use a new run directory.")
        atomic_json(str(output / "build_spec.json"), identity)
        if (output / "manifest.json").exists():
            completed = read_json(output / "manifest.json")
            if completed["build_sha256"] != identity["build_sha256"]:
                raise ValueError("Completed build identity mismatch.")
            for name, digest in completed["artifacts"].items():
                if file_sha256(str(output / name)) != digest:
                    raise ValueError("Completed artifact changed: " + name)
            LOGGER.info("Reusing completed evidence cache %s", output)
            return output
        with np.load(base / "overlap_candidates.npz", allow_pickle=False) as data:
            users, items = data["user_ids"], data["item_ids"]
            features = mask_evidence(data["features"], variant)
        with np.load(base / "overlap_support.npz", allow_pickle=False) as data:
            if not np.array_equal(users, data["user_ids"]) or not np.array_equal(items, data["item_ids"]):
                raise ValueError("Candidates and labels are not aligned.")
            labels, codes = data["labels"], data["split_codes"]
        LOGGER.info("Fitting task=%s variant=%s removed=%s users=%d", task, variant,
                    identity["removed_feature"], len(users))
        model, checkpoint = fit_calibrator(features, labels, codes, settings,
                                           int(experiment["seed"]), config["device"])
        removed = VARIANTS[variant]
        if torch.count_nonzero(model.linear.weight[:, removed]).item() or model.mean[removed].item():
            raise ValueError("Removed evidence still affects the trained classifier.")
        checkpoint.update(variant=variant, removed_feature=identity["removed_feature"],
                          effective_feature_names=[n for i, n in enumerate(FEATURE_NAMES) if i != removed])
        atomic_torch_save(str(output / "calibrator.pt"), checkpoint)
        probabilities = predict_probabilities(model, features, items > 0, config["device"])
        selection = choose_thresholds(items, probabilities, labels, codes, selection_config)
        top_k = int(selection_config["selection"]["top_k"])
        diagnostics = {"variant": variant, "removed_feature": identity["removed_feature"],
                       "best_epoch": checkpoint["best_epoch"], "validation_ce": checkpoint["validation_ce"],
                       "selection": selection, "splits": {}, "cold_selection": {}}
        for code, name in enumerate(("train", "validation", "test")):
            rows = codes == code
            valid = rows[:, None] & (labels >= 0)
            report = confidence_metrics(probabilities[valid], labels[valid])
            report["users"] = int(rows.sum())
            report["selection"] = {eta: support_report(items[rows], probabilities[rows], labels[rows],
                top_k, float(eta), value["threshold"]) for eta, value in selection.items()}
            diagnostics["splits"][name] = report
        atomic_npz(str(output / "overlap_support.npz"), user_ids=users, item_ids=items,
                   probabilities=probabilities, labels=labels, split_codes=codes)
        del features
        with np.load(base / "cold_candidates.npz", allow_pickle=False) as data:
            cold_users, cold_items = data["user_ids"], data["item_ids"]
            cold_features = mask_evidence(data["features"], variant)
        cold_p = predict_probabilities(model, cold_features, cold_items > 0, config["device"])
        atomic_npz(str(output / "cold_support.npz"), user_ids=cold_users, item_ids=cold_items,
                   probabilities=cold_p)
        artifacts = ["calibrator.pt", "overlap_support.npz", "cold_support.npz", "diagnostics.json"]
        for eta, setting in selection.items():
            selected, weights = select_support(cold_items, cold_p, top_k, float(eta), setting["threshold"])
            name = "pseudo_rankings_eta" + eta + ".npz"
            atomic_npz(str(output / name), user_ids=cold_users, item_ids=selected, scores=weights)
            artifacts.append(name)
            diagnostics["cold_selection"][eta] = {"pairs": int((selected > 0).sum()),
                "user_coverage": float((selected > 0).any(1).mean()), "mean_slot_weight": float(weights.mean())}
        atomic_json(str(output / "diagnostics.json"), diagnostics)
        result = {key: manifest[key] for key in ("dataset", "domains", "mapping_sha256", "num_users_overlap",
            "num_users_src", "num_items_tgt", "train_src_sha256", "train_tgt_sha256",
            "teacher_target_users", "overlap_vote_users")}
        result.update(format_version=VERSION, class_names=CLASS_NAMES, feature_names=FEATURE_NAMES,
            variant=variant, removed_feature=identity["removed_feature"], classifier_output_dim=3,
            effective_feature_names=checkpoint["effective_feature_names"], top_k=top_k, selection=selection,
            cold_evaluation_labels_used=False, build_sha256=identity["build_sha256"],
            candidate_routes_unchanged=True, source_files=hashes,
            artifacts={name: file_sha256(str(output / name)) for name in artifacts})
        atomic_json(str(output / "manifest.json"), result)
        LOGGER.info("Completed %s; test=%s", output, diagnostics["splits"]["test"])
        return output
