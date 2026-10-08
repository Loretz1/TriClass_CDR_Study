"""Binary label ablations using immutable TriClass candidates and evidence."""

import argparse
import copy
import fcntl
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
import yaml

from models.grounded_static_confidence import atomic_npz, configure_logging, load_config
from models.grounded_triclass_pseudo import (
    CLASS_NAMES, FEATURE_NAMES, VERSION, choose_thresholds, digest_json, read_json,
    select_support, support_report,
)
from models.target_pattern_pseudo import (
    atomic_json, atomic_torch_save, dataset_split_directory, file_sha256, set_seed,
)


SRC = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("grounded-binary")
DEFAULT_CONFIG = SRC / "configs/model/GroundedBinary.yaml"
VARIANTS = {
    "hit_only": ["hit", "near_or_unsupported"],
    "hit_near": ["hit_or_near", "unsupported"],
}


def load_experiment(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if set(config["variants"]) != set(VARIANTS):
        raise ValueError("Both predeclared binary variants are required.")
    return config


def source_config(experiment, task, device="cpu"):
    return load_config(str(SRC / experiment["reference_config"]), "Amazon2014",
                       experiment["tasks"][task]["domains"], device)


def cache_directory(experiment, task, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown binary variant: " + variant)
    config = source_config(experiment, task)
    return Path(dataset_split_directory(config)) / (experiment["output_prefix"] + "_" + variant)


def binary_labels(labels, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown binary variant: " + variant)
    labels = np.asarray(labels)
    if not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError("Expected original Hit/Near/Unsupported labels and -1 padding.")
    positive = labels == 0 if variant == "hit_only" else (labels == 0) | (labels == 1)
    return np.where(labels < 0, -1, np.where(positive, 0, 1)).astype(np.int8)


class BinaryCalibrator(nn.Module):
    """The same four inputs and train-only normalization, with two outputs."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("scale", torch.ones(4))
        self.linear = nn.Linear(4, 2)

    def forward(self, features):
        return self.linear((features - self.mean) / self.scale)


def fit_calibrator(features, labels, split_codes, settings, seed, device):
    train = (split_codes[:, None] == 0) & (labels >= 0)
    valid = (split_codes[:, None] == 1) & (labels >= 0)
    tx = torch.as_tensor(features[train], dtype=torch.float32, device=device)
    ty = torch.as_tensor(labels[train], dtype=torch.long, device=device)
    vx = torch.as_tensor(features[valid], dtype=torch.float32, device=device)
    vy = torch.as_tensor(labels[valid], dtype=torch.long, device=device)
    counts = torch.bincount(ty, minlength=2)
    if len(counts) != 2 or torch.any(counts == 0) or not len(vy):
        raise ValueError("Both training classes and nonempty validation are required.")
    set_seed(seed)
    model = BinaryCalibrator().to(device)
    with torch.no_grad():
        model.mean.copy_(tx.mean(0))
        model.scale.copy_(tx.std(0, unbiased=False).clamp_min(1e-6))
        model.linear.weight.zero_()
        model.linear.bias.copy_((counts.float() / counts.sum()).log())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]),
                                 weight_decay=float(settings["weight_decay"]))
    rng = np.random.default_rng(seed)
    best, stale, state, best_epoch = float("inf"), 0, None, 0
    batch_size = int(settings["batch_size"])
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        order = rng.permutation(len(ty))
        for start in range(0, len(ty), batch_size):
            indices = torch.as_tensor(order[start:start + batch_size], device=device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(model(vx), vy).cpu())
        if value < best - float(settings["minimum_delta"]):
            best, stale, best_epoch = value, 0, epoch + 1
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info("Binary calibrator epoch=%d validation_ce=%.6f best=%d",
                        epoch + 1, value, best_epoch)
        if stale >= int(settings["patience"]):
            break
    if state is None:
        raise ValueError("Binary calibration failed to produce a finite validation loss.")
    model.load_state_dict(state)
    return model, {"state_dict": state, "best_epoch": best_epoch, "validation_ce": best,
                   "training_class_counts": counts.cpu().tolist(), "feature_names": FEATURE_NAMES}


def predict_probabilities(model, features, valid, device, batch_size=65536):
    flat = features.reshape(-1, 4)
    result = np.zeros((len(flat), 2), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            x = torch.as_tensor(flat[start:start + batch_size], device=device)
            result[start:start + batch_size] = model(x).softmax(-1).cpu().numpy()
    result = result.reshape(features.shape[:-1] + (2,))
    result[~valid] = 0
    return result


def sampler_probabilities(probabilities):
    # Adapt binary scores to the existing sampler without changing backbone code.
    # These are [binary_positive, zero, binary_negative], not three learned classes.
    probabilities = np.asarray(probabilities)
    if probabilities.shape[-1] != 2:
        raise ValueError("Expected two learned probabilities.")
    return np.stack((probabilities[..., 0], np.zeros_like(probabilities[..., 0]),
                     probabilities[..., 1]), axis=-1)


def choose_binary_threshold(items, probabilities, labels, codes, settings):
    selection = copy.deepcopy(settings)
    selection["eta_values"] = [0.0]
    # Both controls maximize validation precision of their own positive class.
    proxy_labels = np.where(labels == 1, 2, labels)
    return choose_thresholds(items, sampler_probabilities(probabilities), proxy_labels,
                             codes, {"selection": selection})


def confidence_metrics(probabilities, binary_y, original_y):
    score = probabilities[:, 0].astype(np.float64)
    positive = binary_y == 0
    calibration = 0.0
    for index in range(15):
        mask = (score >= index / 15) & (score < (index + 1) / 15)
        if index == 14:
            mask = (score >= index / 15) & (score <= 1)
        if mask.any():
            calibration += mask.mean() * abs(score[mask].mean() - positive[mask].mean())
    return {
        "binary_ce": float(-np.log(np.clip(probabilities[np.arange(len(binary_y)), binary_y], 1e-12, 1)).mean()),
        "positive_brier": float(np.square(score - positive).mean()),
        "positive_ece15": float(calibration),
        "positive_auprc": float(average_precision_score(positive, score)),
        "hit_auprc_using_supervision_score": float(average_precision_score(original_y == 0, score)),
        "supported_auprc_using_supervision_score": float(average_precision_score(original_y < 2, score)),
    }


def build(experiment, task, variant, device="auto"):
    config = source_config(experiment, task, device)
    torch.set_num_threads(int(experiment["num_threads"]))
    base = Path(dataset_split_directory(config)) / config["output_subdir"]
    output = cache_directory(experiment, task, variant)
    if output == base:
        raise ValueError("Binary outputs must not overwrite the TriClass cache.")
    output.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output))
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read_json(base / "manifest.json")
        spec = read_json(base / "build_spec.json")
        if manifest["domains"] != config["domains"] or manifest["class_names"] != CLASS_NAMES:
            raise ValueError("The source cache does not match this task or class definition.")
        if manifest["build_sha256"] != spec["build_sha256"]:
            raise ValueError("Source cache build identity mismatch.")
        inputs = ("overlap_candidates.npz", "cold_candidates.npz", "overlap_support.npz")
        hashes = {name: file_sha256(str(base / name)) for name in inputs}
        for name, digest in hashes.items():
            if manifest["artifacts"][name] != digest:
                raise ValueError("Source cache checksum mismatch: " + name)
        settings = spec["config"]["calibrator"]
        selection_settings = spec["config"]["selection"]
        if int(spec["config"]["seed"]) != int(experiment["seed"]):
            raise ValueError("Use the same classifier seed as the source TriClass experiment.")
        identity = {"variant": variant, "source_manifest_sha256": file_sha256(str(base / "manifest.json")),
                    "source_files": hashes, "calibrator": settings, "selection": selection_settings,
                    "seed": experiment["seed"], "builder_sha256": file_sha256(__file__),
                    "shared_triclass_sha256": file_sha256(str(SRC / "models/grounded_triclass_pseudo.py"))}
        identity["build_sha256"] = digest_json(identity)
        destination_spec = output / "build_spec.json"
        if destination_spec.exists() and read_json(destination_spec) != identity:
            raise ValueError("Binary build inputs changed; use a new output_prefix.")
        atomic_json(str(destination_spec), identity)
        if (output / "manifest.json").exists():
            completed = read_json(output / "manifest.json")
            for name, digest in completed["artifacts"].items():
                if file_sha256(str(output / name)) != digest:
                    raise ValueError("Completed binary artifact changed: " + name)
            LOGGER.info("Reusing completed binary cache %s", output)
            return output
        with np.load(base / "overlap_candidates.npz", allow_pickle=False) as data:
            users, items, features = data["user_ids"], data["item_ids"], data["features"]
        with np.load(base / "overlap_support.npz", allow_pickle=False) as data:
            if not np.array_equal(items, data["item_ids"]) or not np.array_equal(users, data["user_ids"]):
                raise ValueError("Source labels and candidates are not aligned.")
            original_y, codes = data["labels"], data["split_codes"]
        labels = binary_labels(original_y, variant)
        LOGGER.info("Starting variant=%s classes=%s task=%s users=%d split=%s",
                    variant, VARIANTS[variant], task, len(users), np.bincount(codes).tolist())
        model, checkpoint = fit_calibrator(features, labels, codes, settings,
                                           int(experiment["seed"]), config["device"])
        checkpoint.update(class_names=VARIANTS[variant], variant=variant)
        atomic_torch_save(str(output / "calibrator.pt"), checkpoint)
        probabilities = predict_probabilities(model, features, items > 0, config["device"])
        selection = choose_binary_threshold(items, probabilities, labels, codes, selection_settings)
        compatible = sampler_probabilities(probabilities)
        threshold = selection["0.0"]["threshold"]
        top_k = int(selection_settings["top_k"])
        diagnostics = {"variant": variant, "classifier_class_names": VARIANTS[variant],
                       "best_epoch": checkpoint["best_epoch"],
                       "training_class_counts": checkpoint["training_class_counts"],
                       "selection": selection, "splits": {}}
        for code, name in enumerate(("train", "validation", "test")):
            mask = codes == code
            valid = mask[:, None] & (labels >= 0)
            report = confidence_metrics(probabilities[valid], labels[valid], original_y[valid])
            report["selection"] = support_report(items[mask], compatible[mask], original_y[mask],
                                                   top_k, 0.0, threshold)
            report["users"] = int(mask.sum())
            diagnostics["splits"][name] = report
        atomic_npz(str(output / "overlap_support.npz"), user_ids=users, item_ids=items,
                   probabilities=compatible, binary_probabilities=probabilities,
                   labels=original_y, binary_labels=labels, split_codes=codes)
        del features
        with np.load(base / "cold_candidates.npz", allow_pickle=False) as data:
            cold_users, cold_items, cold_features = data["user_ids"], data["item_ids"], data["features"]
        cold_p = predict_probabilities(model, cold_features, cold_items > 0, config["device"])
        cold_compatible = sampler_probabilities(cold_p)
        selected, weights = select_support(cold_items, cold_compatible, top_k, 0.0, threshold)
        diagnostics["cold_selection"] = {"users": len(cold_users), "pairs": int((selected > 0).sum()),
            "user_coverage": float((selected > 0).any(1).mean()), "mean_slot_weight": float(weights.mean())}
        atomic_npz(str(output / "cold_support.npz"), user_ids=cold_users, item_ids=cold_items,
                   probabilities=cold_compatible, binary_probabilities=cold_p)
        atomic_npz(str(output / "pseudo_rankings.npz"), user_ids=cold_users, item_ids=selected, scores=weights)
        atomic_json(str(output / "diagnostics.json"), diagnostics)
        result = {key: manifest[key] for key in (
            "dataset", "domains", "mapping_sha256", "num_users_overlap", "num_users_src", "num_items_tgt",
            "train_src_sha256", "train_tgt_sha256", "teacher_target_users", "overlap_vote_users")}
        files = ("calibrator.pt", "overlap_support.npz", "cold_support.npz", "pseudo_rankings.npz", "diagnostics.json")
        result.update(format_version=VERSION, class_names=CLASS_NAMES, feature_names=FEATURE_NAMES,
            classifier_class_names=VARIANTS[variant], classifier_output_dim=2, variant=variant,
            probability_layout=["binary_positive", "zero", "binary_negative"],
            sampler_near_support_weight=0.0, top_k=top_k, selection=selection,
            cold_evaluation_labels_used=False, build_sha256=identity["build_sha256"],
            source_cache=str(base), source_files=hashes,
            artifacts={name: file_sha256(str(output / name)) for name in files})
        atomic_json(str(output / "manifest.json"), result)
        LOGGER.info("Completed binary cache %s threshold=%.4g test=%s cold=%s", output, threshold,
                    diagnostics["splits"]["test"], diagnostics["cold_selection"])
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", choices=["ClothSport", "PhoneElec"], required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    build(load_experiment(args.config), args.task, args.variant, args.device)
