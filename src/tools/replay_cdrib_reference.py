"""Replay one historical CDRIB configuration without changing its operators."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import numpy as np
import torch
from common.trainer import Trainer
from models.target_pattern_pseudo import atomic_json, atomic_torch_save, file_sha256
from tools.run_triclass_paired_overlap import DATA_FILES, require_prepared
from utils.configurator import Config
import utils.quick_start as quick

REFERENCES = {
    "pure": ("CDRIB_Grounded_Backbone", SRC / "runGroundedBackboneClothSport_4_CDRIB.log",
             [0.0198, 0.0167]),
    "triclass": ("CDRIB_Grounded_TriClass_Pseudo", SRC / "triclass_runs/20260908/CDRIB_ClothSport.log",
                 [0.0256, 0.0223]),
}
SPLIT = (SRC.parent / "data/Amazon2014/Clothing_Shoes_and_Jewelry+Sports_and_Outdoors/"
         "all_users/WarmValid0_WarmTest0_ColdValid0.1_ColdTest0.1_shuffle")


def logged_config(path):
    result = {}
    for line in path.read_text().split("====Training====")[0].splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)=(.*)", line)
        if not match:
            continue
        key, value = match.groups()
        if key == "device" or any(x in key.lower() for x in ("api_key", "password", "secret", "token")):
            continue
        try:
            result[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            result[key] = value
    if result.get("seed") != [999] or "training_stages" not in result:
        raise ValueError("Historical log does not contain the expected configuration.")
    return result


def weight_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    os.chdir(SRC)
    # Retain the legacy PyTorch backend defaults, including its sparse operator.
    if torch.are_deterministic_algorithms_enabled() or os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError("Legacy replay must run in a fresh process without deterministic overrides.")
    args.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (args.run_dir / "provenance.json").exists():
        raise RuntimeError("Use a new directory; previous replay outputs are immutable.")
    model_name, original_log, expected = REFERENCES[args.variant]
    historical = logged_config(original_log)
    input_hashes = {str(SPLIT / name): file_sha256(str(SPLIT / name)) for name in DATA_FILES}
    if args.variant == "triclass":
        cache = SPLIT / "grounded_triclass"
        manifest = json.loads((cache / "manifest.json").read_text())
        for name in ("manifest.json", "cold_support.npz"):
            value = file_sha256(str(cache / name))
            if name in manifest["artifacts"] and value != manifest["artifacts"][name]:
                raise RuntimeError("Pseudo cache integrity check failed.")
            input_hashes[str(cache / name)] = value
    provenance = dict(variant=args.variant, model=model_name, reference_log=str(original_log),
        reference_log_sha256=file_sha256(str(original_log)), reference_recall_ndcg20=expected,
        input_sha256=input_hashes, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), num_threads=torch.get_num_threads(),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        note="Single selected legacy configuration; historical runtime scheduling is not recorded.")
    source_paths = list((SRC / "common").rglob("*.py")) + list((SRC / "utils").rglob("*.py"))
    source_paths += [SRC / "models" / name for name in (
        "cdrib.py", "cdrib_grounded_backbone.py", "cdrib_grounded_triclass_pseudo.py",
        "grounded_triclass_pseudo.py", "grounded_static_confidence.py", "target_pattern_pseudo.py",
        "direct_semantic_pseudo.py")]
    source_paths.append(Path(__file__))
    provenance["source_sha256"] = {str(p): file_sha256(str(p)) for p in source_paths}
    atomic_json(str(args.run_dir / "provenance.json"), provenance)

    class ReplayConfig(Config):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.final_config_dict.update(historical)
            self["seed"] = [999]
            self["log_dir"] = str(args.run_dir / "benchmark_logs")
            self["log_model_suffix"] = "legacy_reference_replay"
            if args.variant == "triclass":
                self["pseudo_loss_weight"] = [0.1]
                self["near_support_weight"] = [0.25]
            atomic_json(str(args.run_dir / "effective_config.json"),
                        json.loads(json.dumps(self.final_config_dict, default=str)))

    class ReplayTrainer(Trainer):
        def fit(self, stage_id, train_data, valid_data=None, **kwargs):
            self.replay_valid_data = valid_data
            self.replay_epoch = -1
            self.replay_best_epoch = None
            atomic_json(str(args.run_dir / "initial.json"), dict(weights_sha256=weight_hash(self.model)))
            result = super().fit(stage_id, train_data, valid_data=valid_data, **kwargs)
            metrics = result[0][5]
            observed = [metrics["recall@20"], metrics["ndcg@20"]]
            changed = [name for name, digest in {**input_hashes, **provenance["source_sha256"]}.items()
                       if file_sha256(name) != digest]
            if changed:
                raise RuntimeError("Inputs or source changed during replay: " + str(changed))
            atomic_json(str(args.run_dir / "result.json"), dict(
                validation_and_test=result[0], stop_epoch=result[1][1], best_epoch=self.replay_best_epoch,
                observed_recall_ndcg20=observed, reference_recall_ndcg20=expected,
                matches_historical_report=observed == expected,
                final_weights_sha256=weight_hash(self.model), changed_inputs_or_sources=changed))
            return result

        def _train_epoch(self, train_data, epoch_idx, *positional, **keywords):
            self.replay_epoch = epoch_idx
            return super()._train_epoch(train_data, epoch_idx, *positional, **keywords)

        def _valid_epoch(self, data):
            result = super()._valid_epoch(data)
            if data is self.replay_valid_data and result[2] > self.best_valid_score_cold:
                self.replay_best_epoch = self.replay_epoch
                checkpoint = dict(model_state_dict={k: v.detach().cpu().clone()
                                                    for k, v in self.model.state_dict().items()},
                    optimizer=self.optimizer.state_dict(), scheduler=self.lr_scheduler.state_dict(),
                    epoch=self.replay_epoch, validation_metrics=result[3],
                    config=json.loads(json.dumps(self.config.final_config_dict, default=str)),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
                supervisor = getattr(self.model, "pseudo_supervisor", None)
                if supervisor is not None:
                    checkpoint["pseudo_rng"] = supervisor.rng.bit_generator.state
                atomic_torch_save(str(args.run_dir / "best_validation.pt"), checkpoint)
                with (args.run_dir / "validation_updates.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(epoch=self.replay_epoch, score=result[2],
                                                 weights_sha256=weight_hash(self.model))) + "\n")
            return result

    with patch.object(quick, "Config", ReplayConfig), \
            patch.object(quick, "get_trainer", lambda: ReplayTrainer), \
            patch.object(quick, "check_and_prepare_dataset", require_prepared):
        quick.quick_start(model_name, "Amazon2014", historical["domains"], save_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(REFERENCES))
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.run_dir = arguments.run_dir.resolve()
    run(arguments)
