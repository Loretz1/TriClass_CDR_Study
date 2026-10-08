"""Verify paired-budget data isolation, cache rebuilding and model configuration."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.grounded_triclass_pseudo import CLASS_NAMES, split_overlap_users
from models.target_pattern_pseudo import atomic_json, atomic_npz, file_sha256, mapping_fingerprint
from tools.run_triclass_paired_overlap import (
    DATA_FILES, DEFAULT_CONFIG, build_budget, load_experiment, make_jobs, nested_plan,
    prepare_view, remap_data, training_config, validate_remap,
)
from utils.configurator import Config


def fixture(n=60):
    source = ["PAD"] + ["overlap%d" % i for i in range(1, n + 1)] + ["cold%d" % i for i in range(1, 5)]
    target = source[:n + 1] + ["target%d" % i for i in range(1, 6)]
    mapping = {}
    for domain, names in (("src", source), ("tgt", target)):
        items = ["PAD"] + [domain + str(i) for i in range(1, 11)]
        mapping[domain] = dict(id2user=names, user2id={u: i for i, u in enumerate(names) if i},
                               id2item=items, item2id={v: i for i, v in enumerate(items) if i})
    users = dict(overlap_users=list(range(1, n + 1)), valid_cold_users=[n + 1],
                 test_cold_users=[n + 2], src_only_users=[n + 3, n + 4],
                 tgt_only_users=list(range(n + 1, n + 6)))
    frame = lambda pairs: pd.DataFrame(pairs, columns=["user", "item"], dtype=np.int64)
    frames = {"train_src.pkl": frame([(u, i) for u in range(1, n + 5) for i in (1, 2)]),
              "train_tgt.pkl": frame([(u, i) for u in range(1, n + 6) for i in (u % 4 + 1, 5)]),
              "valid_cold_tgt.pkl": frame([(n + 1, 3)]), "test_cold_tgt.pkl": frame([(n + 2, 4)]),
              "valid_warm_tgt.pkl": frame([]), "test_warm_tgt.pkl": frame([])}
    return users, mapping, frames


class PairedOverlapTests(unittest.TestCase):
    def test_nested_budgets_and_fixed_audit(self):
        previous = set()
        original = split_overlap_users(3128, [.6, .2, .2], 999)
        for ratio in (.2, .4, .6, .8, 1.):
            plan = nested_plan(3128, ratio, 999, .2)
            kept = set(plan["kept_old_ids"])
            self.assertTrue(previous <= kept)
            self.assertTrue(set(plan["audit_old_ids"]) <= kept)
            np.testing.assert_array_equal(plan["codes"], original[np.asarray(plan["kept_old_ids"]) - 1])
            self.assertTrue((original[np.asarray(plan["audit_old_ids"]) - 1] == 2).all())
            self.assertEqual(plan, nested_plan(3128, ratio, 999, .2))
            previous = kept
        self.assertEqual(len(previous), 3128)

    def test_target_removal_keeps_source_and_evaluation(self):
        users, mapping, frames = fixture()
        kept = nested_plan(60, .4, 999, .2)["kept_old_ids"]
        new_users, new_map, new_frames, order = remap_data(users, mapping, frames, kept)
        validate_remap(frames, new_frames, mapping, new_map, kept, 60)
        self.assertEqual(len(new_frames["train_src.pkl"]), len(frames["train_src.pkl"]))
        self.assertLess(len(new_frames["train_tgt.pkl"]), len(frames["train_tgt.pkl"]))
        self.assertEqual(len(new_users["overlap_users"]), len(kept))
        self.assertEqual(len(new_users["src_only_users"]), 2 + 60 - len(kept))
        self.assertEqual(len(order), len(mapping["src"]["id2user"]) - 1)
        full_users, full_map, full_frames, _ = remap_data(users, mapping, frames, list(range(1, 61)))
        self.assertEqual(full_users, users)
        self.assertEqual(full_map, mapping)
        for name in frames:
            pd.testing.assert_frame_equal(frames[name], full_frames[name])
        tampered = copy.deepcopy(new_frames)
        tampered["test_cold_tgt.pkl"].loc[0, "item"] = 9
        with self.assertRaises(ValueError):
            validate_remap(frames, tampered, mapping, new_map, kept, 60)

    def test_pair_configs_and_complete_job_grid(self):
        experiment = load_experiment(DEFAULT_CONFIG)
        for task, settings in experiment["tasks"].items():
            for model in settings["backbones"]:
                configs = []
                for variant in ("pure", "triclass"):
                    config = Config(model + "_Grounded_TriClass_Pseudo", "Amazon2014", settings["domains"])
                    training_config(config, experiment, task, .2, variant, Path("/tmp/test-paired-study"))
                    configs.append(config.final_config_dict)
                a, b = copy.deepcopy(configs)
                for name in ("log_model_suffix", "pseudo_loss_weight"):
                    a.pop(name); b.pop(name)
                self.assertEqual(a, b)
                self.assertEqual(configs[0]["pseudo_loss_weight"], [0.])
                self.assertGreater(configs[1]["pseudo_loss_weight"][0], 0)
        jobs = make_jobs(experiment, DEFAULT_CONFIG, Path("/tmp/test-paired-study"))
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 80)
        self.assertEqual(sum(j["kind"] == "build" for j in jobs), 10)
        self.assertEqual(len(set(j["id"] for j in jobs)), 90)

    def test_complete_tiny_builder_preserves_candidates_and_overwrites_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, view = root / "base", root / "view"
            base.mkdir()
            users, mapping, frames = fixture()
            atomic_json(str(base / "all_users.json"), users)
            atomic_json(str(base / "id_mapping.json"), mapping)
            for name, frame in frames.items():
                frame.to_pickle(base / name)
            cache = base / "grounded_triclass"
            cache.mkdir()
            rng = np.random.default_rng(999)
            features = rng.random((64, 6, 4)).astype(np.float32)
            features[:, :, 2] = 999.
            items = np.tile(np.arange(1, 7), (64, 1))
            for scope, indices in (("overlap", np.arange(60)), ("cold", np.arange(60, 64))):
                atomic_npz(str(cache / (scope + "_candidates.npz")), user_ids=indices + 1,
                           item_ids=items[indices], features=features[indices], sources=np.ones((len(indices), 6)))
            neighbors = np.tile(np.array([6, 7]), (11, 1))
            np.save(cache / "target_item_neighbors.npy", neighbors)
            source_dir = base / "grounded_static_confidence"
            source_dir.mkdir()
            np.save(source_dir / "source_lightgcn_user_embeddings.npy", rng.normal(size=(65, 8)).astype(np.float32))
            atomic_json(str(cache / "build_spec.json"), {"reused_source_teacher_sha256": {
                "source_lightgcn_user_embeddings.npy": file_sha256(str(source_dir / "source_lightgcn_user_embeddings.npy"))}})
            names = ("overlap_candidates.npz", "cold_candidates.npz", "target_item_neighbors.npy")
            manifest = dict(teacher_target_users="target_only", cold_evaluation_labels_used=False,
                mapping_sha256=mapping_fingerprint(mapping),
                train_src_sha256=file_sha256(str(base / "train_src.pkl")),
                train_tgt_sha256=file_sha256(str(base / "train_tgt.pkl")),
                artifacts={name: file_sha256(str(cache / name)) for name in names})
            atomic_json(str(cache / "manifest.json"), manifest)
            settings = dict(device="cpu", dataset="Amazon2014", domains=["source", "target"],
                calibrator=dict(learning_rate=.01, weight_decay=0, batch_size=128, max_epochs=3,
                                patience=2, minimum_delta=0),
                candidates=dict(neighbor_count=20, neighbor_temperature=.1),
                selection=dict(eta_values=[0., .25, .5], top_k=5, thresholds=[0., .1, .2], minimum_user_coverage=.2))
            experiment = dict(protocol="test", seed=999, ratios=[.2, .4, .6, .8, 1.], num_threads=1)
            base_hashes = {name: file_sha256(str(base / name)) for name in DATA_FILES}
            with patch("tools.run_triclass_paired_overlap.locations", return_value=(settings, base, view, view / "grounded_triclass")):
                first = prepare_view(experiment, "ClothSport", .4, root)
                self.assertEqual(first, prepare_view(experiment, "ClothSport", .4, root))
                report = build_budget(experiment, "ClothSport", .4, root)
                self.assertEqual(report, build_budget(experiment, "ClothSport", .4, root))
            revised = view / "grounded_triclass"
            with np.load(revised / "all_candidates.npz") as data:
                order = np.asarray(first["source_new_to_old"][1:]) - 1
                np.testing.assert_array_equal(data["item_ids"], items[order])
                for feature in (0, 1, 3):
                    np.testing.assert_array_equal(data["features"][:, :, feature], features[order, :, feature])
                self.assertTrue((data["features"][:, :, 2] <= 1.000001).all())
            with np.load(revised / "cold_support.npz") as data:
                n = first["retained_overlap_users"]
                np.testing.assert_array_equal(data["user_ids"], np.arange(n + 1, 65))
                np.testing.assert_allclose(data["probabilities"].sum(2), 1., atol=1e-6)
            self.assertEqual(base_hashes, {name: file_sha256(str(base / name)) for name in DATA_FILES})


if __name__ == "__main__":
    unittest.main()
