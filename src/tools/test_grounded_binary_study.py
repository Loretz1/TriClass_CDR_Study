"""Check binary labels, split isolation, cached training and backbone settings."""

import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models import grounded_binary_pseudo as binary
from models.grounded_triclass_pseudo import TriClassPseudoSupervisor
from models.target_pattern_pseudo import apply_model_domain_profile, atomic_json, atomic_npz, file_sha256
from tools.run_grounded_binary_study import create_jobs, reference_results, train
from tools.test_grounded_triclass import TriClassTests
from utils.configurator import Config


SETTINGS = dict(learning_rate=0.01, weight_decay=0.0001, batch_size=32,
                max_epochs=4, patience=2, minimum_delta=0.000001)


class BinaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_label_merging_preserves_padding(self):
        original = np.array([[0, 1, 2, -1]])
        np.testing.assert_array_equal(binary.binary_labels(original, "hit_only"), [[0, 1, 1, -1]])
        np.testing.assert_array_equal(binary.binary_labels(original, "hit_near"), [[0, 0, 1, -1]])
        with self.assertRaises(ValueError):
            binary.binary_labels(np.array([4]), "hit_only")

    def test_fitting_is_deterministic_and_does_not_use_test_users(self):
        features = np.random.default_rng(11).normal(size=(12, 6, 4)).astype(np.float32)
        labels = np.tile([0, 1, 0, 1, 0, 1], (12, 1))
        codes = np.repeat([0, 1, 2], 4)
        model, state = binary.fit_calibrator(features, labels, codes, SETTINGS, 999, "cpu")
        altered_x, altered_y = features.copy(), labels.copy()
        altered_x[codes == 2] = 1000
        altered_y[codes == 2] = 0
        _, changed = binary.fit_calibrator(altered_x, altered_y, codes, SETTINGS, 999, "cpu")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10)
        np.testing.assert_allclose(model.mean.numpy(), features[codes == 0].reshape(-1, 4).mean(0), atol=1e-6)
        for key in state["state_dict"]:
            torch.testing.assert_close(state["state_dict"][key], changed["state_dict"][key], rtol=0, atol=0)

    def test_threshold_uses_validation_and_binary_positive_class(self):
        items = np.tile([1, 2, 3], (6, 1))
        codes = np.array([0, 0, 1, 1, 2, 2])
        labels = np.tile([0, 1, 2], (6, 1))
        p = np.tile(np.array([[.3, .7], [.8, .2], [.01, .99]]), (6, 1, 1))
        settings = dict(eta_values=[0, .25, .5], thresholds=[0, .5], top_k=1, minimum_user_coverage=.2)
        for variant in binary.VARIANTS:
            y = binary.binary_labels(labels, variant)
            result = binary.choose_binary_threshold(items, p, y, codes, settings)
            altered = y.copy()
            altered[codes == 2] = 1 - altered[codes == 2]
            self.assertEqual(result, binary.choose_binary_threshold(items, p, altered, codes, settings))
            expected_precision = 0 if variant == "hit_only" else 1
            self.assertEqual(result["0.0"]["validation"]["hit_precision"], expected_precision)
        compatible = binary.sampler_probabilities(p)
        np.testing.assert_array_equal(compatible[..., 1], 0)
        np.testing.assert_allclose(compatible[..., 0], p[..., 0])

    def test_real_builder_reuses_inputs_and_loads_into_existing_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loader = TriClassTests().fixture(temporary)
            manifest = binary.read_json(root / "manifest.json")
            selection = dict(eta_values=[0, .25, .5], thresholds=[0, .01, .1], top_k=5,
                             minimum_user_coverage=.2)
            spec = {"config": {"seed": 999, "calibrator": SETTINGS, "selection": selection},
                    "build_sha256": "fixture"}
            rng = np.random.default_rng(42)
            original = np.tile([0, 1, 2], (6, 1)).astype(np.int8)
            codes = np.repeat([0, 1, 2], 2)
            ids = np.tile([1, 2, 3], (6, 1))
            atomic_npz(str(root / "overlap_candidates.npz"), user_ids=np.arange(1, 7),
                       item_ids=ids, features=rng.normal(size=(6, 3, 4)).astype(np.float32))
            atomic_npz(str(root / "overlap_support.npz"), user_ids=np.arange(1, 7), item_ids=ids,
                       labels=original, split_codes=codes)
            atomic_npz(str(root / "cold_candidates.npz"), user_ids=np.arange(7, 11),
                       item_ids=np.tile([1, 2, 3], (4, 1)), features=rng.normal(size=(4, 3, 4)).astype(np.float32))
            inputs = ("overlap_candidates.npz", "overlap_support.npz", "cold_candidates.npz")
            manifest.update(build_sha256="fixture", teacher_target_users="target_only",
                            overlap_vote_users="calibrator_train_excluding_query")
            manifest["artifacts"].update({name: file_sha256(str(root / name)) for name in inputs})
            atomic_json(str(root / "manifest.json"), manifest)
            atomic_json(str(root / "build_spec.json"), spec)
            source_hashes = {name: file_sha256(str(root / name)) for name in inputs + ("manifest.json",)}
            config = dict(device="cpu", dataset="Amazon2014", domains=manifest["domains"], output_subdir=".")
            for variant in binary.VARIANTS:
                dest = root / ("binary_" + variant)
                with patch.object(binary, "source_config", return_value=config), \
                     patch.object(binary, "dataset_split_directory", return_value=str(root)), \
                     patch.object(binary, "cache_directory", return_value=dest):
                    binary.build({"num_threads": 1, "seed": 999}, "ClothSport", variant)
                    snapshot = file_sha256(str(dest / "cold_support.npz"))
                    binary.build({"num_threads": 1, "seed": 999}, "ClothSport", variant)
                    self.assertEqual(snapshot, file_sha256(str(dest / "cold_support.npz")))
                sampler_config = dict(config, triclass_cache_path=str(dest), near_support_weight=0.0,
                                      pseudo_sampler_seed=999)
                with patch("models.grounded_triclass_pseudo.dataset_split_directory", return_value=str(root)):
                    supervisor = TriClassPseudoSupervisor(sampler_config, loader)
                    with np.load(dest / "pseudo_rankings.npz") as selected:
                        np.testing.assert_array_equal(supervisor.items, selected["item_ids"])
                        np.testing.assert_array_equal(supervisor.weights, selected["scores"])
                    users, pos, neg, weights = supervisor.sample(10)
                    self.assertEqual(len(users), 50)
                    self.assertTrue(torch.isfinite(weights).all())
                    for u, n, w in zip(users, neg, weights):
                        if w > 0:
                            self.assertNotIn(int(n), supervisor.forbidden[int(u) - 7])
                with np.load(dest / "overlap_support.npz") as saved:
                    np.testing.assert_array_equal(saved["labels"], original)
                    np.testing.assert_array_equal(saved["split_codes"], codes)
                    np.testing.assert_array_equal(saved["binary_labels"], binary.binary_labels(original, variant))
                for name, digest in source_hashes.items():
                    self.assertEqual(digest, file_sha256(str(root / name)))

    def test_backbone_configuration_and_reference_log_matching(self):
        experiment = binary.load_experiment()
        jobs = create_jobs(experiment, binary.DEFAULT_CONFIG)
        self.assertEqual(sum(j["kind"] == "build" for j in jobs), 4)
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 16)
        changed_keys = {"domain_profiles", "seed", "triclass_cache_path", "pseudo_loss_weight",
                        "near_support_weight", "pseudo_sampler_seed", "ablation_classifier",
                        "ablation_classifier_class_names", "reference_triclass_near_support_weight",
                        "log_model_suffix", "hyper_parameters"}
        for task, settings in experiment["tasks"].items():
            for model, (weight, eta) in settings["backbones"].items():
                path = binary.SRC / experiment["reference_run"] / f"{model}_{task}.log"
                best, hit = reference_results(path, 999, weight, eta)
                self.assertGreater(best[0], 0)
                self.assertGreater(hit[0], 0)
                baseline = Config(model + "_Grounded_TriClass_Pseudo", "Amazon2014", settings["domains"])
                apply_model_domain_profile(baseline)

                def inspect(model_name, dataset_name, domains, save_model):
                    import utils.quick_start as quick
                    active = quick.Config(model_name, dataset_name, domains)
                    for key, value in baseline.final_config_dict.items():
                        if key not in changed_keys:
                            self.assertEqual(value, active[key], (task, model, key))
                    self.assertEqual(active["pseudo_loss_weight"], [weight])
                    self.assertEqual(active["near_support_weight"], [0.0])
                    self.assertEqual(active["seed"], [999])

                for variant in binary.VARIANTS:
                    with patch("tools.run_grounded_binary_study.read_json", return_value={"variant": variant, "classifier_output_dim": 2}), \
                         patch("utils.quick_start.quick_start", side_effect=inspect):
                        train(experiment, task, model, variant)


if __name__ == "__main__":
    unittest.main()
