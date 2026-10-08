"""Evidence removal, split isolation, cache compatibility and fixed settings."""

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models import grounded_triclass_evidence_ablation as evidence
from models.grounded_triclass_pseudo import TriClassPseudoSupervisor, fit_calibrator, predict_probabilities
from models.target_pattern_pseudo import apply_model_domain_profile, atomic_json, atomic_npz, file_sha256
from tools import run_triclass_evidence_study as runner
from tools.test_grounded_triclass import TriClassTests
from utils.configurator import Config


SETTINGS = dict(learning_rate=.01, weight_decay=.0001, batch_size=32,
                max_epochs=3, patience=2, minimum_delta=.000001)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_removed_feature_is_inert_and_original_input_unchanged(self):
        original = np.random.default_rng(1).normal(size=(12, 6, 4)).astype(np.float32)
        labels = np.tile([0, 1, 2, 0, 1, 2], (12, 1))
        codes = np.repeat([0, 1, 2], 4)
        for variant, column in evidence.VARIANTS.items():
            source = original.copy()
            x = evidence.mask_evidence(source, variant)
            np.testing.assert_array_equal(source, original)
            np.testing.assert_array_equal(x[..., column], 0)
            retained = [i for i in range(4) if i != column]
            np.testing.assert_array_equal(x[..., retained], original[..., retained])
            model, state = fit_calibrator(x, labels, codes, SETTINGS, 999, "cpu")
            changed = source.copy()
            changed[..., column] = 1000
            changed[codes == 2] = -1000
            y = labels.copy()
            y[codes == 2] = 0
            _, other = fit_calibrator(evidence.mask_evidence(changed, variant), y, codes, SETTINGS, 999, "cpu")
            for key in state["state_dict"]:
                torch.testing.assert_close(state["state_dict"][key], other["state_dict"][key], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(model.linear.weight[:, column]).item(), 0)
            source[..., column] = -1e9
            one = predict_probabilities(model, x, labels >= 0, "cpu")
            two = predict_probabilities(model, evidence.mask_evidence(source, variant), labels >= 0, "cpu")
            np.testing.assert_array_equal(one, two)

    def test_build_preserves_candidates_labels_splits_and_existing_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            loader = TriClassTests().fixture(temporary)
            manifest = evidence.read_json(base / "manifest.json")
            spec = dict(config=dict(seed=999, calibrator=SETTINGS, selection=dict(eta_values=[0., .25, .5],
                thresholds=[0, .01, .1], top_k=5, minimum_user_coverage=.2)), build_sha256="fixture")
            rng = np.random.default_rng(42)
            users, items = np.arange(1, 7), np.tile([1, 2, 3], (6, 1))
            labels, codes = np.tile([0, 1, 2], (6, 1)), np.repeat([0, 1, 2], 2)
            atomic_npz(str(base / "overlap_candidates.npz"), user_ids=users, item_ids=items,
                       features=rng.normal(size=(6, 3, 4)).astype(np.float32))
            atomic_npz(str(base / "overlap_support.npz"), user_ids=users, item_ids=items, labels=labels, split_codes=codes)
            atomic_npz(str(base / "cold_candidates.npz"), user_ids=np.arange(7, 11),
                       item_ids=np.tile([1, 2, 3], (4, 1)), features=rng.normal(size=(4, 3, 4)).astype(np.float32))
            inputs = ("overlap_candidates.npz", "overlap_support.npz", "cold_candidates.npz")
            manifest.update(build_sha256="fixture", teacher_target_users="target_only",
                feature_names=evidence.FEATURE_NAMES, overlap_vote_users="calibrator_train_excluding_query")
            manifest["artifacts"].update({name: file_sha256(str(base / name)) for name in inputs})
            atomic_json(str(base / "manifest.json"), manifest)
            atomic_json(str(base / "build_spec.json"), spec)
            hashes = {name: file_sha256(str(base / name)) for name in inputs + ("manifest.json",)}
            config = dict(device="cpu", dataset="Amazon2014", domains=manifest["domains"])
            for variant in evidence.VARIANTS:
                with patch.object(evidence, "locations", return_value=(config, base, base / "output")):
                    output = evidence.build({"seed": 999, "num_threads": 1}, "ClothSport", variant, base)
                    digest = file_sha256(str(output / "cold_support.npz"))
                    evidence.build({"seed": 999, "num_threads": 1}, "ClothSport", variant, base)
                    self.assertEqual(digest, file_sha256(str(output / "cold_support.npz")))
                with np.load(output / "overlap_support.npz") as saved:
                    np.testing.assert_array_equal(saved["item_ids"], items)
                    np.testing.assert_array_equal(saved["labels"], labels)
                    np.testing.assert_array_equal(saved["split_codes"], codes)
                for eta in (0., .25, .5):
                    cfg = dict(config, triclass_cache_path=str(output), near_support_weight=eta, pseudo_sampler_seed=999)
                    with patch("models.grounded_triclass_pseudo.dataset_split_directory", return_value=str(base)):
                        sampler = TriClassPseudoSupervisor(cfg, loader)
                    with np.load(output / ("pseudo_rankings_eta" + str(eta) + ".npz")) as ranks:
                        np.testing.assert_array_equal(sampler.items, ranks["item_ids"])
                        np.testing.assert_array_equal(sampler.weights, ranks["scores"])
                for name, digest in hashes.items():
                    self.assertEqual(digest, file_sha256(str(base / name)))

    def test_eight_builds_and_32_single_setting_runs_keep_backbone_config(self):
        experiment = evidence.load_experiment()
        jobs = runner.create_jobs(experiment, evidence.DEFAULT_CONFIG, Path("/tmp/test-evidence"))
        self.assertEqual(sum(j["kind"] == "build" for j in jobs), 8)
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 32)
        allowed = {"domain_profiles", "data_path", "seed", "triclass_cache_path", "pseudo_loss_weight",
                   "near_support_weight", "pseudo_sampler_seed", "evidence_ablation", "log_model_suffix", "log_dir", "hyper_parameters"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task, settings in experiment["tasks"].items():
                for model, (weight, eta) in settings["backbones"].items():
                    baseline = Config(model + "_Grounded_TriClass_Pseudo", "Amazon2014", settings["domains"])
                    apply_model_domain_profile(baseline)

                    def inspect(model_name, dataset, domains, save_model):
                        import utils.quick_start as quick
                        actual = quick.Config(model_name, dataset, domains)
                        for key, value in baseline.final_config_dict.items():
                            if key not in allowed:
                                self.assertEqual(actual[key], value, (task, model, key))
                        self.assertEqual(actual["seed"], [999])
                        self.assertEqual(actual["pseudo_loss_weight"], [weight])
                        self.assertEqual(actual["near_support_weight"], [eta])

                    for variant in evidence.VARIANTS:
                        dest = root / "cache" / task / variant
                        dest.mkdir(parents=True, exist_ok=True)
                        atomic_json(str(dest / "manifest.json"), dict(variant=variant, candidate_routes_unchanged=True))
                        with patch("utils.quick_start.quick_start", side_effect=inspect):
                            runner.train(experiment, task, model, variant, root)

    def test_collection_verifies_files_and_does_not_delete_remote_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            atomic_json(str(directory / "status.json"), {"finished_at": "now", "jobs": [{"status": "complete"}]})
            hashes = {"status.json": file_sha256(str(directory / "status.json"))}
            responses = [json.dumps({"finished_at": "now", "jobs": [{"status": "complete"}]}), json.dumps(hashes)]
            with patch.object(runner, "ssh", side_effect=responses), patch.object(runner.subprocess, "run") as transfer:
                self.assertEqual(runner.collect("testhost", Path("/remote"), directory, 1), 0)
            self.assertNotIn("--remove-source-files", transfer.call_args[0][0])
            self.assertEqual(evidence.read_json(directory / "collector_state.json")["verified_sha256"], hashes)


if __name__ == "__main__":
    unittest.main()
