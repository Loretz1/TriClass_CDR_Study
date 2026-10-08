"""Count matching, unchanged unions, common negatives and backbone integration."""

import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models import grounded_retrieval_budget as budget
from models import grounded_triclass_pseudo as triclass
from models.target_pattern_pseudo import atomic_json, atomic_npz, file_sha256, set_seed
from tools import run_retrieval_budget_study as runner
from tools.test_grounded_triclass import TriClassTests
from utils.configurator import Config
from utils.utils import get_model


class RetrievalBudgetTests(unittest.TestCase):
    def test_counts_are_per_user_not_fixed_150(self):
        ranked = np.tile(np.arange(1, 9), (3, 1))
        counts = np.array([3, 5, 7])
        items = budget.truncate_per_user(ranked, counts)
        budget.check_candidates(items, counts, 8)
        with self.assertRaises(ValueError):
            budget.check_candidates(items, 5, 8)
        items[0, 1] = items[0, 0]
        with self.assertRaises(ValueError):
            budget.check_candidates(items, counts, 8)

    def test_rrf_fixed_selection_does_not_change_union_or_use_padding(self):
        items = np.array([[3, 2, 1, 0]])
        features = np.array([[[.1, .9, .2, 1.], [.9, .2, .1, 0.], [.8, .8, 0., 2.], [0., 0., 0., 0.]]])
        origins = np.array([[2, 1, 7, 0]], dtype=np.uint8)
        before = items.copy()
        scores = budget.rrf_scores(items, features, origins, 60)
        self.assertEqual(int(budget.fixed_top(items, scores, 1)[0, 0]), 1)
        tied = budget.fixed_top(items, np.zeros(items.shape), 2)
        np.testing.assert_array_equal(tied, [[1, 2]])
        np.testing.assert_array_equal(before, items)

    def fixture(self, root):
        loader = TriClassTests().fixture(str(root))
        manifest = json.loads((root / "manifest.json").read_text())
        manifest.update(protocol=budget.PROTOCOL, loss_weighting="uniform", final_top_k=2)
        users, counts = np.arange(7, 11), np.array([3, 4, 3, 4])
        all_candidates = {}
        for index, variant in enumerate(budget.VARIANTS):
            ranked = np.tile(np.roll(np.arange(1, 7), index), (4, 1))
            candidates = budget.truncate_per_user(ranked, counts)
            all_candidates[variant] = candidates
            name = f"cold_{variant}_eta0.25.npz"
            atomic_npz(str(root / name), user_ids=users, candidate_ids=candidates, candidate_counts=counts,
                item_ids=candidates[:, :2], loss_weights=np.ones((4, 2), dtype=np.float32))
            manifest["artifacts"][name] = file_sha256(str(root / name))
        name = "cold_negative_exclusions.npz"
        atomic_npz(str(root / name), user_ids=users, item_ids=budget.common_negative_exclusions(all_candidates))
        manifest["artifacts"][name] = file_sha256(str(root / name))
        atomic_json(str(root / "manifest.json"), manifest)
        return loader, manifest

    def test_identical_user_and_negative_samples_across_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loader, manifest = self.fixture(root)
            samples = []
            for variant in budget.VARIANTS:
                config = dict(device="cpu", dataset="Amazon2014", domains=manifest["domains"],
                    retrieval_cache_path=str(root), retrieval_variant=variant,
                    near_support_weight=.25, pseudo_sampler_seed=999)
                with patch.object(budget, "dataset_split_directory", return_value=str(root)):
                    sampler = budget.FixedBudgetSupervisor(config, loader)
                self.assertFalse(sampler.target_positive[7])
                samples.append(sampler.sample(16))
                self.assertTrue(torch.all(samples[-1][3] == 1))
            for sample in samples[1:]:
                torch.testing.assert_close(sample[0], samples[0][0], atol=0, rtol=0)
                torch.testing.assert_close(sample[2], samples[0][2], atol=0, rtol=0)

    def test_all_backbones_accept_controlled_supervisor(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loader, manifest = self.fixture(root)
            for name in ("EMCDR", "UniCDR", "CD_CDR_Normal", "CDRIB"):
                config = Config(name + "_Grounded_TriClass_Pseudo", "Amazon2014", manifest["domains"])
                config["device"] = torch.device("cuda" if name == "CDRIB" else "cpu")
                for key, value in dict(seed=999, retrieval_cache_path=str(root), retrieval_variant="trusted",
                    pseudo_loss_weight=.1, near_support_weight=.25, pseudo_sampler_seed=999,
                    pseudo_num_per_batch=2, user_batch_size=2).items():
                    config[key] = value
                set_seed(999)
                with patch.object(budget, "dataset_split_directory", return_value=str(root)), \
                        patch.object(triclass, "TriClassPseudoSupervisor", budget.FixedBudgetSupervisor):
                    model = get_model(config["model"])(config, loader).to(config["device"])
                model.train()
                model.set_train_stage(2 if name == "EMCDR" else 0)
                interaction = {k: torch.tensor(v, device=config["device"]) for k, v in dict(
                    users_overlapped=[1, 2], users_src=[1, 7], users_tgt=[2, 8],
                    pos_items_src=[1, 7], pos_items_tgt=[2, 8], neg_items_src=[5, 6], neg_items_tgt=[5, 6]).items()}
                loss = model.calculate_loss(interaction, 0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_40_runs_one_configuration_each(self):
        experiment = budget.load_experiment()
        jobs = runner.create_jobs(experiment, budget.DEFAULT_CONFIG, Path("/tmp/matched-study"))
        self.assertEqual(sum(j["kind"] == "build" for j in jobs), 2)
        self.assertEqual(sum(j["kind"] == "train" for j in jobs), 40)
        for task, settings in experiment["tasks"].items():
            for model in settings["backbones"]:
                config = Config(model + "_Grounded_TriClass_Pseudo", "Amazon2014", settings["domains"])
                runner.apply_model_domain_profile(config)
                original = copy.deepcopy(config["training_stages"])
                runner.configure(config, experiment, task, model, "trusted", Path("/tmp/matched-study"))
                self.assertEqual(config["training_stages"], original)
                self.assertEqual(config["seed"], [999])
                self.assertEqual(len(config["pseudo_loss_weight"]), 1)


if __name__ == "__main__":
    unittest.main()
