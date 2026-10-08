"""Focused tests; CDRIB integration uses CUDA because its base requires it."""

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

from models.grounded_triclass_pseudo import (
    CLASS_NAMES, VERSION, TriClassPseudoSupervisor, build, build_candidates, build_neighbors,
    choose_thresholds, exact_overlap_votes, fit_calibrator, isolate_teacher_edges,
    predict_probabilities, select_support, split_overlap_users, support_labels,
)
from models.target_pattern_pseudo import (
    apply_model_domain_profile, atomic_json, atomic_npz, file_sha256,
    mapping_fingerprint, set_seed,
)
from utils.utils import get_model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["Clothing_Shoes_and_Jewelry", "Sports_and_Outdoors"]


class TriClassTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_user_splits_and_teacher_exclusion(self):
        a = split_overlap_users(30, [0.6, 0.2, 0.2], 999)
        np.testing.assert_array_equal(a, split_overlap_users(30, [0.6, 0.2, 0.2], 999))
        np.testing.assert_array_equal(np.bincount(a), [18, 6, 6])
        frame = pd.DataFrame({"user": [1, 2, 3, 4, 5], "item": [4, 3, 2, 1, 5]})
        filtered = isolate_teacher_edges(frame, 3)
        self.assertEqual(filtered.user.tolist(), [4, 5])
        changed = frame.copy(); changed.loc[changed.user <= 3, "item"] = 99
        pd.testing.assert_frame_equal(filtered, isolate_teacher_edges(changed, 3))

    def test_votes_exclude_query_and_nonreference_histories(self):
        embeddings = np.array([[0, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float32)
        candidates = np.array([[5, 6, 7, 0]])
        h = {1: {5}, 2: {6}, 3: {7}}
        votes = exact_overlap_votes(np.array([1]), candidates, embeddings, [1, 2], h, 20, 0.1)
        np.testing.assert_allclose(votes, [[0, 1, 0, 0]])
        h[1] = h[3] = {5, 6, 7}
        np.testing.assert_allclose(votes, exact_overlap_votes(np.array([1]), candidates, embeddings, [1, 2], h, 20, 0.1))

    def test_label_priority_and_padding(self):
        neighbors = np.zeros((6, 2), dtype=np.int64)
        neighbors[1] = [2, 3]; neighbors[2] = [1, 3]
        labels = support_labels([1], np.array([[1, 2, 3, 4, 0]]), {1: {1, 2}}, neighbors)
        np.testing.assert_array_equal(labels, [[0, 0, 1, 2, -1]])

    def test_fixed_slots_preserve_absolute_weight_and_gradient(self):
        u = torch.ones(4, 2, requires_grad=True)
        p = torch.ones(4, 2); n = torch.zeros(4, 2)
        loss = TriClassPseudoSupervisor.weighted_bpr_loss(u, p, n, torch.ones(4))
        gradient = torch.autograd.grad(loss, u, retain_graph=True)[0]
        small = TriClassPseudoSupervisor.weighted_bpr_loss(u, p, n, torch.full((4,), 0.01))
        small_gradient = torch.autograd.grad(small, u, retain_graph=True)[0]
        torch.testing.assert_close(small, loss * 0.01)
        torch.testing.assert_close(small_gradient, gradient * 0.01)
        padded = TriClassPseudoSupervisor.weighted_bpr_loss(u, p, n, torch.tensor([1., 0, 0, 0]))
        torch.testing.assert_close(padded, loss / 4)

    def test_selection_can_abstain_and_eta_zero_excludes_near(self):
        items = np.array([[3, 1, 2, 0], [1, 2, 3, 0]])
        probs = np.array([[[0.2, 0.5, 0.3], [0.2, 0.1, 0.7], [0, 1, 0], [0, 0, 0]],
                          [[0.001, 0.001, 0.998]] * 3 + [[0, 0, 0]]], dtype=np.float32)
        selected, weights = select_support(items, probs, 3, 0, 0.1)
        np.testing.assert_array_equal(selected, [[1, 3, 0], [0, 0, 0]])
        self.assertEqual(weights[1].sum(), 0)
        selected, _ = select_support(items, probs, 3, 0.5, 0.1)
        np.testing.assert_array_equal(selected[0], [2, 3, 1])

    def test_classifier_training_normalization_and_no_test_selection(self):
        rng = np.random.default_rng(99)
        features = rng.normal(size=(30, 6, 4)).astype(np.float32)
        labels = np.tile([0, 1, 2, 0, 1, 2], (30, 1)).astype(np.int8)
        codes = split_overlap_users(30, [0.6, 0.2, 0.2], 999)
        settings = dict(learning_rate=0.01, weight_decay=0, batch_size=64,
                        max_epochs=3, patience=2, minimum_delta=0)
        model, ckpt = fit_calibrator(features, labels, codes, settings, 999, "cpu")
        np.testing.assert_allclose(model.mean.numpy(), features[codes == 0].reshape(-1, 4).mean(0), atol=1e-6)
        altered = features.copy(); altered[codes == 2] = 1000
        altered_y = labels.copy(); altered_y[codes == 2] = 2
        _, other = fit_calibrator(altered, altered_y, codes, settings, 999, "cpu")
        for key in ckpt["state_dict"]:
            torch.testing.assert_close(ckpt["state_dict"][key], other["state_dict"][key], rtol=0, atol=0)
        probabilities = predict_probabilities(model, features, labels >= 0, "cpu")
        np.testing.assert_allclose(probabilities.sum(-1), 1, atol=1e-6)
        config = {"selection": dict(eta_values=[0, 0.25], thresholds=[0, 0.1], top_k=2, minimum_user_coverage=0.2)}
        items = np.tile(np.arange(1, 7), (30, 1))
        thresholds = choose_thresholds(items, probabilities, labels, codes, config)
        self.assertEqual(thresholds, choose_thresholds(items, probabilities, altered_y, codes, config))

    def test_union_and_four_features(self):
        queries = np.array([[0, 0], [1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
        semantic = np.array([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], dtype=np.float32)
        collab = np.vstack((np.zeros((1, 2), dtype=np.float32), semantic))
        counts = np.array([0, 2, 3, 8, 1])
        config = {"device": "cpu", "candidates": dict(per_source_k=2, batch_size=2, neighbor_count=2, neighbor_temperature=0.1)}
        items, features, origins = build_candidates(np.array([3]), queries, semantic, torch.nn.Identity(),
            collab, queries, np.array([1, 2]), {1: {1, 2}, 2: {3}}, counts, config)
        self.assertEqual(features.shape, items.shape + (4,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(set(items[0][items[0] > 0])), int((items[0] > 0).sum()))
        self.assertTrue(np.any(origins & 1)); self.assertTrue(np.any(origins & 2)); self.assertTrue(np.any(origins & 4))
        neighbor = build_neighbors(collab, counts, 2, "cpu", 2)
        for item in range(1, 5):
            self.assertNotIn(item, neighbor[item])

    def fixture(self, directory):
        overlap, num_users, num_items = 6, 10, 12
        src_names = ["PAD"] + ["overlap_" + str(i) for i in range(1, 7)] + ["source_" + str(i) for i in range(7, 11)]
        tgt_names = src_names[:7] + ["target_" + str(i) for i in range(7, 11)]
        mapping = {domain: {"id2user": users, "user2id": {u: i for i, u in enumerate(users)},
                           "id2item": ["PAD"] + [domain + str(i) for i in range(1, 13)]}
                   for domain, users in (("src", src_names), ("tgt", tgt_names))}
        frame = pd.DataFrame([(u, i) for u in range(1, 11) for i in (u, u + 1)], columns=["user", "item"])
        frame.to_pickle(os.path.join(directory, "train_src.pkl")); frame.to_pickle(os.path.join(directory, "train_tgt.pkl"))
        dataset = SimpleNamespace(num_users_overlap=overlap, num_users_src=num_users, num_users_tgt=num_users,
            num_items_src=num_items, num_items_tgt=num_items, id_mapping=mapping,
            df={"train_src": frame, "train_tgt": frame},
            positive_items_src={u: {u, u + 1} for u in range(1, 11)},
            positive_items_tgt={u: {u, u + 1} for u in range(1, 11)})
        matrix = sp.coo_matrix((np.ones(len(frame)), (frame.user.to_numpy() - 1, frame.item.to_numpy() - 1)), shape=(10, 12))
        loader = SimpleNamespace(dataset=dataset, inter_matrix=lambda domain, form: matrix)
        path = os.path.join(directory, "cold_support.npz")
        probabilities = np.tile(np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.01, 0.01, 0.98]]), (4, 1, 1)).astype(np.float32)
        atomic_npz(path, user_ids=np.arange(7, 11), item_ids=np.tile([1, 2, 3], (4, 1)), probabilities=probabilities)
        manifest = dict(format_version=VERSION, dataset="Amazon2014", domains=DOMAINS,
                        mapping_sha256=mapping_fingerprint(mapping), num_users_src=10, num_users_overlap=6,
                        num_items_tgt=12, class_names=CLASS_NAMES, cold_evaluation_labels_used=False,
                        train_src_sha256=file_sha256(os.path.join(directory, "train_src.pkl")),
                        train_tgt_sha256=file_sha256(os.path.join(directory, "train_tgt.pkl")), top_k=5,
                        selection={"0.25": {"threshold": 0.1}}, artifacts={"cold_support.npz": file_sha256(path)})
        atomic_json(os.path.join(directory, "manifest.json"), manifest)
        return loader

    def test_sampler_negative_identity_and_fixed_slots(self):
        with tempfile.TemporaryDirectory() as d, patch("models.grounded_triclass_pseudo.dataset_split_directory", return_value=None) as resolve:
            resolve.return_value = d
            loader = self.fixture(d)
            config = dict(device="cpu", dataset="Amazon2014", domains=DOMAINS,
                          triclass_cache_path=d, near_support_weight=0.25, pseudo_sampler_seed=999)
            before = torch.get_rng_state().clone()
            a = TriClassPseudoSupervisor(config, loader); b = TriClassPseudoSupervisor(config, loader)
            torch.testing.assert_close(before, torch.get_rng_state())
            # Target ID 7 belongs to somebody else; its observations must not leak.
            self.assertEqual(a.target_positive[7], set())
            sample = a.sample(100)
            for x, y in zip(sample, b.sample(100)):
                torch.testing.assert_close(x, y)
            self.assertEqual(len(sample[0]), 100 * 5)
            self.assertTrue((sample[3].reshape(100, 5)[:, 2:] == 0).all())
            for uid, negative, weight in zip(sample[0], sample[2], sample[3]):
                if weight > 0:
                    self.assertNotIn(int(negative), a.forbidden[int(uid) - 7])

    def test_all_backbones_real_loss_backward(self):
        names = ["EMCDR", "UniCDR", "CDRIB", "CD_CDR_Normal"]
        with tempfile.TemporaryDirectory() as d, patch("models.grounded_triclass_pseudo.dataset_split_directory", return_value=None) as resolve:
            resolve.return_value = d
            loader = self.fixture(d)
            for domains in (DOMAINS, ["Cell_Phones_and_Accessories", "Electronics"]):
                manifest_path = os.path.join(d, "manifest.json")
                with open(manifest_path) as handle:
                    manifest = json.load(handle)
                manifest["domains"] = domains
                atomic_json(manifest_path, manifest)
                for name in names:
                    with self.subTest(backbone=name, domains=domains):
                        model_name = name + "_Grounded_TriClass_Pseudo"
                        with open(os.path.join(ROOT, "configs/model", model_name + ".yaml")) as handle:
                            config = yaml.safe_load(handle)
                        if name == "CDRIB" and not torch.cuda.is_available():
                            continue
                        device = "cuda" if name == "CDRIB" else "cpu"
                        config.update(device=device, dataset="Amazon2014", domains=domains, seed=999,
                                      triclass_cache_path=d, pseudo_loss_weight=0.1,
                                      near_support_weight=0.25, pseudo_num_per_batch=2,
                                      user_batch_size=2)
                        set_seed(999)
                        model = get_model(model_name)(config, loader).to(device)
                        model.train(); model.set_train_stage(2 if name == "EMCDR" else 0)
                        batch = {"users_overlapped": torch.tensor([1, 2]), "users_src": torch.tensor([1, 7]),
                                 "users_tgt": torch.tensor([2, 8]), "pos_items_src": torch.tensor([1, 7]),
                                 "pos_items_tgt": torch.tensor([2, 8]), "neg_items_src": torch.tensor([5, 6]),
                                 "neg_items_tgt": torch.tensor([5, 6])}
                        batch = {k: v.to(device) for k, v in batch.items()}
                        loss = model.calculate_loss(batch, 0)
                        self.assertTrue(torch.isfinite(loss).all(), str(loss))
                        loss.backward()
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
                        self.assertIsNone(model._triclass_source_graph)
                        self.assertIsNone(model._triclass_target_graph)
                        model.zero_grad()
                        # A second forward must not retain the previous autograd graph.
                        model.calculate_loss(batch, 1).backward()

    def test_config_backbone_settings_unchanged(self):
        for name in ["EMCDR", "UniCDR", "CDRIB", "CD_CDR_Normal"]:
            base_name = name if name == "CD_CDR_Normal" else name + "_Grounded_Backbone"
            with open(os.path.join(ROOT, "configs/model", base_name + ".yaml")) as f:
                base = yaml.safe_load(f)
            with open(os.path.join(ROOT, "configs/model", name + "_Grounded_TriClass_Pseudo.yaml")) as f:
                new = yaml.safe_load(f)
            for domains in [DOMAINS, ["Cell_Phones_and_Accessories", "Electronics"]]:
                a, b = copy.deepcopy(base), copy.deepcopy(new)
                a["domains"] = b["domains"] = domains
                apply_model_domain_profile(a); apply_model_domain_profile(b)
                for key in a:
                    if key not in ("domain_profiles", "hyper_parameters", "log_model_suffix"):
                        self.assertEqual(a[key], b.get(key), name + ": " + key)

    def test_offline_builder_end_to_end_and_resume(self):
        rng = np.random.default_rng(2027)
        with tempfile.TemporaryDirectory() as d:
            semantic_dir = os.path.join(d, "target_pattern_pseudo")
            os.makedirs(semantic_dir)
            overlap, source_count, target_count, item_count, dim = 30, 35, 60, 40, 16
            common = ["PAD"] + ["overlap" + str(i) for i in range(1, overlap + 1)]
            mapping = {}
            for domain, count in (("src", source_count), ("tgt", target_count)):
                users = common + [domain + str(i) for i in range(overlap + 1, count + 1)]
                mapping[domain] = {"id2user": users, "user2id": {u: i for i, u in enumerate(users)},
                                   "id2item": ["PAD"] + [str(i) for i in range(1, item_count + 1)]}
            frames = []
            for count in (source_count, target_count):
                frames.append(pd.DataFrame([(u, int(i)) for u in range(1, count + 1)
                    for i in rng.choice(np.arange(1, item_count + 1), 6, replace=False)], columns=["user", "item"]))
            src, tgt = frames
            src.to_pickle(os.path.join(d, "train_src.pkl")); tgt.to_pickle(os.path.join(d, "train_tgt.pkl"))
            atomic_json(os.path.join(semantic_dir, "manifest.json"),
                        {"dataset": "Amazon2014", "domains": DOMAINS, "mapping_sha256": mapping_fingerprint(mapping)})

            def embedding_cache(prefix, keys):
                base = os.path.join(semantic_dir, prefix + "_semantic_embeddings")
                np.save(base + ".npy", rng.normal(size=(len(keys), dim)).astype(np.float32))
                np.save(base + "_complete.npy", np.ones(len(keys), dtype=bool))
                atomic_json(base + "_metadata.json", {"keys": keys, "model": "synthetic-test"})

            embedding_cache("target_item", [str(i) for i in range(1, item_count + 1)])
            for scope, users in (("overlap", range(1, overlap + 1)), ("source", range(overlap + 1, source_count + 1))):
                with open(os.path.join(semantic_dir, scope + "_user_interests.jsonl"), "w") as f:
                    for user in users:
                        f.write(json.dumps({"user_id": user, "interests": [{"weight": 1.0}]}) + "\n")
                embedding_cache(scope + "_interest", [str(u) + ":0" for u in users])
            with open(os.path.join(ROOT, "configs/model/GroundedTriClass.yaml")) as handle:
                config = yaml.safe_load(handle)
            config.update(device="cpu", dataset="Amazon2014", domains=DOMAINS)
            config["lightgcn"].update(embedding_dim=8, layers=1, max_epochs=2, patience=2,
                                      samples_per_epoch=100, validation_samples=50)
            config["bridge"].update(hidden_dim=16, max_epochs=2, patience=2, validation_users=3,
                                    batch_size=16, validation_top_k=5, negatives_per_positive=2)
            config["calibrator"].update(max_epochs=3, patience=2, batch_size=64)
            config["candidates"].update(per_source_k=5, batch_size=8, neighbor_count=3)
            config["labels"]["neighbor_k"] = 2
            config["selection"].update(top_k=3, thresholds=[0, 0.1])
            all_users = {"overlap_users": common[1:]}
            with patch("models.grounded_triclass_pseudo.load_split", return_value=(d, all_users, mapping, src, tgt)), \
                 patch("models.grounded_triclass_pseudo.dataset_split_directory", return_value=d), \
                 patch("models.grounded_triclass_pseudo.configure_logging"):
                output = build(config)
                with open(os.path.join(output, "manifest.json")) as f:
                    manifest = json.load(f)
                self.assertEqual(manifest["teacher_target_users"], "target_only")
                self.assertFalse(manifest["cold_evaluation_labels_used"])
                with np.load(os.path.join(output, "cold_support.npz")) as cached:
                    np.testing.assert_array_equal(cached["user_ids"], np.arange(31, 36))
                    valid = cached["item_ids"] > 0
                    np.testing.assert_allclose(cached["probabilities"][valid].sum(-1), 1, atol=1e-6)
                digest = file_sha256(os.path.join(output, "cold_support.npz"))
                self.assertEqual(build(config), output)
                self.assertEqual(file_sha256(os.path.join(output, "cold_support.npz")), digest)
                config["labels"]["neighbor_k"] = 3
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    build(config)


if __name__ == "__main__":
    unittest.main()
