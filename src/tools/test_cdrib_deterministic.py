"""Focused tests for deterministic propagation and the repeat gate."""

import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from models.cdrib import GraphConvolution
from models.cdrib_grounded_triclass_deterministic import DeterministicGraphConvolution, enable_determinism
from tools.run_cdrib_deterministic import check_repeat, job_id, make_jobs


class DeterministicCDRIBTests(unittest.TestCase):
    def test_replacement_preserves_parameters_and_rng(self):
        original = GraphConvolution(4, 3)
        before = torch.get_rng_state().clone()
        replacement = DeterministicGraphConvolution(original)
        self.assertIs(original.weight, replacement.weight)
        self.assertIs(original.bias, replacement.bias)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(list(original.state_dict()), list(replacement.state_dict()))

    def check_layer(self, device):
        enable_determinism()
        torch.manual_seed(999)
        original = GraphConvolution(4, 3).to(device)
        layer = DeterministicGraphConvolution(original)
        dense = torch.tensor([[.2, .3, .5], [0., 1., 0.]], device=device)
        graph = dense.to_sparse().coalesce()
        inputs = torch.randn(3, 4, device=device, requires_grad=True)
        value = layer(inputs, graph)
        expected = dense @ (inputs @ original.weight) + original.bias
        torch.testing.assert_close(value, expected)
        gradient = torch.randn_like(value)
        records = []
        for _ in range(3):
            layer.zero_grad(set_to_none=True)
            inputs.grad = None
            output = layer(inputs, graph)
            output.backward(gradient)
            records.append((output.detach().clone(), inputs.grad.clone(), layer.weight.grad.clone()))
        for run in records[1:]:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(records[0], run)))

    def test_cpu_layer(self):
        self.check_layer("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_layer(self):
        self.check_layer("cuda")

    def test_queue_counts_and_dependencies(self):
        jobs = make_jobs(Path("/tmp/run"), Path("/tmp/input"))
        self.assertEqual(len(jobs), 24)
        self.assertEqual(sum(j["repeat"] == 1 for j in jobs), 4)
        self.assertEqual(sum(j["ratio"] < 1 for j in jobs), 16)
        for job in jobs:
            if job["ratio"] < 1:
                self.assertEqual(job["dependency"], job_id(job["task"], 1., job["variant"], 1))

    def test_repeat_requires_identical_trace_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(check_repeat(root, "ClothSport", "pure")["status"], "waiting")
            record = {k: "equal" for k in ("validation_and_test", "stop_epoch", "final_weights_sha256",
                "trace_sha256", "data", "cache_manifest_sha256", "cold_support_sha256", "source_sha256",
                "torch_version", "cuda_version", "gpu", "num_threads", "deterministic", "cublas_workspace")}
            for repeat in (0, 1):
                (root / (job_id("ClothSport", 1., "pure", repeat) + ".result.json")).write_text(json.dumps(record))
            self.assertEqual(check_repeat(root, "ClothSport", "pure")["status"], "passed")
            record["trace_sha256"] = "different"
            (root / (job_id("ClothSport", 1., "pure", 1) + ".result.json")).write_text(json.dumps(record))
            self.assertEqual(check_repeat(root, "ClothSport", "pure")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
