"""CDRIB/TriClass with deterministic sparse graph propagation."""

import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.cdrib import GCN
from models.cdrib_grounded_triclass_pseudo import CDRIB_Grounded_TriClass_Pseudo


def enable_determinism():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def weights_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class DeterministicGraphConvolution(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        # Reuse parameters without drawing another initialization sample.
        self.register_parameter("weight", original.weight)
        self.register_parameter("bias", original.bias)
        self._cached_adjacency = None
        self.register_buffer("_batched_adjacency", None, persistent=False)

    def forward(self, inputs, adjacency):
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("Deterministic graph propagation requires deterministic algorithms.")
        if self._cached_adjacency is not adjacency:
            self._cached_adjacency = adjacency
            self._batched_adjacency = adjacency.unsqueeze(0).coalesce()
        support = torch.mm(inputs, self.weight)
        output = torch.bmm(self._batched_adjacency, support.unsqueeze(0)).squeeze(0)
        return output + self.bias if self.bias is not None else output


class CDRIB_Grounded_TriClass_Deterministic(CDRIB_Grounded_TriClass_Pseudo):
    def __init__(self, config, dataloader):
        enable_determinism()
        super().__init__(config, dataloader)
        for module in list(self.modules()):
            if isinstance(module, GCN):
                module.gc1 = DeterministicGraphConvolution(module.gc1)
        self._trace_path = config.get("repro_trace_path")
        if self._trace_path:
            path = Path(self._trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8"):
                pass
            self._record_reproducibility(-1)

    def _record_reproducibility(self, epoch):
        digest = lambda value: hashlib.sha256(pickle.dumps(value)).hexdigest()
        record = dict(epoch=epoch, weights_sha256=weights_digest(self),
            python_rng=digest(random.getstate()), numpy_rng=digest(np.random.get_state()),
            torch_rng=digest(torch.get_rng_state().numpy().tobytes()),
            cuda_rng=[digest(state.cpu().numpy().tobytes()) for state in torch.cuda.get_rng_state_all()])
        if self.pseudo_supervisor is not None:
            record["pseudo_rng"] = self.pseudo_supervisor.rng.bit_generator.state
        with Path(self._trace_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def post_epoch_processing(self, epoch_idx):
        result = super().post_epoch_processing(epoch_idx)
        if self._trace_path:
            self._record_reproducibility(epoch_idx)
        return result
