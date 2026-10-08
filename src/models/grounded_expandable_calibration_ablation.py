"""Confidence-calibration ablations for the grounded expandable model."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from models.grounded_expandable_pseudo import (
    ExpandableGroundedPseudoSupervisor,
    GroundedExpandablePseudoMixin,
)
from models.grounded_static_confidence import (
    PositiveLinearCalibrator,
    average_precision,
)
from models.target_pattern_pseudo import config_scalar


CALIBRATION_ABLATIONS = {
    "mean_all": (0, 1, 2),
    "without_semantic": (1, 2),
    "without_overlap": (0, 2),
}
EVIDENCE_NAMES = ("semantic", "overlap", "model")


class CalibrationAblationExpandableSupervisor(
    ExpandableGroundedPseudoSupervisor
):
    """Replace only the dynamic confidence-combination rule."""

    def __init__(self, config, dataloader):
        mode = str(config_scalar(config.get("calibration_ablation"), "mean_all"))
        if mode not in CALIBRATION_ABLATIONS:
            raise ValueError(
                "Unknown calibration_ablation {!r}. Available modes: {}.".format(
                    mode, ", ".join(CALIBRATION_ABLATIONS)
                )
            )
        self.calibration_ablation = mode
        self.evidence_indices = CALIBRATION_ABLATIONS[mode]
        super().__init__(config, dataloader)
        self.logger.info(
            "[V3 CALIBRATION ABLATION] mode=%s evidence=%s",
            mode,
            [EVIDENCE_NAMES[index] for index in self.evidence_indices],
        )

    def _selected_features(self, features):
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != len(EVIDENCE_NAMES):
            raise ValueError("Dynamic evidence must contain exactly three channels.")
        return np.take(values, self.evidence_indices, axis=-1)

    def _calibration_threshold(self, confidence, labels):
        valid_rows = self.overlap_split_codes == 1
        test_rows = self.overlap_split_codes == 2
        valid_scores = confidence[valid_rows].reshape(-1)
        valid_labels = labels[valid_rows].reshape(-1)
        quantile_threshold = float(
            np.quantile(valid_scores, self.calibration_quantile)
        )
        precision_threshold = self._precision_threshold(
            valid_scores,
            valid_labels,
            self.minimum_validation_precision,
            int(config_scalar(self.config.get("dynamic_min_validation_hits"), 3)),
        )
        threshold = max(self.minimum_confidence, quantile_threshold)
        if precision_threshold is not None:
            threshold = max(threshold, precision_threshold)
        return {
            "threshold": threshold,
            "quantile_threshold": quantile_threshold,
            "precision_threshold": precision_threshold,
            "valid_ap": average_precision(valid_scores, valid_labels),
            "test_ap": average_precision(
                confidence[test_rows].reshape(-1),
                labels[test_rows].reshape(-1),
            ),
            "valid_brier": float(np.mean((valid_scores - valid_labels) ** 2)),
        }

    def _fit_dynamic_calibrator(self, features, labels, epoch_idx):
        selected = self._selected_features(features)
        train_rows = self.overlap_split_codes == 0
        valid_rows = self.overlap_split_codes == 1
        train_x = selected[train_rows].reshape(-1, selected.shape[-1])
        valid_x = selected[valid_rows].reshape(-1, selected.shape[-1])
        train_y = labels[train_rows].reshape(-1)
        valid_y = labels[valid_rows].reshape(-1)
        train_positive = float(train_y.sum())
        valid_positive = float(valid_y.sum())
        if train_positive <= 0 or valid_positive <= 0:
            raise RuntimeError("Dynamic calibration split contains no positive hit.")

        if self.calibration_ablation == "mean_all":
            checkpoint = {
                "mode": self.calibration_ablation,
                "evidence_indices": self.evidence_indices,
            }
            confidence = self._predict_confidence(checkpoint, features)
            diagnostics = self._calibration_threshold(confidence, labels)
            self._log_calibration(epoch_idx, None, diagnostics)
            return checkpoint, diagnostics["threshold"]

        mean = train_x.mean(axis=0).astype(np.float32)
        std = np.maximum(train_x.std(axis=0), 1e-6).astype(np.float32)
        train_x = (train_x - mean) / std
        valid_x = (valid_x - mean) / std
        positive_weight = min(
            float(config_scalar(self.config.get("dynamic_positive_weight_cap"), 100.0)),
            max(1.0, (len(train_y) - train_positive) / train_positive),
        )
        correction = float(math.log(positive_weight))
        torch.manual_seed(self.seed + 4049 + int(epoch_idx))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + 4049 + int(epoch_idx))

        model = PositiveLinearCalibrator(train_x.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(
                config_scalar(
                    self.config.get("dynamic_calibrator_learning_rate"), 0.01
                )
            ),
            weight_decay=float(
                config_scalar(
                    self.config.get("dynamic_calibrator_weight_decay"), 0.0001
                )
            ),
        )
        batch_size = int(
            config_scalar(self.config.get("dynamic_calibrator_batch_size"), 65536)
        )
        generator = np.random.default_rng(self.seed + 5051 + int(epoch_idx))
        positive_weight_t = torch.tensor(positive_weight, device=self.device)
        valid_x_t = torch.as_tensor(valid_x, dtype=torch.float32, device=self.device)
        valid_y_t = torch.as_tensor(valid_y, dtype=torch.float32, device=self.device)
        best_loss = float("inf")
        best_state = None
        stale = 0
        patience = int(
            config_scalar(self.config.get("dynamic_calibrator_patience"), 8)
        )
        minimum_delta = float(
            config_scalar(self.config.get("dynamic_calibrator_minimum_delta"), 1e-6)
        )
        maximum_epochs = int(
            config_scalar(self.config.get("dynamic_calibrator_max_epochs"), 60)
        )
        for _calibration_epoch in range(maximum_epochs):
            model.train()
            order = generator.permutation(len(train_y))
            for start in range(0, len(train_y), batch_size):
                indices = order[start : start + batch_size]
                x = torch.as_tensor(
                    train_x[indices], dtype=torch.float32, device=self.device
                )
                y = torch.as_tensor(
                    train_y[indices], dtype=torch.float32, device=self.device
                )
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    model(x), y, pos_weight=positive_weight_t
                )
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(
                    F.binary_cross_entropy_with_logits(
                        model(valid_x_t), valid_y_t, pos_weight=positive_weight_t
                    ).cpu()
                )
            if valid_loss < best_loss - minimum_delta:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is None:
            raise RuntimeError("Dynamic calibrator did not produce a checkpoint.")
        model.load_state_dict(best_state)
        checkpoint = {
            "mode": self.calibration_ablation,
            "evidence_indices": self.evidence_indices,
            "state_dict": best_state,
            "mean": mean,
            "std": std,
            "correction": correction,
            "weights": model.weights.detach().cpu().numpy(),
            "bias": float(model.bias.detach().cpu()),
        }
        confidence = self._predict_confidence(checkpoint, features)
        diagnostics = self._calibration_threshold(confidence, labels)
        self._log_calibration(epoch_idx, checkpoint, diagnostics)
        return checkpoint, diagnostics["threshold"]

    def _log_calibration(self, epoch_idx, checkpoint, diagnostics):
        if checkpoint is None:
            weights = [round(1.0 / len(self.evidence_indices), 4)] * len(
                self.evidence_indices
            )
            bias = 0.0
        else:
            weights = np.round(checkpoint["weights"], 4).tolist()
            bias = checkpoint["bias"]
        precision_threshold = diagnostics["precision_threshold"]
        self.logger.info(
            "[V3 CALIBRATION ABLATION] epoch=%d mode=%s evidence=%s "
            "weights=%s bias=%.5f valid_AP=%.6f test_AP=%.6f "
            "valid_Brier=%.6f threshold=%.6f (quantile=%.6f precision=%s)",
            epoch_idx,
            self.calibration_ablation,
            [EVIDENCE_NAMES[index] for index in self.evidence_indices],
            weights,
            bias,
            diagnostics["valid_ap"],
            diagnostics["test_ap"],
            diagnostics["valid_brier"],
            diagnostics["threshold"],
            diagnostics["quantile_threshold"],
            (
                "none"
                if precision_threshold is None
                else "{:.6f}".format(precision_threshold)
            ),
        )

    def _predict_confidence(self, checkpoint, features):
        selected = np.take(
            np.asarray(features, dtype=np.float32),
            checkpoint["evidence_indices"],
            axis=-1,
        )
        if checkpoint["mode"] == "mean_all":
            return selected.mean(axis=-1, dtype=np.float32)

        shape = selected.shape[:-1]
        values = (
            selected.reshape(-1, selected.shape[-1]) - checkpoint["mean"]
        ) / checkpoint["std"]
        model = PositiveLinearCalibrator(selected.shape[-1]).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        output = np.empty(len(values), dtype=np.float32)
        batch_size = int(
            config_scalar(
                self.config.get("dynamic_calibrator_inference_batch_size"), 262144
            )
        )
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                tensor = torch.as_tensor(
                    values[start:stop], dtype=torch.float32, device=self.device
                )
                output[start:stop] = torch.sigmoid(
                    model(tensor) - checkpoint["correction"]
                ).cpu().numpy()
        return output.reshape(shape)


class GroundedExpandableCalibrationAblationMixin(GroundedExpandablePseudoMixin):
    """Use the calibration-ablation supervisor in the unchanged v3 loop."""

    def _init_grounded_expandable(self, config, dataloader):
        self.config = config
        self.pseudo_supervisor = CalibrationAblationExpandableSupervisor(
            config, dataloader
        )
        self.pseudo_supervisor.set_user_gate(
            self.adaptive_user_gate.detach().cpu().numpy()
        )
        object.__setattr__(self, "_dynamic_dataloader", dataloader)
        object.__setattr__(self, "_ema_teacher", None)
        self._ema_decay = float(
            config_scalar(config.get("ema_teacher_decay"), 0.995)
        )
        self._expansion_start_epoch = int(
            config_scalar(config.get("expansion_start_epoch"), 10)
        )
        self._expansion_interval = int(
            config_scalar(config.get("expansion_interval"), 10)
        )
        self._expansion_max_rounds = int(
            config_scalar(config.get("expansion_max_rounds"), 3)
        )
        if not 0.0 <= self._ema_decay < 1.0:
            raise ValueError("ema_teacher_decay must lie in [0, 1).")
        if self._expansion_interval <= 0:
            raise ValueError("expansion_interval must be positive.")
        if self._expansion_max_rounds <= 0:
            raise ValueError("expansion_max_rounds must be positive.")
