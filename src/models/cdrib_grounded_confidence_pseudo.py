from models.cdrib import CDRIB
from models.grounded_confidence_pseudo import GroundedConfidencePseudoSupervisor
from models.target_pattern_pseudo import apply_model_domain_profile, config_scalar


class CDRIB_Grounded_Confidence_Pseudo(CDRIB):
    """CDRIB with confidence-weighted fixed pseudo interactions."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.pseudo_supervisor = GroundedConfidencePseudoSupervisor(
            config, dataloader
        )
        self._source_graph_output = None
        self._target_graph_output = None
        self._graph_hooks = [
            self.source_GNN.register_forward_hook(self._capture_source_graph),
            self.target_GNN.register_forward_hook(self._capture_target_graph),
        ]

    def _pseudo_weight(self):
        return float(config_scalar(self.config.get("pseudo_loss_weight"), 0.0))

    def _capture_source_graph(self, _module, _inputs, output):
        self._source_graph_output = output

    def _capture_target_graph(self, _module, _inputs, output):
        self._target_graph_output = output

    def _pseudo_loss(self):
        if self._source_graph_output is None or self._target_graph_output is None:
            raise RuntimeError(
                "CDRIB graph embeddings were not captured from the base loss."
            )
        users, positive_items, negative_items, confidence = (
            self.pseudo_supervisor.sample(self.config["pseudo_num_per_batch"])
        )
        source_learn_user = self._source_graph_output[0]
        target_learn_item = self._target_graph_output[1]
        return self.pseudo_supervisor.weighted_bpr_loss(
            source_learn_user[users - 1],
            target_learn_item[positive_items - 1],
            target_learn_item[negative_items - 1],
            confidence,
        )

    def calculate_loss(self, interaction, epoch_idx):
        self._source_graph_output = None
        self._target_graph_output = None
        loss = super().calculate_loss(interaction, epoch_idx)
        if epoch_idx < int(self.config["pseudo_start_epoch"]):
            return loss
        return loss + self._pseudo_weight() * self._pseudo_loss()
