from models.target_pattern_pseudo import (
    TargetPatternPseudoSupervisor,
    apply_model_domain_profile,
    config_scalar,
)
from models.cdrib import CDRIB


class CDRIB_Target_Pattern_Pseudo(CDRIB):
    """CDRIB with a shared target-pattern pseudo-interaction plugin."""

    def __init__(self, config, dataloader):
        apply_model_domain_profile(config)
        super().__init__(config, dataloader)
        self.pseudo_supervisor = None
        self._source_graph_output = None
        self._target_graph_output = None
        self._graph_hooks = []
        if self._pseudo_weight() > 0:
            self.pseudo_supervisor = TargetPatternPseudoSupervisor(config, dataloader)
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

        users, positive_items, negative_items = self.pseudo_supervisor.sample(
            self.config["pseudo_num_per_batch"]
        )
        source_learn_user = self._source_graph_output[0]
        target_learn_item = self._target_graph_output[1]
        return self.pseudo_supervisor.bpr_loss(
            source_learn_user[users - 1],
            target_learn_item[positive_items - 1],
            target_learn_item[negative_items - 1],
        )

    def calculate_loss(self, interaction, epoch_idx):
        self._source_graph_output = None
        self._target_graph_output = None
        loss = super().calculate_loss(interaction, epoch_idx)
        pseudo_weight = self._pseudo_weight()
        if pseudo_weight <= 0 or epoch_idx < int(self.config["pseudo_start_epoch"]):
            return loss
        return loss + pseudo_weight * self._pseudo_loss()
