from models.cd_cdr_normal_pseudo import CD_CDR_Normal_Pseudo_Base
from models.grounded_confidence_pseudo import GroundedConfidencePseudoSupervisor


class CD_CDR_Normal_Grounded_Confidence_Pseudo(CD_CDR_Normal_Pseudo_Base):
    """Corrected CD-CDR with confidence-weighted fixed pseudo pairs."""

    supervisor_class = GroundedConfidencePseudoSupervisor

    def _pseudo_loss(self):
        users, positive_items, negative_items, confidence = (
            self.pseudo_supervisor.sample(self.config["pseudo_num_per_batch"])
        )
        return self.pseudo_supervisor.weighted_bpr_loss(
            self._pseudo_user_embedding(users),
            self.item_emb_tgt(positive_items),
            self.item_emb_tgt(negative_items),
            confidence,
        )

