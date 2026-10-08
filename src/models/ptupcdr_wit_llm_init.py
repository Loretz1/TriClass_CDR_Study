from models.llm_pseudo_ptupcdr import (
    LLM_Pseudo_PTUPCDR,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)
from models.ptupcdr import PTUPCDR


class PTUPCDR_wit_LLM_init(LLM_Pseudo_PTUPCDR):
    """Best PTUPCDR backbone with fixed LSDCDR semantic pseudo supervision."""

    def calculate_loss(self, interaction, epoch_idx):
        loss = PTUPCDR.calculate_loss(self, interaction, epoch_idx)
        pseudo_weight = float(self.config.get("pseudo_rec_weight", 0.0))
        if pseudo_weight <= 0:
            return loss
        if self.stage_id == 2 and epoch_idx >= int(self.config["pseudo_start_epoch"]):
            loss = loss + pseudo_weight * self._pseudo_meta_loss()
        return loss
