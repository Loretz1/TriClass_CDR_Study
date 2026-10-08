from models.emcdr import EMCDR
from models.grounded_triclass_pseudo import TriClassPseudoMixin


class EMCDR_Grounded_TriClass_Pseudo(TriClassPseudoMixin, EMCDR):
    triclass_backbone = "emcdr"
