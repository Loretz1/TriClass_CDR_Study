from models.unicdr import UniCDR
from models.grounded_triclass_pseudo import TriClassPseudoMixin


class UniCDR_Grounded_TriClass_Pseudo(TriClassPseudoMixin, UniCDR):
    triclass_backbone = "unicdr"
