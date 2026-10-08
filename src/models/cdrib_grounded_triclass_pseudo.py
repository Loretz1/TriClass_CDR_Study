from models.cdrib import CDRIB
from models.grounded_triclass_pseudo import TriClassPseudoMixin


class CDRIB_Grounded_TriClass_Pseudo(TriClassPseudoMixin, CDRIB):
    triclass_backbone = "cdrib"
