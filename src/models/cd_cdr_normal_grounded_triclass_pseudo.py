from models.cd_cdr_normal import CD_CDR_Normal
from models.grounded_triclass_pseudo import TriClassPseudoMixin


class CD_CDR_Normal_Grounded_TriClass_Pseudo(TriClassPseudoMixin, CD_CDR_Normal):
    triclass_backbone = "cd_cdr"
