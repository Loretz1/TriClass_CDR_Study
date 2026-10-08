from models.cd_cdr_normal_pseudo import CD_CDR_Normal_Pseudo_Base
from models.direct_semantic_pseudo import DirectSemanticPseudoSupervisor


class CD_CDR_Normal_Direct_Semantic_Pseudo(CD_CDR_Normal_Pseudo_Base):
    """Corrected CD-CDR with direct-semantic pseudo interactions."""

    supervisor_class = DirectSemanticPseudoSupervisor

