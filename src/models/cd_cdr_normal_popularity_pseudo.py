from models.cd_cdr_normal_pseudo import CD_CDR_Normal_Pseudo_Base
from models.popularity_pseudo import PopularityPseudoSupervisor


class CD_CDR_Normal_Popularity_Pseudo(CD_CDR_Normal_Pseudo_Base):
    """Corrected CD-CDR with popularity pseudo interactions."""

    supervisor_class = PopularityPseudoSupervisor
