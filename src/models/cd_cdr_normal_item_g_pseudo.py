from models.cd_cdr_normal_pseudo import (
    ArchivedItemGPseudoSupervisor,
    CD_CDR_Normal_Pseudo_Base,
)


class CD_CDR_Normal_Item_G_Pseudo(CD_CDR_Normal_Pseudo_Base):
    """Corrected CD-CDR with the archived item-level G pseudo set."""

    supervisor_class = ArchivedItemGPseudoSupervisor
    pseudo_cache_subdirectory = "item_bridge_v1_archive"

