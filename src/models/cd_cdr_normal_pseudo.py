import json
import logging
import os

import numpy as np

from models.cd_cdr_normal import CD_CDR_Normal
from models.target_pattern_pseudo import (
    TargetPatternPseudoSupervisor,
    config_scalar,
    dataset_split_directory,
    mapping_fingerprint,
)


class ArchivedItemGPseudoSupervisor(TargetPatternPseudoSupervisor):
    """Read the immutable item-level G cache produced in format version 2."""

    def _load_and_validate(self, dataloader):
        if not os.path.isfile(self.cache_path):
            raise FileNotFoundError(
                "Archived item-level G cache does not exist: {}".format(
                    self.cache_path
                )
            )
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(
                "Archived item-level G manifest does not exist: {}".format(
                    self.manifest_path
                )
            )

        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = {
            "format_version": 2,
            "dataset": dataloader.dataset.config["dataset"],
            "domains": list(dataloader.dataset.config["domains"]),
            "num_users_overlap": self.num_users_overlap,
            "num_users_src": self.num_users_src,
            "num_items_tgt": self.num_items_tgt,
            "mapping_sha256": mapping_fingerprint(self.id_mapping),
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            detail = ", ".join(
                "{}: cache={!r}, expected={!r}".format(key, actual, wanted)
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(
                "Archived item-level G cache is incompatible with the active "
                "split ({}).".format(detail)
            )

        with np.load(self.cache_path, allow_pickle=False) as cache:
            user_ids = np.asarray(cache["user_ids"], dtype=np.int64)
            ranked_item_ids = np.asarray(cache["item_ids"], dtype=np.int64)
            ranked_scores = np.asarray(cache["scores"], dtype=np.float32)
        official_top_k = int(manifest["official_top_k"])
        if ranked_item_ids.ndim != 2 or ranked_scores.shape != ranked_item_ids.shape:
            raise ValueError(
                "Archived item-level G item_ids/scores must be equally shaped "
                "2-D arrays."
            )
        if len(user_ids) != ranked_item_ids.shape[0]:
            raise ValueError("Archived item-level G user rows are misaligned.")
        if official_top_k <= 0 or official_top_k > ranked_item_ids.shape[1]:
            raise ValueError("Archived item-level G Top-K is invalid.")
        if len(np.unique(user_ids)) != len(user_ids):
            raise ValueError("Archived item-level G cache has duplicate users.")
        if np.any(user_ids <= self.num_users_overlap) or np.any(
            user_ids > self.num_users_src
        ):
            raise ValueError(
                "Archived item-level G cache must contain cold source users only."
            )

        fixed_items = ranked_item_ids[:, :official_top_k]
        if np.any(fixed_items <= 0) or np.any(
            fixed_items > self.num_items_tgt
        ):
            raise ValueError("Archived item-level G cache has an invalid item id.")

        self.official_top_k = official_top_k
        self.user_ids = user_ids
        self.item_ids = fixed_items
        self.pair_users = np.repeat(user_ids, official_top_k)
        self.pair_items = fixed_items.reshape(-1)
        self.pseudo_positive = {
            int(user): set(map(int, items))
            for user, items in zip(user_ids.tolist(), fixed_items.tolist())
        }
        self.target_positive = self._target_training_positives(dataloader)
        logging.getLogger().info(
            "[NORMAL ITEM-G PSEUDO] Loaded %d users x Top-%d items from %s",
            len(self.user_ids),
            self.official_top_k,
            self.cache_path,
        )


class CD_CDR_Normal_Pseudo_Base(CD_CDR_Normal):
    """Shared direct pseudo-ranking path for the corrected CD-CDR core."""

    supervisor_class = TargetPatternPseudoSupervisor
    pseudo_cache_subdirectory = None

    def __init__(self, config, dataloader):
        if (
            self.pseudo_cache_subdirectory
            and config.get("pseudo_cache_path", "auto") in (None, "", "auto")
        ):
            config["pseudo_cache_path"] = os.path.join(
                dataset_split_directory(config),
                "target_pattern_pseudo",
                self.pseudo_cache_subdirectory,
                "pseudo_rankings.npz",
            )
        super().__init__(config, dataloader)
        self.pseudo_supervisor = None
        initial_weight = float(
            config_scalar(self.config.get("pseudo_loss_weight"), 0.0)
        )
        if initial_weight > 0:
            self.pseudo_supervisor = self.supervisor_class(config, dataloader)

    def _pseudo_weight(self):
        return float(config_scalar(self.config.get("pseudo_loss_weight"), 0.0))

    def _base_loss(self, interaction, epoch_idx):
        return CD_CDR_Normal.calculate_loss(self, interaction, epoch_idx)

    def _pseudo_user_embedding(self, users):
        return self.cold_user_condition(users)

    def _pseudo_loss(self):
        users, positive_items, negative_items = self.pseudo_supervisor.sample(
            self.config["pseudo_num_per_batch"]
        )
        return self.pseudo_supervisor.bpr_loss(
            self._pseudo_user_embedding(users),
            self.item_emb_tgt(positive_items),
            self.item_emb_tgt(negative_items),
        )

    def calculate_loss(self, interaction, epoch_idx):
        loss = self._base_loss(interaction, epoch_idx)
        pseudo_weight = self._pseudo_weight()
        if pseudo_weight <= 0 or epoch_idx < int(
            config_scalar(self.config.get("pseudo_start_epoch"), 0)
        ):
            return loss
        return loss + pseudo_weight * self._pseudo_loss()
