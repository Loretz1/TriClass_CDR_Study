import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


logger = logging.getLogger()


class LSDCDR_Pseudo_Quality_Base(LLM_Diff_interaction_as_condition_mask_both_tag_EMA):
    """LSDCDR with static pseudo-supervision quality diagnostics.

    This keeps the original diffusion backbone and inference path, but disables
    EMA/self-evolving pseudo-label rebuilding so the pseudo set quality is the
    only controlled variable.
    """

    pseudo_variant = "overlaponly"

    def __init__(self, config, dataloader):
        self._pseudo_quality_dataloader = dataloader
        super(LSDCDR_Pseudo_Quality_Base, self).__init__(config, dataloader)
        self._validate_pseudo_sets()
        self._log_pseudo_stats()

    def pre_epoch_processing(self, epoch_idx):
        return None

    def post_batch_processing(self, epoch_idx, batch_idx):
        return None

    def calculate_loss(self, interaction, epoch_idx):
        pseudo_weight = self._float_config("pseudo_rec_weight", 0.0)
        if len(self.pseudo_ui_pairs) > 0 and pseudo_weight > 0:
            return super(LSDCDR_Pseudo_Quality_Base, self).calculate_loss(interaction, epoch_idx)

        old_start = self.config["pseudo_start_epoch"]
        old_start_value = int(self._scalar(old_start))
        if epoch_idx >= old_start_value:
            self.config["pseudo_start_epoch"] = int(epoch_idx) + 1
        try:
            return super(LSDCDR_Pseudo_Quality_Base, self).calculate_loss(interaction, epoch_idx)
        finally:
            self.config["pseudo_start_epoch"] = old_start

    @staticmethod
    def _scalar(value):
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def _int_config(self, key, default):
        return int(self._scalar(self.config.get(key, default)))

    def _float_config(self, key, default):
        return float(self._scalar(self.config.get(key, default)))

    def _dataset_path(self):
        config = self.config
        split_dir = (
            f"WarmValid{config['warm_valid_ratio']}_"
            f"WarmTest{config['warm_test_ratio']}_"
            f"ColdValid{config['t_cold_valid']}_"
            f"ColdTest{config['t_cold_test']}_"
            f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
        )
        if config["only_overlap_users"]:
            split_dir += f"_{config['k_cores']}cores"
            user_dir = "only_overlap_users"
        else:
            user_dir = "all_users"
        return os.path.join(
            config["data_path"],
            config["dataset"],
            "+".join(config["domains"]),
            user_dir,
            split_dir,
        )

    def _load_cold_eval_frame(self, frame_name):
        base_path = self._dataset_path()
        path = os.path.join(base_path, f"{frame_name}.pkl")
        if not os.path.exists(path) and base_path.endswith("_shuffle"):
            path = os.path.join(base_path[: -len("_shuffle")], f"{frame_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cold evaluation split: {path}")
        return pd.read_pickle(path)

    def _build_cold_user_true_target_sets(self):
        dataloader = self._pseudo_quality_dataloader
        true_pos = {}
        cold_users = list(dataloader.dataset.all_users.get("valid_cold_users", []))
        cold_users += list(dataloader.dataset.all_users.get("test_cold_users", []))
        for uid in cold_users:
            true_pos[int(uid)] = set()

        for frame_name in ["valid_cold_tgt", "test_cold_tgt"]:
            frame = self._load_cold_eval_frame(frame_name)
            for user, item in frame[["user", "item"]].values:
                true_pos.setdefault(int(user), set()).add(int(item))
        return true_pos

    def _target_train_items(self):
        dataloader = self._pseudo_quality_dataloader
        train_tgt = dataloader.dataset.df["train_tgt"]
        items = sorted({int(item) for item in train_tgt["item"].values if int(item) > 0})
        self._pseudo_candidate_size = len(items)
        return items

    def _rng(self):
        return random.Random(self._int_config("pseudo_seed", 999))

    def _pseudo_top_k(self):
        return self._int_config("pseudo_set_size", self.config.get("top_k_pos", 5))

    def _build_src_only_pseudo_positives(self):
        self.user_true_tgt_pos = self._build_cold_user_true_target_sets()
        self._pseudo_similarity_values = []

        variant = self.pseudo_variant
        if variant == "overlaponly":
            return {u: set() for u in self.user_true_tgt_pos}
        if variant == "exacthit":
            return self._build_exact_hit_pseudo()
        if variant == "random":
            return self._build_random_pseudo()
        if variant in {"preference_neighbor", "inconsistent"}:
            return self._build_similarity_pseudo(farthest=(variant == "inconsistent"))
        raise ValueError(f"Unsupported pseudo quality variant: {variant}")

    def _build_exact_hit_pseudo(self):
        rng = self._rng()
        top_k = self._pseudo_top_k()
        pseudo_pos = {}
        for uid, true_items in self.user_true_tgt_pos.items():
            items = sorted(true_items)
            if len(items) > top_k:
                items = sorted(rng.sample(items, top_k))
            pseudo_pos[uid] = set(items)
        return pseudo_pos

    def _build_random_pseudo(self):
        rng = self._rng()
        top_k = self._pseudo_top_k()
        candidate_items = self._target_train_items()
        pseudo_pos = {}
        for uid, true_items in self.user_true_tgt_pos.items():
            candidates = [item for item in candidate_items if item not in true_items]
            if len(candidates) > top_k:
                candidates = rng.sample(candidates, top_k)
            pseudo_pos[uid] = set(candidates)
        return pseudo_pos

    def _cache_dir(self):
        path = os.path.join(self._dataset_path(), "pseudo_quality_cache")
        os.makedirs(path, exist_ok=True)
        return path

    def _target_mf_cache_path(self):
        dim = self._int_config("target_mf_embedding_dim", self.embedding_dim)
        epochs = self._int_config("target_mf_epochs", 50)
        lr = self._float_config("target_mf_lr", 0.001)
        seed = self._int_config("pseudo_seed", 999)
        name = f"target_mf_item_emb_dim{dim}_epochs{epochs}_lr{lr}_seed{seed}.npy"
        return os.path.join(self._cache_dir(), name)

    def _save_rng_state(self):
        state = {
            "torch_cpu": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        torch.random.set_rng_state(state["torch_cpu"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _sample_target_mf_negatives(self, users, positive_items, rng):
        neg_items = []
        for user in users:
            forbid = positive_items.get(int(user), set())
            neg = None
            for _ in range(100):
                cand = int(rng.integers(1, self.num_items_tgt + 1))
                if cand not in forbid:
                    neg = cand
                    break
            if neg is None:
                neg = int(rng.integers(1, self.num_items_tgt + 1))
            neg_items.append(neg)
        return neg_items

    def _load_or_train_target_mf_item_embeddings(self):
        cache_path = self._target_mf_cache_path()
        if os.path.exists(cache_path):
            logger.info("[LSDCDR_PSEUDO_QUALITY] Loading target-only MF item embeddings: %s", cache_path)
            return np.load(cache_path)

        logger.info("[LSDCDR_PSEUDO_QUALITY] Training target-only MF item embeddings: %s", cache_path)
        rng_state = self._save_rng_state()
        try:
            item_emb = self._train_target_mf_item_embeddings()
        finally:
            self._restore_rng_state(rng_state)

        np.save(cache_path, item_emb.astype(np.float32))
        logger.info("[LSDCDR_PSEUDO_QUALITY] Saved target-only MF item embeddings: %s", cache_path)
        return item_emb.astype(np.float32)

    def _train_target_mf_item_embeddings(self):
        dataloader = self._pseudo_quality_dataloader
        train_tgt = dataloader.dataset.df["train_tgt"][["user", "item"]].copy()
        if train_tgt.empty:
            raise ValueError("train_tgt is empty; cannot train target-only MF item embeddings.")

        seed = self._int_config("pseudo_seed", 999)
        dim = self._int_config("target_mf_embedding_dim", self.embedding_dim)
        epochs = self._int_config("target_mf_epochs", 50)
        lr = self._float_config("target_mf_lr", 0.001)
        batch_size = self._int_config("target_mf_batch_size", 2048)
        weight_decay = self._float_config("target_mf_weight_decay", 0.0)
        rng = np.random.default_rng(seed)

        positive_items = {}
        for user, group in train_tgt.groupby("user"):
            positive_items[int(user)] = {int(item) for item in group["item"].values}

        user_arr = train_tgt["user"].astype(np.int64).to_numpy()
        pos_arr = train_tgt["item"].astype(np.int64).to_numpy()

        user_emb = nn.Embedding(self.num_users_tgt + 1, dim, padding_idx=0).to(self.device)
        item_emb = nn.Embedding(self.num_items_tgt + 1, dim, padding_idx=0).to(self.device)
        nn.init.xavier_uniform_(user_emb.weight)
        nn.init.xavier_uniform_(item_emb.weight)
        user_emb.weight.data[0].zero_()
        item_emb.weight.data[0].zero_()

        optimizer = torch.optim.Adam(
            list(user_emb.parameters()) + list(item_emb.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        n = len(user_arr)
        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                users = user_arr[idx]
                pos_items = pos_arr[idx]
                neg_items = self._sample_target_mf_negatives(users, positive_items, rng)

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=self.device)

                u = user_emb(users_t)
                pos = item_emb(pos_t)
                neg = item_emb(neg_t)
                loss = -torch.log(torch.sigmoid((u * pos).sum(dim=-1) - (u * neg).sum(dim=-1)) + 1e-12).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                num_batches += 1
            if epoch == 0 or (epoch + 1) == epochs or (epoch + 1) % 10 == 0:
                logger.info(
                    "[LSDCDR_PSEUDO_QUALITY] Target MF epoch %d/%d loss=%.6f",
                    epoch + 1,
                    epochs,
                    epoch_loss / max(num_batches, 1),
                )

        item_weight = item_emb.weight.detach().cpu().numpy().astype(np.float32)
        item_weight[0] = 0.0
        return item_weight

    def _build_similarity_pseudo(self, farthest):
        top_k = self._pseudo_top_k()
        candidate_items = self._target_train_items()
        item_emb_np = self._load_or_train_target_mf_item_embeddings()
        item_emb = torch.from_numpy(item_emb_np).float()
        item_emb = torch.nn.functional.normalize(item_emb, dim=-1)

        pseudo_pos = {}
        self._pseudo_similarity_values = []
        for uid, true_items in self.user_true_tgt_pos.items():
            candidates = [item for item in candidate_items if item not in true_items]
            anchors = [item for item in sorted(true_items) if 0 < item <= self.num_items_tgt]
            if not candidates or not anchors:
                pseudo_pos[uid] = set()
                continue

            cand_tensor = torch.tensor(candidates, dtype=torch.long)
            anchor_tensor = torch.tensor(anchors, dtype=torch.long)
            sim = torch.matmul(item_emb[cand_tensor], item_emb[anchor_tensor].t()).max(dim=1).values
            k = min(top_k, len(candidates))
            if farthest:
                selected_idx = torch.topk(-sim, k=k).indices
            else:
                selected_idx = torch.topk(sim, k=k).indices
            selected_items = [candidates[int(idx)] for idx in selected_idx.tolist()]
            pseudo_pos[uid] = set(selected_items)
            self._pseudo_similarity_values.extend(sim[selected_idx].tolist())
        return pseudo_pos

    def _validate_pseudo_sets(self):
        variant = self.pseudo_variant
        if variant == "overlaponly":
            return

        bad_users = []
        for uid, items in self.src_only_pseudo_pos.items():
            pseudo_items = set(items)
            true_items = self.user_true_tgt_pos.get(uid, set())
            hit_items = pseudo_items & true_items

            if variant == "exacthit":
                if pseudo_items - true_items:
                    bad_users.append((uid, sorted(pseudo_items - true_items)[:5]))
            elif hit_items:
                bad_users.append((uid, sorted(hit_items)[:5]))

        if bad_users:
            preview = bad_users[:5]
            if variant == "exacthit":
                raise ValueError(
                    "[LSDCDR_PSEUDO_QUALITY] ExactHit pseudo set contains non-ground-truth items: "
                    f"{preview}"
                )
            raise ValueError(
                f"[LSDCDR_PSEUDO_QUALITY] {variant} pseudo set contains held-out target hits: "
                f"{preview}"
            )

    def _log_pseudo_stats(self):
        cold_users = len(self.user_true_tgt_pos)
        total_pairs = sum(len(items) for items in self.src_only_pseudo_pos.values())
        empty_users = sum(1 for items in self.src_only_pseudo_pos.values() if len(items) == 0)
        hit_count = 0
        for uid, items in self.src_only_pseudo_pos.items():
            hit_count += len(set(items) & self.user_true_tgt_pos.get(uid, set()))
        avg_size = total_pairs / cold_users if cold_users else 0.0
        hit_rate = hit_count / total_pairs if total_pairs else 0.0
        candidate_size = getattr(self, "_pseudo_candidate_size", None)
        sim_values = getattr(self, "_pseudo_similarity_values", [])
        logger.info(
            "[LSDCDR_PSEUDO_QUALITY] variant=%s cold_users=%d total_pairs=%d avg_size=%.4f "
            "empty_users=%d hit_rate=%.4f candidate_size=%s avg_selected_similarity=%s no_ema=True",
            self.pseudo_variant,
            cold_users,
            total_pairs,
            avg_size,
            empty_users,
            hit_rate,
            "NA" if candidate_size is None else candidate_size,
            "NA" if not sim_values else f"{float(np.mean(sim_values)):.6f}",
        )

    def sample_tgt_neg_excluding_pseudo(self, users, num_items_tgt, max_retry=100):
        neg_items = []
        for u in users.tolist():
            forbid = set(self.src_only_pseudo_pos.get(int(u), set()))
            forbid.update(self.user_true_tgt_pos.get(int(u), set()))
            neg = None
            for _ in range(max_retry):
                cand = torch.randint(
                    low=1,
                    high=num_items_tgt + 1,
                    size=(1,),
                    device=self.device,
                ).item()
                if cand not in forbid:
                    neg = cand
                    break
            if neg is None:
                neg = cand
            neg_items.append(neg)
        return torch.tensor(neg_items, device=self.device)


class LSDCDR_Pseudo_Quality_OverlapOnly(LSDCDR_Pseudo_Quality_Base):
    pseudo_variant = "overlaponly"
