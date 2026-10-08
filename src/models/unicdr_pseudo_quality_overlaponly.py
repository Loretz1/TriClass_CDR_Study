import logging
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unicdr import UniCDR


logger = logging.getLogger()


class UniCDR_Pseudo_Quality_Base(UniCDR):
    """UniCDR pseudo-supervision quality diagnostic base."""

    pseudo_variant = "overlaponly"

    def __init__(self, config, dataloader):
        super(UniCDR_Pseudo_Quality_Base, self).__init__(config, dataloader)
        self.user_true_tgt_pos = self._build_cold_user_true_target_sets(dataloader)
        self.src_only_pseudo_pos = self._build_static_pseudo_positives(dataloader)
        self.pseudo_ui_pairs = self._flatten_pseudo_pairs(self.src_only_pseudo_pos)
        self._validate_pseudo_sets()
        self._log_pseudo_stats()

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
        return os.path.join(
            config["data_path"],
            config["dataset"],
            "+".join(config["domains"]),
            "only_overlap_users" if config["only_overlap_users"] else "all_users",
            split_dir,
        )

    def _load_cold_eval_frame(self, name):
        path = os.path.join(self._dataset_path(), f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cold evaluation split: {path}")
        return pd.read_pickle(path)

    def _build_cold_user_true_target_sets(self, dataloader):
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

    @staticmethod
    def _flatten_pseudo_pairs(pseudo_pos):
        return [(u, i) for u, items in pseudo_pos.items() for i in items]

    def _build_static_pseudo_positives(self, dataloader):
        variant = self.pseudo_variant
        if variant == "overlaponly":
            return {u: set() for u in self.user_true_tgt_pos}
        if variant == "exacthit":
            return self._build_exact_hit_pseudo()
        if variant == "random":
            return self._build_random_pseudo(dataloader)
        if variant in {"preference_neighbor", "inconsistent"}:
            return self._build_similarity_pseudo(dataloader, farthest=(variant == "inconsistent"))
        raise ValueError(f"Unsupported pseudo quality variant: {variant}")

    def _target_train_items(self, dataloader):
        train_tgt = dataloader.dataset.df["train_tgt"]
        items = sorted({int(item) for item in train_tgt["item"].values if int(item) > 0})
        self._pseudo_candidate_size = len(items)
        return items

    def _rng(self):
        return random.Random(int(self.config.get("pseudo_seed", 999)))

    def _build_exact_hit_pseudo(self):
        rng = self._rng()
        top_k = int(self.config.get("pseudo_set_size", 5))
        pseudo_pos = {}
        for uid, true_items in self.user_true_tgt_pos.items():
            items = sorted(true_items)
            if len(items) > top_k:
                items = sorted(rng.sample(items, top_k))
            pseudo_pos[uid] = set(items)
        return pseudo_pos

    def _build_random_pseudo(self, dataloader):
        rng = self._rng()
        top_k = int(self.config.get("pseudo_set_size", 5))
        candidate_items = self._target_train_items(dataloader)
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
        dim = int(self.config.get("target_mf_embedding_dim", self.feature_dim))
        epochs = int(self.config.get("target_mf_epochs", 50))
        lr = float(self.config.get("target_mf_lr", 0.001))
        seed = int(self.config.get("pseudo_seed", 999))
        name = f"target_mf_item_emb_dim{dim}_epochs{epochs}_lr{lr}_seed{seed}.npy"
        return os.path.join(self._cache_dir(), name)

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

    def _load_or_train_target_mf_item_embeddings(self, dataloader):
        cache_path = self._target_mf_cache_path()
        if os.path.exists(cache_path):
            logger.info("[PSEUDO_QUALITY] Loading target-only MF item embeddings: %s", cache_path)
            return np.load(cache_path)

        logger.info("[PSEUDO_QUALITY] Training target-only MF item embeddings: %s", cache_path)
        rng_state = self._save_rng_state()
        try:
            item_emb = self._train_target_mf_item_embeddings(dataloader)
        finally:
            self._restore_rng_state(rng_state)

        np.save(cache_path, item_emb.astype(np.float32))
        logger.info("[PSEUDO_QUALITY] Saved target-only MF item embeddings: %s", cache_path)
        return item_emb.astype(np.float32)

    def _train_target_mf_item_embeddings(self, dataloader):
        train_tgt = dataloader.dataset.df["train_tgt"][["user", "item"]].copy()
        if train_tgt.empty:
            raise ValueError("train_tgt is empty; cannot train target-only MF item embeddings.")

        seed = int(self.config.get("pseudo_seed", 999))
        dim = int(self.config.get("target_mf_embedding_dim", self.feature_dim))
        epochs = int(self.config.get("target_mf_epochs", 50))
        lr = float(self.config.get("target_mf_lr", 0.001))
        batch_size = int(self.config.get("target_mf_batch_size", 2048))
        weight_decay = float(self.config.get("target_mf_weight_decay", 0.0))
        rng = np.random.default_rng(seed)

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

        pairs = train_tgt.values.astype(np.int64)
        positive_items = dataloader.dataset.positive_items_tgt
        num_pairs = len(pairs)
        for epoch in range(epochs):
            order = rng.permutation(num_pairs)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, num_pairs, batch_size):
                idx = order[start:start + batch_size]
                batch = pairs[idx]
                users_np = batch[:, 0]
                pos_np = batch[:, 1]
                neg_np = self._sample_target_mf_negatives(users_np, positive_items, rng)

                users = torch.tensor(users_np, dtype=torch.long, device=self.device)
                pos_items = torch.tensor(pos_np, dtype=torch.long, device=self.device)
                neg_items = torch.tensor(neg_np, dtype=torch.long, device=self.device)

                u_emb = user_emb(users)
                pos_emb = item_emb(pos_items)
                neg_emb = item_emb(neg_items)
                pos_scores = (u_emb * pos_emb).sum(dim=-1)
                neg_scores = (u_emb * neg_emb).sum(dim=-1)
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                num_batches += 1
            if epoch == 0 or (epoch + 1) == epochs or (epoch + 1) % 10 == 0:
                logger.info(
                    "[PSEUDO_QUALITY] Target MF epoch %d/%d loss=%.6f",
                    epoch + 1,
                    epochs,
                    epoch_loss / max(num_batches, 1),
                )

        item_weight = item_emb.weight.detach().cpu().numpy().astype(np.float32)
        item_weight[0] = 0.0
        return item_weight

    def _build_similarity_pseudo(self, dataloader, farthest):
        top_k = int(self.config.get("pseudo_set_size", 5))
        candidate_items = self._target_train_items(dataloader)
        item_emb_np = self._load_or_train_target_mf_item_embeddings(dataloader)
        item_emb = torch.from_numpy(item_emb_np).float()
        item_emb = F.normalize(item_emb, dim=-1)

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
                    "[PSEUDO_QUALITY] ExactHit pseudo set contains non-ground-truth items: "
                    f"{preview}"
                )
            raise ValueError(
                f"[PSEUDO_QUALITY] {variant} pseudo set contains held-out target hits: "
                f"{preview}"
            )

    def _sample_tgt_neg_excluding_pseudo(self, users, max_retry=100):
        neg_items = []
        for u in users.tolist():
            forbid = set(self.src_only_pseudo_pos.get(u, set()))
            forbid.update(self.user_true_tgt_pos.get(u, set()))
            neg = None
            for _ in range(max_retry):
                cand = torch.randint(
                    low=1,
                    high=self.num_items_tgt + 1,
                    size=(1,),
                    device=self.device,
                ).item()
                if cand not in forbid:
                    neg = cand
                    break
            neg_items.append(cand if neg is None else neg)
        return torch.tensor(neg_items, dtype=torch.long, device=self.device)

    def _sample_pseudo_batch(self):
        num_pseudo = int(self.config["pseudo_num_per_batch"])
        idx = torch.randint(
            low=0,
            high=len(self.pseudo_ui_pairs),
            size=(num_pseudo,),
            device=self.device,
        )
        users, items = [], []
        for j in idx.tolist():
            u, i = self.pseudo_ui_pairs[j]
            users.append(u)
            items.append(i)
        users = torch.tensor(users, dtype=torch.long, device=self.device)
        items = torch.tensor(items, dtype=torch.long, device=self.device)
        neg_items = self._sample_tgt_neg_excluding_pseudo(users)
        return users, items, neg_items

    def _encode_cold_src_users(self, users):
        batch_size = users.size(0)
        global_user_emb = self.emb_user_shared(users)

        global_item = torch.zeros(
            (batch_size, 2, self.max_len),
            dtype=torch.long,
            device=self.device,
        )
        global_score = torch.full(
            (batch_size, 2, self.max_len),
            -100.0,
            dtype=torch.float,
            device=self.device,
        )
        global_item[:, 0, :] = self.history_items_src[users]
        global_score[:, 0, :] = self.history_scores_src[users]

        global_item_emb = torch.cat(
            [
                self.emb_item_src(global_item[:, 0, :]),
                self.emb_item_tgt(global_item[:, 1, :]),
            ],
            dim=1,
        )
        user_emb = self.agg_list[-1](global_user_emb, global_item_emb, global_score)
        return F.dropout(user_emb, self.dropout, training=self.training)

    def _pseudo_shared_loss(self):
        if len(self.pseudo_ui_pairs) == 0:
            return torch.zeros((), device=self.device)

        pseudo_users, pseudo_items, neg_items = self._sample_pseudo_batch()
        user_emb = self._encode_cold_src_users(pseudo_users)
        pos_emb = self.emb_item_tgt(pseudo_items)
        pos_emb = F.dropout(pos_emb, self.dropout, training=self.training)
        neg_emb = self.emb_item_tgt(neg_items)
        neg_emb = F.dropout(neg_emb, self.dropout, training=self.training)

        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

    def calculate_loss(self, interaction, epoch_idx):
        loss = super(UniCDR_Pseudo_Quality_Base, self).calculate_loss(interaction, epoch_idx)
        pseudo_weight = float(self.config["pseudo_rec_weight"])
        pseudo_start_epoch = int(self.config["pseudo_start_epoch"])
        if (
            self.pseudo_variant != "overlaponly"
            and pseudo_weight > 0
            and epoch_idx >= pseudo_start_epoch
        ):
            loss = loss + pseudo_weight * self._pseudo_shared_loss()
        return loss

    def _log_pseudo_stats(self):
        cold_users = len(self.user_true_tgt_pos)
        sizes = [len(items) for items in self.src_only_pseudo_pos.values()]
        total_pairs = sum(sizes)
        hit_count = 0
        for uid, items in self.src_only_pseudo_pos.items():
            hit_count += len(set(items) & self.user_true_tgt_pos.get(uid, set()))
        avg_size = total_pairs / cold_users if cold_users else 0.0
        empty_users = sum(1 for size in sizes if size == 0)
        hit_rate = hit_count / total_pairs if total_pairs else 0.0
        candidate_size = getattr(self, "_pseudo_candidate_size", None)
        sim_values = getattr(self, "_pseudo_similarity_values", [])
        logger.info(
            "[PSEUDO_QUALITY] variant=%s cold_users=%d pseudo_pairs=%d avg_set_size=%.4f "
            "empty_users=%d hit_rate=%.4f candidate_size=%s avg_selected_similarity=%s",
            self.pseudo_variant,
            cold_users,
            total_pairs,
            avg_size,
            empty_users,
            hit_rate,
            "NA" if candidate_size is None else candidate_size,
            "NA" if not sim_values else f"{float(np.mean(sim_values)):.6f}",
        )


class UniCDR_Pseudo_Quality_OverlapOnly(UniCDR_Pseudo_Quality_Base):
    pseudo_variant = "overlaponly"
