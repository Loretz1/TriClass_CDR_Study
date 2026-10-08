import csv
import json
import os

import numpy as np
import pandas as pd
import torch

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA as _BaseModel,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)
from utils.metrics import metrics_dict
from utils.utils import get_local_time


class PseudoQualityRecorderMixin:
    """Add pseudo-label quality logging without changing the model backbone."""

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        self._quality_epoch_idx = -1
        self._quality_batch_idx = -1
        self.quality_topk = int(config.get("pseudo_quality_topk", 20))
        self.metric_ks = [
            int(k)
            for k in config.get("pseudo_quality_metric_ks", [5, 10, 20])
            if int(k) <= self.quality_topk
        ]
        if self.quality_topk <= 0:
            raise ValueError("pseudo_quality_topk must be positive.")
        if not self.metric_ks:
            self.metric_ks = [self.quality_topk]

        self.save_epoch_topk = bool(config.get("pseudo_quality_save_epoch_topk", True))
        self.rebuild_count = 0
        self.latest_quality_source = "semantic_initial"
        self.latest_quality_topk = None
        self.latest_quality_metrics = {}
        self._prev_update_topk = None
        self._prev_epoch_topk = None

        self.quality_users = self._collect_quality_users(dataloader)
        self.quality_user_to_row = {u: idx for idx, u in enumerate(self.quality_users)}
        self.quality_gt = self._collect_quality_ground_truth(dataloader)
        self.quality_pos_items = [self.quality_gt[u] for u in self.quality_users]
        self.quality_pos_len = np.asarray([len(items) for items in self.quality_pos_items], dtype=np.int64)

        self.record_dir = self._build_record_dir(config)
        self.top20_dir = os.path.join(self.record_dir, "top20_epoch")
        os.makedirs(self.top20_dir, exist_ok=True)
        self.epoch_metrics_path = os.path.join(self.record_dir, "epoch_metrics.csv")
        self.update_metrics_path = os.path.join(self.record_dir, "update_metrics.csv")

        self._write_config_snapshot(config)
        self._write_metadata()

        self.latest_quality_topk = self._compute_semantic_topk_for_quality_users()
        self.latest_quality_metrics = self._evaluate_topk(self.latest_quality_topk)
        self._prev_update_topk = self.latest_quality_topk.copy()
        self._append_metrics(
            self.update_metrics_path,
            event="initial",
            epoch=-1,
            global_step=0,
            source="semantic_initial",
            metrics=self.latest_quality_metrics,
            change_rate=0.0,
        )

    @staticmethod
    def _jsonable(value):
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [PseudoQualityRecorderMixin._jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): PseudoQualityRecorderMixin._jsonable(v)
                    for k, v in value.items()}
        return value

    @staticmethod
    def _format_config_value(value):
        if isinstance(value, float):
            return str(value)
        return str(value)

    def _split_dir(self, config):
        split_dir = (
            f"WarmValid{self._format_config_value(config['warm_valid_ratio'])}_"
            f"WarmTest{self._format_config_value(config['warm_test_ratio'])}_"
            f"ColdValid{self._format_config_value(config['t_cold_valid'])}_"
            f"ColdTest{self._format_config_value(config['t_cold_test'])}_"
            f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
        )
        if config["only_overlap_users"]:
            split_dir += f'_{config["k_cores"]}cores'
        return split_dir

    def _dataset_path(self, config):
        return os.path.abspath(
            os.path.join(
                config["data_path"],
                config["dataset"],
                "+".join(config["domains"]),
                "only_overlap_users" if config["only_overlap_users"] else "all_users",
                self._split_dir(config),
            )
        )

    def _build_record_dir(self, config):
        root = config.get("pseudo_quality_record_dir", "pseudo_quality_records")
        if not os.path.isabs(root):
            root = os.path.abspath(root)
        name = (
            f"{config['model']}-{config['dataset']}-"
            f"{'+'.join(config['domains'])}-{get_local_time()}"
        )
        record_dir = os.path.join(root, name)
        os.makedirs(record_dir, exist_ok=False)
        return record_dir

    def _write_config_snapshot(self, config):
        if hasattr(config, "final_config_dict"):
            config_dict = config.final_config_dict
        else:
            config_dict = dict(config)
        path = os.path.join(self.record_dir, "config_snapshot.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._jsonable(config_dict), f, indent=2, ensure_ascii=False)

    def _write_metadata(self):
        metadata = {
            "record_dir": self.record_dir,
            "quality_topk": self.quality_topk,
            "metric_ks": self.metric_ks,
            "num_quality_users": len(self.quality_users),
            "num_target_items": self.num_items_tgt,
            "dataset_path": self._dataset_path(self.config),
            "quality_user_source": "valid_cold_users + test_cold_users",
            "quality_users": self.quality_users,
            "ground_truth": {str(u): self.quality_gt[u] for u in self.quality_users},
        }
        path = os.path.join(self.record_dir, "metadata.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _collect_quality_users(self, dataloader):
        all_users = dataloader.dataset.all_users
        users = list(all_users.get("valid_cold_users", [])) + list(all_users.get("test_cold_users", []))
        return [int(u) for u in users]

    def _collect_quality_ground_truth(self, dataloader):
        dataset = dataloader.dataset
        if "valid_cold_tgt" in getattr(dataset, "df", {}) and "test_cold_tgt" in getattr(dataset, "df", {}):
            frames = [dataset.df["valid_cold_tgt"], dataset.df["test_cold_tgt"]]
        elif hasattr(dataset, "valid_cold_tgt") and hasattr(dataset, "test_cold_tgt"):
            frames = [dataset.valid_cold_tgt, dataset.test_cold_tgt]
        else:
            dataset_path = self._dataset_path(self.config)
            frames = [
                pd.read_pickle(os.path.join(dataset_path, "valid_cold_tgt.pkl")),
                pd.read_pickle(os.path.join(dataset_path, "test_cold_tgt.pkl")),
            ]

        gt = {}
        for df in frames:
            if df is None or len(df) == 0:
                continue
            for uid, group in df.groupby("user"):
                gt[int(uid)] = sorted({int(i) for i in group["item"].values})

        missing = [u for u in self.quality_users if u not in gt or len(gt[u]) == 0]
        if missing:
            raise ValueError(
                f"Missing held-out target positives for {len(missing)} cold users in "
                f"{self._dataset_path(self.config)}."
            )
        return gt

    @torch.no_grad()
    def _compute_semantic_topk_for_quality_users(self):
        rows = np.zeros((len(self.quality_users), self.quality_topk), dtype=np.int32)
        if len(self.quality_users) == 0:
            return rows

        item_emb = self.tgt_item_text_emb[1:]
        k = min(self.quality_topk, item_emb.size(0))
        batch_size = int(self.config.get("pseudo_quality_eval_batch_size", 1024))

        for s in range(0, len(self.quality_users), batch_size):
            e = min(s + batch_size, len(self.quality_users))
            users = torch.tensor(self.quality_users[s:e], dtype=torch.long, device=self.user_text_emb.device)
            user_emb = self.user_text_emb[users]
            sim = torch.matmul(user_emb, item_emb.t())
            topk_idx = torch.topk(sim, k=k, dim=1).indices + 1
            rows[s:e, :k] = topk_idx.cpu().numpy().astype(np.int32)
        return rows

    def _evaluate_topk(self, topk_matrix):
        if len(self.quality_users) == 0:
            return {}

        bool_rec_matrix = []
        for gt_items, rec_items in zip(self.quality_pos_items, topk_matrix):
            gt_set = set(gt_items)
            bool_rec_matrix.append([item in gt_set for item in rec_items])
        bool_rec_matrix = np.asarray(bool_rec_matrix, dtype=bool)

        metrics = {}
        for metric_name in ["recall", "ndcg", "precision", "map"]:
            values = metrics_dict[metric_name](bool_rec_matrix, self.quality_pos_len)
            for k in self.metric_ks:
                metrics[f"{metric_name}@{k}"] = float(values[k - 1])
        return metrics

    def _topk_change_rate(self, previous, current):
        if previous is None or current is None or len(current) == 0:
            return 0.0
        return float(np.mean(np.any(previous[:, :self.quality_topk] != current[:, :self.quality_topk], axis=1)))

    @property
    def _csv_fieldnames(self):
        fields = [
            "event",
            "epoch",
            "global_step",
            "batch_idx",
            "rebuild_count",
            "source",
            "num_users",
            "quality_topk",
        ]
        for metric in ["recall", "ndcg", "precision", "map"]:
            for k in self.metric_ks:
                fields.append(f"{metric}@{k}")
        fields.append(f"change_rate@{self.quality_topk}")
        return fields

    def _append_csv_row(self, path, row):
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self._csv_fieldnames})

    def _append_metrics(self, path, event, epoch, global_step, source, metrics, change_rate):
        row = {
            "event": event,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_idx": int(self._quality_batch_idx),
            "rebuild_count": int(self.rebuild_count),
            "source": source,
            "num_users": len(self.quality_users),
            "quality_topk": self.quality_topk,
            f"change_rate@{self.quality_topk}": change_rate,
        }
        row.update({k: f"{v:.8f}" for k, v in metrics.items()})
        self._append_csv_row(path, row)

    @torch.no_grad()
    def rebuild_src_only_pseudo_with_ema(self):
        assert self.diff_tgt_ema is not None

        top_k = int(self.config["top_k_pos"])
        select_k = min(max(top_k, self.quality_topk), self.num_items_tgt)
        batch_size = int(self.config.get("pseudo_quality_rebuild_batch_size", 2048))

        u_start = self.num_users_overlap + 1
        u_end = self.num_users_src + 1

        item_emb = self.emb_item_tgt.weight[1:]

        new_pseudo_pos = {}
        new_pairs = []
        quality_topk = self.latest_quality_topk.copy()

        for s in range(u_start, u_end, batch_size):
            e = min(s + batch_size, u_end)
            users = torch.arange(s, e, device=self.device)

            u = self.emb_user(users)

            hist_src_items = self.history_src_user_src[users]
            hist_tgt_items = self.history_src_user_tgt[users]
            hist_src = self.emb_item_src(hist_src_items)
            hist_tgt = self.emb_item_tgt(hist_tgt_items)
            cond_src = self.src_interaction_agg(hist_src, u)
            cond_tgt = self.tgt_interaction_agg(hist_tgt, u)

            t = torch.zeros(users.size(0), dtype=torch.long, device=self.device)

            _, u_denoised = self.diff_tgt_ema.p_losses(
                x_start=u,
                t=t,
                cond_src=cond_src,
                cond_tgt=cond_tgt,
                loss_type="l2",
            )

            u_final = u + self.config["lambda_user_emb"] * u_denoised

            sim = torch.matmul(u_final, item_emb.t())
            topk_idx = torch.topk(sim, k=select_k, dim=1).indices + 1

            for i, uid in enumerate(range(s, e)):
                pseudo_items = set(topk_idx[i, :top_k].tolist())
                new_pseudo_pos[uid] = pseudo_items
                for it in pseudo_items:
                    new_pairs.append((uid, it))

                row = self.quality_user_to_row.get(uid)
                if row is not None:
                    quality_topk[row, :self.quality_topk] = (
                        topk_idx[i, :self.quality_topk].detach().cpu().numpy().astype(np.int32)
                    )

        self.src_only_pseudo_pos = new_pseudo_pos
        self.pseudo_ui_pairs = new_pairs

        self.rebuild_count += 1
        change_rate = self._topk_change_rate(self._prev_update_topk, quality_topk)
        self._prev_update_topk = quality_topk.copy()
        self.latest_quality_topk = quality_topk
        self.latest_quality_source = "ema_rebuild"
        self.latest_quality_metrics = self._evaluate_topk(self.latest_quality_topk)
        self._append_metrics(
            self.update_metrics_path,
            event="ema_rebuild",
            epoch=self._quality_epoch_idx,
            global_step=self.global_step,
            source=self.latest_quality_source,
            metrics=self.latest_quality_metrics,
            change_rate=change_rate,
        )

    def pre_batch_processing(self, epoch_idx, batch_idx):
        self._quality_epoch_idx = int(epoch_idx)
        self._quality_batch_idx = int(batch_idx)

    def calculate_loss(self, interaction, epoch_idx):
        self._quality_epoch_idx = int(epoch_idx)
        return super().calculate_loss(interaction, epoch_idx)

    def post_epoch_processing(self, epoch_idx):
        change_rate = self._topk_change_rate(self._prev_epoch_topk, self.latest_quality_topk)
        self._prev_epoch_topk = self.latest_quality_topk.copy()

        if self.save_epoch_topk:
            path = os.path.join(self.top20_dir, f"epoch_{int(epoch_idx):03d}.npy")
            np.save(path, self.latest_quality_topk.astype(np.int32))

        self._append_metrics(
            self.epoch_metrics_path,
            event="epoch_end",
            epoch=epoch_idx,
            global_step=self.global_step,
            source=self.latest_quality_source,
            metrics=self.latest_quality_metrics,
            change_rate=change_rate,
        )

        recall_key = f"recall@{self.quality_topk}" if self.quality_topk in self.metric_ks else f"recall@{self.metric_ks[-1]}"
        ndcg_key = f"ndcg@{self.quality_topk}" if self.quality_topk in self.metric_ks else f"ndcg@{self.metric_ks[-1]}"
        return (
            f"Pseudo quality saved to {self.record_dir} | "
            f"{recall_key}={self.latest_quality_metrics.get(recall_key, 0.0):.4f}, "
            f"{ndcg_key}={self.latest_quality_metrics.get(ndcg_key, 0.0):.4f}"
        )


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_pseudo_quality(
    PseudoQualityRecorderMixin,
    _BaseModel,
):
    """Cloth/Sport EMA LSDCDR with pseudo-label quality logging."""
