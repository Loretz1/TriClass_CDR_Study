import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss


class LightGCN_Both(GeneralRecommender):
    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)

        self.config = config
        self.feature_dim = config["feature_dim"]
        self.n_layers = config["n_layers"]
        self.alpha = config["alpha"]
        self.loss_type = str(config.get("loss_type", "bpr")).lower()

        self.num_users_all = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        self.num_items_all = self.num_items_src + self.num_items_tgt

        src_mat = dataloader.inter_matrix(domain=0, form="coo")  # [num_users_src, num_items_src]
        tgt_mat = dataloader.inter_matrix(domain=1, form="coo")  # [num_users_tgt, num_items_tgt]

        all_R = self.build_all_adj_from_src_tgt(
            src_mat=src_mat,
            tgt_mat=tgt_mat,
            num_users_src=self.num_users_src,
            num_users_tgt=self.num_users_tgt,
            num_items_src=self.num_items_src,
            num_items_tgt=self.num_items_tgt,
            num_overlap_user=self.num_users_overlap,
        )  # [U_all, I_all] coo

        all_R_norm = self.normalize_bipartite(all_R)  # D_u^-1/2 R D_i^-1/2
        self.all_adj = self._scipy_coo_to_torch(all_R_norm).to(self.device)  # sparse [U_all, I_all]

        # --- embeddings on global space ---
        self.user_embedding = nn.Embedding(self.num_users_all, self.feature_dim)
        self.item_embedding = nn.Embedding(self.num_items_all, self.feature_dim)

        # losses
        self.bpr_loss = BPRLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # init
        self.apply(xavier_uniform_initialization)

    @staticmethod
    def build_all_adj_from_src_tgt(
        src_mat: sp.coo_matrix,
        tgt_mat: sp.coo_matrix,
        num_users_src: int,
        num_users_tgt: int,
        num_items_src: int,
        num_items_tgt: int,
        num_overlap_user: int,
    ) -> sp.coo_matrix:
        """
        Build R_all (U_all x I_all) from src and tgt interaction matrices.

        Users:
          - overlap users: [0 .. Uo-1] aligned between src and tgt
          - tgt-only users: shifted by (Us - Uo)

        Items:
          - src items: [0 .. Is-1]
          - tgt items: shifted by Is
        """
        Us, Ut, Is, It, Uo = num_users_src, num_users_tgt, num_items_src, num_items_tgt, num_overlap_user
        U_all = Us + Ut - Uo
        I_all = Is + It

        src = src_mat.tocoo()
        tgt = tgt_mat.tocoo()

        R_all = sp.dok_matrix((U_all, I_all), dtype=np.float32)

        # src edges (u, i)
        for u, i, v in zip(src.row, src.col, src.data):
            R_all[int(u), int(i)] = float(v)

        # tgt edges with shifts
        user_shift = Us - Uo
        item_shift = Is
        for u, i, v in zip(tgt.row, tgt.col, tgt.data):
            u = int(u)
            i = int(i)
            if u < Uo:
                u_all = u
            else:
                u_all = u + user_shift
            i_all = i + item_shift
            R_all[u_all, i_all] = float(v)

        return R_all.tocoo()

    @staticmethod
    def normalize_bipartite(R: sp.coo_matrix) -> sp.coo_matrix:
        """
        Normalize bipartite interaction matrix:
          R_hat[u,i] = R[u,i] * (deg_u[u])^{-1/2} * (deg_i[i])^{-1/2}
        """
        R = R.tocoo()
        du = np.asarray(R.sum(axis=1)).flatten()  # [U]
        di = np.asarray(R.sum(axis=0)).flatten()  # [I]
        du[du == 0.0] = 1.0
        di[di == 0.0] = 1.0
        du_inv_sqrt = 1.0 / np.sqrt(du)
        di_inv_sqrt = 1.0 / np.sqrt(di)
        val = R.data * du_inv_sqrt[R.row] * di_inv_sqrt[R.col]
        return sp.coo_matrix((val, (R.row, R.col)), shape=R.shape)

    @staticmethod
    def _scipy_coo_to_torch(mat: sp.coo_matrix) -> torch.Tensor:
        mat = mat.tocoo()
        indices = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        values = torch.from_numpy(mat.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape))

    def compute_embeddings(self):
        """
        Returns:
            user_final: [U_all, d]
            item_final: [I_all, d]
        """
        u0 = self.user_embedding.weight
        v0 = self.item_embedding.weight

        user_list = [u0]
        item_list = [v0]

        for _ in range(self.n_layers):
            # user <- R * item
            u_next = torch.sparse.mm(self.all_adj, item_list[-1])
            # item <- R^T * user
            v_next = torch.sparse.mm(self.all_adj.transpose(0, 1), user_list[-1])
            user_list.append(u_next)
            item_list.append(v_next)

        user_final = torch.mean(torch.stack(user_list, dim=0), dim=0)
        item_final = torch.mean(torch.stack(item_list, dim=0), dim=0)
        return user_final, item_final

    def map_tgt_users_to_all(self, users_tgt_0based: torch.Tensor) -> torch.Tensor:
        """
        Map target-domain user ids (0-based in [0..num_users_tgt-1]) to global user ids (0-based).
        overlap users: unchanged
        tgt-only users: + (num_users_src - num_users_overlap)
        """
        Uo = self.num_users_overlap
        shift = self.num_users_src - self.num_users_overlap
        return torch.where(users_tgt_0based < Uo, users_tgt_0based, users_tgt_0based + shift)

    def calculate_loss(self, interaction, epoch_idx=None):
        # 1-based -> 0-based
        users_src = interaction["users_src"].long() - 1
        pos_items_src = interaction["pos_items_src"].long() - 1
        neg_items_src = interaction["neg_items_src"].long() - 1

        users_tgt = interaction["users_tgt"].long() - 1
        pos_items_tgt = interaction["pos_items_tgt"].long() - 1
        neg_items_tgt = interaction["neg_items_tgt"].long() - 1

        user_final, item_final = self.compute_embeddings()

        # SRC loss
        u_s = user_final[users_src]
        pos_s = item_final[pos_items_src]
        neg_s = item_final[neg_items_src]
        pos_scores_s = torch.sum(u_s * pos_s, dim=1)
        neg_scores_s = torch.sum(u_s * neg_s, dim=1)

        if self.loss_type == "bce":
            loss_src = self.bce_loss(pos_scores_s, torch.ones_like(pos_scores_s)) + \
                       self.bce_loss(neg_scores_s, torch.zeros_like(neg_scores_s))
        else:
            loss_src = self.bpr_loss(pos_scores_s, neg_scores_s)

        # TGT loss
        users_tgt_all = self.map_tgt_users_to_all(users_tgt)
        u_t = user_final[users_tgt_all]

        pos_t = item_final[pos_items_tgt + self.num_items_src]
        neg_t = item_final[neg_items_tgt + self.num_items_src]
        pos_scores_t = torch.sum(u_t * pos_t, dim=1)
        neg_scores_t = torch.sum(u_t * neg_t, dim=1)

        if self.loss_type == "bce":
            loss_tgt = self.bce_loss(pos_scores_t, torch.ones_like(pos_scores_t)) + \
                       self.bce_loss(neg_scores_t, torch.zeros_like(neg_scores_t))
        else:
            loss_tgt = self.bpr_loss(pos_scores_t, neg_scores_t)

        loss = self.alpha * loss_src + (1.0 - self.alpha) * loss_tgt
        return loss

    def full_sort_predict(self, interaction, is_warm=True):
        # [B] 1-based tgt user ids
        user_tgt = interaction[0].long() - 1
        user_all = self.map_tgt_users_to_all(user_tgt)

        user_final, item_final = self.compute_embeddings()
        u = user_final[user_all]  # [B, d]

        item_tgt = item_final[self.num_items_src: self.num_items_src + self.num_items_tgt]  # [It, d]

        scores = torch.matmul(u, item_tgt.t())  # [B, It]

        # padding column for item_id = 0
        pad = torch.zeros(scores.size(0), 1, device=scores.device)
        scores = torch.cat([pad, scores], dim=1)  # [B, It + 1]

        return scores

