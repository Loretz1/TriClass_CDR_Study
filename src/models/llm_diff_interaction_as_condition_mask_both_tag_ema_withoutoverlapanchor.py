import torch

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_WithoutOverlapAnchor(
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA
):
    """Keep overlap diffusion and pseudo ranking, but remove overlap BPR."""

    @torch.no_grad()
    def rebuild_src_only_pseudo_with_ema(self):
        """Run the canonical rebuild and retain its ranking for passive recorders."""
        assert self.diff_tgt_ema is not None

        top_k = int(self.config["top_k_pos"])
        batch_size = 2048
        user_start = self.num_users_overlap + 1
        user_end = self.num_users_src + 1
        item_emb = self.emb_item_tgt.weight[1:]

        new_pseudo_pos = {}
        new_pairs = []
        rebuild_rankings = {}

        for start in range(user_start, user_end, batch_size):
            end = min(start + batch_size, user_end)
            users = torch.arange(start, end, device=self.device)
            user_emb = self.emb_user(users)

            hist_src = self.emb_item_src(self.history_src_user_src[users])
            hist_tgt = self.emb_item_tgt(self.history_src_user_tgt[users])
            cond_src = self.src_interaction_agg(hist_src, user_emb)
            cond_tgt = self.tgt_interaction_agg(hist_tgt, user_emb)

            timesteps = torch.zeros(users.size(0), dtype=torch.long, device=self.device)
            _, user_denoised = self.diff_tgt_ema.p_losses(
                x_start=user_emb,
                t=timesteps,
                cond_src=cond_src,
                cond_tgt=cond_tgt,
                loss_type="l2",
            )
            user_final = user_emb + self.config["lambda_user_emb"] * user_denoised
            similarity = torch.matmul(user_final, item_emb.T)
            topk_idx = torch.topk(similarity, k=top_k, dim=1).indices + 1

            for row, user_id in enumerate(range(start, end)):
                ranked_items = topk_idx[row].tolist()
                rebuild_rankings[user_id] = tuple(ranked_items)
                items = set(ranked_items)
                new_pseudo_pos[user_id] = items
                for item_id in items:
                    new_pairs.append((user_id, item_id))

        self.src_only_pseudo_pos = new_pseudo_pos
        self.pseudo_ui_pairs = new_pairs
        self._last_rebuild_rankings = rebuild_rankings

    def calculate_loss(self, interaction, epoch_idx):
        use_pseudo = epoch_idx >= self.config["pseudo_start_epoch"]

        if (
            use_pseudo
            and self.diff_tgt_ema is not None
            and self.global_step > 0
            and self.global_step % self.config["pseudo_update_interval"] == 0
        ):
            self.rebuild_src_only_pseudo_with_ema()

        users_src = interaction["users_src"]
        pos_items_src = interaction["pos_items_src"]
        neg_items_src = interaction["neg_items_src"]
        users_tgt = interaction["users_tgt"]
        pos_items_tgt = interaction["pos_items_tgt"]
        neg_items_tgt = interaction["neg_items_tgt"]

        user_src = self.emb_user(users_src)
        item_pos_src = self.emb_item_src(pos_items_src)
        item_neg_src = self.emb_item_src(neg_items_src)

        hist_src_items = self.history_src_user_src[users_src]
        hist_tgt_items = self.history_src_user_tgt[users_src]
        hist_src_items = self.batch_random_mask(
            hist_src_items, self.config["mask_rate"], min_keep=1
        )
        hist_tgt_items = self.batch_random_mask(
            hist_tgt_items, self.config["mask_rate"], min_keep=1
        )
        hist_src = self.emb_item_src(hist_src_items)
        hist_tgt = self.emb_item_tgt(hist_tgt_items)
        cond_src = self.src_interaction_agg(hist_src, user_src)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, user_src)

        batch_size = user_src.size(0)
        timesteps_src = torch.randint(
            low=0,
            high=self.diff_src.timesteps,
            size=(batch_size,),
            device=user_src.device,
        )
        diff_loss_src, user_src_denoised = self.diff_src.p_losses(
            x_start=user_src,
            t=timesteps_src,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            loss_type="l2",
        )
        user_src_final = (
            user_src + self.config["lambda_user_emb"] * user_src_denoised
        )
        pos_score_src = (user_src_final * item_pos_src).sum(dim=-1)
        neg_score_src = (user_src_final * item_neg_src).sum(dim=-1)
        bpr_loss_src = self.bpr_loss(pos_score_src, neg_score_src)

        offset = self.num_users_src - self.num_users_overlap
        users_tgt_global = users_tgt + (
            users_tgt > self.num_users_overlap
        ).long() * offset
        user_tgt = self.emb_user(users_tgt_global)
        item_pos_tgt = self.emb_item_tgt(pos_items_tgt)
        item_neg_tgt = self.emb_item_tgt(neg_items_tgt)

        hist_src_items = self.history_tgt_user_src[users_tgt]
        hist_tgt_items = self.history_tgt_user_tgt[users_tgt]
        hist_src_items = self.batch_random_mask(
            hist_src_items, self.config["mask_rate"], min_keep=1
        )
        hist_tgt_items = self.batch_random_mask(
            hist_tgt_items, self.config["mask_rate"], min_keep=1
        )
        hist_src = self.emb_item_src(hist_src_items)
        hist_tgt = self.emb_item_tgt(hist_tgt_items)
        cond_src = self.src_interaction_agg(hist_src, user_tgt)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, user_tgt)

        batch_size = user_tgt.size(0)
        timesteps_tgt = torch.randint(
            low=0,
            high=self.diff_tgt.timesteps,
            size=(batch_size,),
            device=user_tgt.device,
        )
        diff_loss_tgt, user_tgt_denoised = self.diff_tgt.p_losses(
            x_start=user_tgt,
            t=timesteps_tgt,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            loss_type="l2",
        )
        user_tgt_final = (
            user_tgt + self.config["lambda_user_emb"] * user_tgt_denoised
        )
        pos_score_tgt = (user_tgt_final * item_pos_tgt).sum(dim=-1)
        neg_score_tgt = (user_tgt_final * item_neg_tgt).sum(dim=-1)
        bpr_loss_tgt = self.bpr_loss(pos_score_tgt, neg_score_tgt)

        pseudo_loss = user_src.new_zeros(())
        if use_pseudo:
            num_pseudo = int(self.config["pseudo_num_per_batch"])
            indices = torch.randint(
                low=0,
                high=len(self.pseudo_ui_pairs),
                size=(num_pseudo,),
                device=self.device,
            )
            pseudo_pairs = [self.pseudo_ui_pairs[index] for index in indices.tolist()]
            pseudo_users = torch.tensor(
                [user for user, _ in pseudo_pairs], device=self.device
            )
            pseudo_items = torch.tensor(
                [item for _, item in pseudo_pairs], device=self.device
            )
            neg_items = self.sample_tgt_neg_excluding_pseudo(
                pseudo_users, self.num_items_tgt
            )

            user_pseudo = self.emb_user(pseudo_users)
            hist_src_items = self.history_src_user_src[pseudo_users]
            hist_tgt_items = self.history_src_user_tgt[pseudo_users]
            hist_src_items = self.batch_random_mask(
                hist_src_items, self.config["mask_rate"], min_keep=1
            )
            hist_tgt_items = self.batch_random_mask(
                hist_tgt_items, self.config["mask_rate"], min_keep=1
            )
            hist_src = self.emb_item_src(hist_src_items)
            hist_tgt = self.emb_item_tgt(hist_tgt_items)
            cond_src = self.src_interaction_agg(hist_src, user_pseudo)
            cond_tgt = self.tgt_interaction_agg(hist_tgt, user_pseudo)

            pseudo_batch_size = user_pseudo.size(0)
            timesteps_pseudo = torch.randint(
                low=0,
                high=self.diff_tgt.timesteps,
                size=(pseudo_batch_size,),
                device=user_pseudo.device,
            )
            _, user_pseudo_denoised = self.diff_tgt.p_losses(
                x_start=user_pseudo,
                t=timesteps_pseudo,
                cond_src=cond_src,
                cond_tgt=cond_tgt,
                loss_type="l2",
            )
            user_pseudo_final = (
                user_pseudo
                + self.config["lambda_user_emb"] * user_pseudo_denoised
            )
            item_pos_pseudo = self.emb_item_tgt(pseudo_items)
            item_neg_pseudo = self.emb_item_tgt(neg_items)
            pos_score = (user_pseudo_final * item_pos_pseudo).sum(dim=-1)
            neg_score = (user_pseudo_final * item_neg_pseudo).sum(dim=-1)
            pseudo_loss = self.bpr_loss(pos_score, neg_score)

        overlap_rec_weight = float(self.config.get("overlap_rec_weight", 0.0))
        loss = overlap_rec_weight * (bpr_loss_src + bpr_loss_tgt)
        loss = loss + self.diff_weight * (diff_loss_src + diff_loss_tgt)
        if use_pseudo:
            loss = loss + self.config["pseudo_rec_weight"] * pseudo_loss
        return loss
