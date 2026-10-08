import torch

from models.llm_diff_interaction_as_condition_mask_both_tag_ema import (
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA,
    extract_CrossDomain_semantics_both_tag_modality_data,
    generate_CrossDomain_semantics_both_tag_embs,
    generate_CrossDomain_semantics_both_tag_final_embs,
)


class LLM_Diff_interaction_as_condition_mask_both_tag_EMA_PseudoOnly(
    LLM_Diff_interaction_as_condition_mask_both_tag_EMA
):
    """Train LSDCDR primarily from cold-user pseudo interactions."""

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

        u_src = self.emb_user(users_src)
        i_pos_src = self.emb_item_src(pos_items_src)
        i_neg_src = self.emb_item_src(neg_items_src)

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
        cond_src = self.src_interaction_agg(hist_src, u_src)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_src)

        batch_size = u_src.size(0)
        t_src = torch.randint(
            low=0,
            high=self.diff_src.timesteps,
            size=(batch_size,),
            device=u_src.device,
        )
        diff_loss_src, u_src_denoised = self.diff_src.p_losses(
            x_start=u_src,
            t=t_src,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            loss_type="l2",
        )
        u_src_final = u_src + self.config["lambda_user_emb"] * u_src_denoised
        pos_score_src = (u_src_final * i_pos_src).sum(dim=-1)
        neg_score_src = (u_src_final * i_neg_src).sum(dim=-1)
        bpr_loss_src = self.bpr_loss(pos_score_src, neg_score_src)

        users_tgt_local = users_tgt
        offset = self.num_users_src - self.num_users_overlap
        users_tgt_global = users_tgt_local + (
            users_tgt_local > self.num_users_overlap
        ).long() * offset
        u_tgt = self.emb_user(users_tgt_global)
        i_pos_tgt = self.emb_item_tgt(pos_items_tgt)
        i_neg_tgt = self.emb_item_tgt(neg_items_tgt)

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
        cond_src = self.src_interaction_agg(hist_src, u_tgt)
        cond_tgt = self.tgt_interaction_agg(hist_tgt, u_tgt)

        batch_size = u_tgt.size(0)
        t_tgt = torch.randint(
            low=0,
            high=self.diff_tgt.timesteps,
            size=(batch_size,),
            device=u_tgt.device,
        )
        diff_loss_tgt, u_tgt_denoised = self.diff_tgt.p_losses(
            x_start=u_tgt,
            t=t_tgt,
            cond_src=cond_src,
            cond_tgt=cond_tgt,
            loss_type="l2",
        )
        u_tgt_final = u_tgt + self.config["lambda_user_emb"] * u_tgt_denoised
        pos_score_tgt = (u_tgt_final * i_pos_tgt).sum(dim=-1)
        neg_score_tgt = (u_tgt_final * i_neg_tgt).sum(dim=-1)
        bpr_loss_tgt = self.bpr_loss(pos_score_tgt, neg_score_tgt)

        pseudo_loss = u_src.new_zeros(())
        if use_pseudo:
            num_pseudo = self.config["pseudo_num_per_batch"]
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

            u_pseudo = self.emb_user(pseudo_users)
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
            cond_src = self.src_interaction_agg(hist_src, u_pseudo)
            cond_tgt = self.tgt_interaction_agg(hist_tgt, u_pseudo)

            pseudo_batch_size = u_pseudo.size(0)
            t_pseudo = torch.randint(
                low=0,
                high=self.diff_tgt.timesteps,
                size=(pseudo_batch_size,),
                device=u_pseudo.device,
            )
            _, u_pseudo_denoised = self.diff_tgt.p_losses(
                x_start=u_pseudo,
                t=t_pseudo,
                cond_src=cond_src,
                cond_tgt=cond_tgt,
                loss_type="l2",
            )
            u_pseudo_final = (
                u_pseudo + self.config["lambda_user_emb"] * u_pseudo_denoised
            )
            pos_emb = self.emb_item_tgt(pseudo_items)
            neg_emb = self.emb_item_tgt(neg_items)
            pos_score = (u_pseudo_final * pos_emb).sum(dim=-1)
            neg_score = (u_pseudo_final * neg_emb).sum(dim=-1)
            pseudo_loss = self.bpr_loss(pos_score, neg_score)

        overlap_loss = (
            bpr_loss_src
            + bpr_loss_tgt
            + self.diff_weight * (diff_loss_src + diff_loss_tgt)
        )
        loss = self.config["overlap_loss_weight"] * overlap_loss
        if use_pseudo:
            loss = loss + self.config["pseudo_rec_weight"] * pseudo_loss
        return loss
