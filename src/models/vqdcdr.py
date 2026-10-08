import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
from common.loss import BPRLoss
import numpy as np
import math
from abc import ABC, abstractmethod

class VQDCDR(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(VQDCDR, self).__init__(config, dataloader)
        self.config = config
        self.bpr_loss = BPRLoss()
        self.feature_dim = config["feature_dim"]
        self.maxlen_source = config["maxlen_source"]

        self.user_embeddings = uidEmbedding(self.num_users_overlap, self.feature_dim)
        self.item_embeddings = iidEmbedding(self.num_items_src + self.num_items_tgt, self.feature_dim)
        self.vqdnet = VQDNet(config)

        self.history_items_src = self._build_history(dataloader)
        self.apply(xavier_uniform_initialization)

    def _build_history(self, dataloader):
        device = self.device
        max_len = self.maxlen_source

        H_src = torch.zeros((self.num_users_src + 1, max_len), dtype=torch.long, device=device)
        for u, items in dataloader.dataset.positive_items_src.items():
            items = list(items)
            n = len(items)
            if n == 0:
                continue
            if n > max_len:
                # random truncation
                rand_idx = torch.randperm(n)[:max_len]
                sampled = torch.tensor([items[i] for i in rand_idx], dtype=torch.long, device=device)
                H_src[u] = sampled
            else:
                sampled = torch.tensor(items, dtype=torch.long, device=device)
                H_src[u, :n] = sampled

        return H_src

    def calculate_loss(self, interaction, epoch_idx):
        user = interaction['users']
        tgt_pos = interaction['pos_items_tgt'] + self.num_items_src
        tgt_neg = interaction['neg_items_tgt'] + self.num_items_src

        x_start = self.user_embeddings(user)
        items_src = self.item_embeddings(self.history_items_src[user])
        items_tgt_pos = self.item_embeddings(tgt_pos)
        items_tgt_neg = self.item_embeddings(tgt_neg)
        centroids, labels = self.vqdnet.get_codebook(items_src)
        mask = (self.history_items_src[user] > 0).float()

        pred_user, loss_diff = self.vqdnet.diffusion(x_start, items_src, mask, centroids, labels)

        # unique_user, inverse_idx = torch.unique( user, return_inverse=True)
        # first_occurrence = torch.zeros_like(user, dtype=torch.bool, device=self.device)
        # for u_idx in range(unique_user.size(0)):
        #     first_pos = (inverse_idx == u_idx).nonzero(as_tuple=False)[0]
        #     first_occurrence[first_pos] = True
        # pred_user_unique = pred_user[first_occurrence]
        # loss_contra = self.vqdnet.contra_loss(pred_user_unique)

        unique_user, inverse = torch.unique(user, return_inverse=True)
        perm = torch.arange(user.size(0), device=user.device)
        order = torch.argsort(inverse)
        inverse_sorted = inverse[order]
        perm_sorted = perm[order]
        first_pos = torch.ones_like(unique_user, dtype=torch.long) * -1
        mask = torch.ones_like(inverse_sorted, dtype=torch.bool)
        mask[1:] = inverse_sorted[1:] != inverse_sorted[:-1]
        first_pos[inverse_sorted[mask]] = perm_sorted[mask]
        first_idx = first_pos

        pred_user_unique = pred_user[first_idx]
        loss_contra = self.vqdnet.contra_loss(pred_user_unique)

        pos_score = (pred_user * items_tgt_pos).sum(dim=-1)
        neg_score = (pred_user * items_tgt_neg).sum(dim=-1)
        loss_rec = self.bpr_loss(pos_score, neg_score)

        loss = self.config['lambda_con'] * loss_contra + (1 - self.config['lambda_con']) * (loss_rec + loss_diff)
        # loss = self.config['lambda_con'] * loss_contra + (1 - self.config['lambda_con']) * (loss_rec)
        return loss

    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        B = users.shape[0]
        d = self.feature_dim

        x_t = torch.randn((B,d)).to(self.device)
        items_src = self.item_embeddings(self.history_items_src[users])
        centroids, labels = self.vqdnet.get_codebook(items_src)
        mask = (self.history_items_src[users] > 0).float()
        pred_user = self.vqdnet.inference(x_t, items_src, mask, centroids, labels)

        pad_idx = torch.tensor([0], device=self.device)
        tgt_idx = torch.arange(
            self.num_items_src + 1,
            self.num_items_src + self.num_items_tgt + 1,
            device=self.device
        )
        tgt_items = torch.cat([self.item_embeddings(pad_idx), self.item_embeddings(tgt_idx)], dim=0)

        scores = torch.matmul(pred_user, tgt_items.t())
        return scores



class VQDNet(nn.Module):
    def __init__(self, config):
        super(VQDNet, self).__init__()
        self.emb_dim = config['feature_dim']
        self.schedule_sampler_name = config['schedule_sampler_name']
        self.diffusion_steps = config['diffusion_steps']
        self.noise_schedule = config['noise_schedule']
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.num_timesteps = int(self.betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)
        self.time_embed = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim * 4), SiLU(),
                                        nn.Linear(self.emb_dim * 4, self.emb_dim))
        self.att = Transformer_rep(config)
        self.norm_diffu_rep = LayerNorm(self.emb_dim)
        self.loss_ce = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(config['dropout'])
        self.rescale_timesteps = config['rescale_timesteps']
        self.lambda_history = config['lambda_history']
        self.lambda_intent = config['lambda_intent']
        self.n_clusters = config['num_cluster']
        self.mlp = nn.Sequential(nn.Linear(self.emb_dim * config['maxlen_source'], self.n_clusters))

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        return betas

    def _scale_timesteps(self, t):
        if not self.rescale_timesteps:
            return t
        scale_factor = 1000.0 / self.num_timesteps
        return t.float() * scale_factor

    def q_posterior_mean_variance(self, x_start, u_t, t):
        assert x_start.shape == u_t.shape
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, u_t.shape) * x_start
                + extract_into_tensor(self.posterior_mean_coef2, t, u_t.shape) * u_t
        )
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, item_emb, u_t, t, mask, centroids, labels):
        model_output = self.denoise(u_t, t, item_emb, mask, centroids, labels)
        x_0 = model_output
        model_log_variance = extract_into_tensor(self.posterior_variance, t, u_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, u_t=u_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_emb, noise_u_t, t, mask, centroids, labels):
        model_mean, model_log_variance = self.p_mean_variance(item_emb, noise_u_t, t, mask, centroids, labels)
        noise = torch.randn_like(noise_u_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_u_t.shape) - 1))))
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample_xt

    def add_noise(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        u_t = (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )
        if mask == None:
            return u_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, u_t)

    def denoise(self, u_t, t, item_emb, mask, centroids, labels):
        t = self._scale_timesteps(t)
        emb_t = self.time_embed(self.timestep_embedding(t, self.emb_dim))
        u_t = u_t + emb_t
        res = self.att(item_emb * self.lambda_history + centroids[labels] * self.lambda_intent, mask)
        res = self.norm_diffu_rep(self.dropout(res))
        out = res[:, -1, :]
        return out

    def diffusion(self, u_0, item_emb, mask, centroids, labels):
        noise = torch.randn_like(u_0)
        t, weights = self.schedule_sampler.sample(item_emb.shape[0], u_0.device)
        u_t = self.add_noise(u_0, t, noise=noise)
        u_0_pred = self.denoise(u_t, t, item_emb, mask, centroids, labels)
        loss_diff = F.mse_loss(u_0, u_0_pred)
        return u_0_pred, loss_diff

    def inference(self, u_t, item_emb, mask, centroids, labels):
        device = item_emb.device
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * item_emb.shape[0], device=device)
            with torch.no_grad():
                u_t = self.p_sample(item_emb, u_t, t, mask, centroids, labels)
        return u_t

    def get_codebook(self, items):
        flattened_features = items.view(items.shape[0], -1)
        random_indices = torch.randperm(flattened_features.size(0))[:self.n_clusters]
        centers = flattened_features[random_indices].to(items.device)
        logits = self.mlp(flattened_features)
        hard_assignments = F.gumbel_softmax(logits, tau=0.1, hard=True)
        for cluster_idx in range(self.n_clusters):
            cluster_mask = hard_assignments[:, cluster_idx].bool()
            if torch.sum(cluster_mask) == 0:
                random_sample_idx = torch.randint(0, flattened_features.size(0), (1,))
                centers[cluster_idx] = flattened_features[random_sample_idx]
            else:
                current_cluster_samples = flattened_features[cluster_mask]
                new_center = torch.mean(current_cluster_samples, dim=0)
                centers[cluster_idx] = 0.2 * centers[cluster_idx] + 0.8 * new_center
        output_centers = centers.view(
            self.n_clusters,
            items.shape[1],
            items.shape[-1]
        )
        final_assignments = torch.argmax(hard_assignments, dim=1)
        return output_centers, final_assignments

    def contra_loss(self, embeddings):
        temp = 0.07

        cos_sim = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        ) / temp
        cos_sim = cos_sim.to(embeddings.device)
        pos_sim = torch.diag(cos_sim).to(embeddings.device)
        neg_mask = ~torch.eye(len(embeddings), dtype=torch.bool).to(embeddings.device)
        neg_sim = cos_sim[neg_mask].view(len(embeddings), -1).to(embeddings.device)
        contrastive_logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        targets = torch.zeros(len(embeddings), dtype=torch.long).to(embeddings.device)
        contrastive_loss = F.cross_entropy(contrastive_logits, targets)
        # contrastive_loss = F.nll_loss(F.log_softmax(logits), labels)
        # contrastive_loss = F.nll_loss(F.softmax(logits).log(), labels)
        return contrastive_loss

class uidEmbedding(torch.nn.Module):

    def __init__(self, uid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all + 1, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(self.uid_embedding(x))

class iidEmbedding(torch.nn.Module):

    def __init__(self, iid_all, emb_dim):
        super().__init__()

        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(self.iid_embedding(x))

def create_named_schedule_sampler(name, num_timesteps):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == "lossaware":
        return LossSecondMomentResampler(num_timesteps)  ## default setting
    elif name == "fixstep":
        return FixSampler(num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])

    def weights(self):
        return self._weights

class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """

class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [self.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([self.num_timesteps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

class FixSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        ###############################################################
        ### You can custome your own sampling weight of steps here. ###
        ###############################################################
        self._weights = np.concatenate([np.ones([num_timesteps // 2]), np.zeros([num_timesteps // 2]) + 0.5])

    def weights(self):
        return self._weights

class Transformer_rep(nn.Module):
    def __init__(self, config):
        super(Transformer_rep, self).__init__()
        self.emb_dim = config['feature_dim']
        self.heads = 4
        self.dropout = config['dropout']
        self.n_blocks = config['num_blocks']
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.emb_dim, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))

class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
