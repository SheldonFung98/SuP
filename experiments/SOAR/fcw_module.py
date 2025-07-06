import torch
import torch.nn as nn
from geotransformer.modules.ops.transformation import apply_transform
import torch
import torch.nn as nn
import torch.nn.functional as F


# class WeightingNet(nn.Module):

# 	def __init__(self, num_clusters=768, dim=768):
# 		super().__init__()
# 		self.num_clusters = num_clusters
# 		self.dim = dim
# 		self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)

# 		self.proj = nn.Sequential(
# 			nn.Linear(dim, 64),
# 			nn.PReLU(),
# 		)

# 	def forward(self, ref_feat, src_feat):

# 		if ref_feat.shape[0] == 0 or src_feat.shape[0] == 0:
# 			# no points → return empty tensor
# 			return None
# 		N, C = ref_feat.shape
# 		prior = ref_feat * src_feat
# 		post = self.conv(prior.T[None, :, :, None]).view(N, self.num_clusters, -1).squeeze(-1)
# 		o = prior + post
# 		out = self.proj(o)
# 		return out.max(dim=0, keepdim=True).values

class WeightingNet(nn.Module):

	def __init__(self, num_clusters=768, dim=768,
		dropout: float = 0.2,
	):
		super().__init__()
		self.num_clusters = num_clusters
		self.dim = dim
		# self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
		self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
		self.dropout = nn.Dropout(dropout)

		self.proj = nn.Sequential(
			nn.Linear(dim, 64),
			nn.GELU(),
		)

	def forward(self, ref_feat, src_feat):

		if ref_feat.shape[0] == 0 or src_feat.shape[0] == 0:
			# no points → return empty tensor
			return None

		prior = ref_feat * src_feat
		N, C = ref_feat.shape
		post = self.attn(prior, prior, prior)[0]
		o = prior + F.normalize(self.dropout(post))
		out = self.proj(o)
		pooled, _ = out.max(dim=0, keepdim=True)
		return pooled


# class WeightingNet(nn.Module):
# 	def __init__(
# 		self,
# 		in_dim: int = 768,
# 		out_dim: int = 64,
# 		num_clusters: int = 8,
# 		num_heads: int = 8,
# 		ff_multiplier: int = 4,
# 		dropout: float = 0.1,
# 	):
# 		super().__init__()
# 		self.in_dim       = in_dim
# 		self.out_dim      = out_dim
# 		self.num_clusters = num_clusters

# 		# 1×1 conv for soft‐assignment to K clusters
# 		# takes [1, in_dim, N] → [1, num_clusters, N]
# 		self.cluster_conv = nn.Conv1d(in_dim, num_clusters, kernel_size=1, bias=True)

# 		# Transformer‐style block over K cluster tokens
# 		self.norm1    = nn.LayerNorm(in_dim)
# 		self.attn     = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
# 		self.dropout1 = nn.Dropout(dropout)

# 		self.norm2    = nn.LayerNorm(in_dim)
# 		self.ffn      = nn.Sequential(
# 			nn.Linear(in_dim, in_dim * ff_multiplier),
# 			nn.GELU(),
# 			nn.Dropout(dropout),
# 			nn.Linear(in_dim * ff_multiplier, in_dim),
# 		)
# 		self.dropout2 = nn.Dropout(dropout)

# 		# final proj head to out_dim
# 		self.proj = nn.Sequential(
# 			nn.Linear(in_dim, out_dim),
# 			nn.GELU(),
# 		)

# 	def forward(self, ref_feat: torch.Tensor, src_feat: torch.Tensor) -> torch.Tensor:
# 		"""
# 		ref_feat, src_feat: [N, in_dim]
# 		→ returns [1, out_dim] (pooled over clusters) or [0, out_dim] if N=0
# 		"""
# 		N = ref_feat.size(0)
# 		if ref_feat.shape[0] == 0 or src_feat.shape[0] == 0:
# 			# no points → return empty tensor
# 			return None
# 		# 1) pointwise product
# 		prior = ref_feat * src_feat                   # [N, in_dim]

# 		# 2) soft‐assign clusters
# 		#    conv1d wants [B, C, L] = [1, in_dim, N]
# 		x = prior.transpose(0,1).unsqueeze(0)         # [1, in_dim, N]
# 		logits = self.cluster_conv(x)                 # [1, num_clusters, N]
# 		assign = F.softmax(logits, dim=1)             # softmax over cluster dim
# 													 # → [1, num_clusters, N]

# 		# 3) build K “cluster tokens” by weighted sum of points
# 		#    torch.matmul([1,K,N], [1,N,in_dim]) → [1,K,in_dim]
# 		cluster_tokens = torch.matmul(assign, prior.unsqueeze(0))

# 		# 4) Transformer block on these K tokens
# 		# 4a) Self‐attn with pre‐norm + residual
# 		y = self.norm1(cluster_tokens)               # [1, K, in_dim]
# 		attn_out, _ = self.attn(y, y, y)              # [1, K, in_dim]
# 		cluster_tokens = cluster_tokens + self.dropout1(attn_out)

# 		# 4b) FFN with post‐norm + residual
# 		y = self.norm2(cluster_tokens)               # [1, K, in_dim]
# 		ffn_out     = self.ffn(y)                     # [1, K, in_dim]
# 		cluster_tokens = cluster_tokens + self.dropout2(ffn_out)

# 		# 5) project each cluster‐token to out_dim
# 		#    → [1, K, out_dim]
# 		out_tokens = self.proj(cluster_tokens)

# 		# 6) max‐pool across clusters → [1, out_dim]
# 		pooled, _ = out_tokens.max(dim=1, keepdim=True)
# 		return pooled.squeeze(1)


class FeatureConsistencyWeighting(nn.Module):
	def __init__(self, feat_dim=256, hidden_dim=32, radius=0.03):
		"""
		feat_dim: dimensionality of per‑point features
		hidden_dim: hidden size for MLPs
		radius: neighborhood radius for local edges
		"""
		super().__init__()
		self.radius = radius
		self.max_points = 2**13
		self.wnet = WeightingNet()

		self.proj = nn.Sequential(
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

	def compute_rmse(self, est_transform, src_points, gt_transform):
		points = src_points[None, ...].repeat(est_transform.shape[0], 1, 1)
		realignment_transform = torch.matmul(torch.inverse(gt_transform), est_transform)
		realigned_src_points_f = apply_transform(points, realignment_transform)
		rmse = torch.linalg.norm(realigned_src_points_f - points, dim=-1).mean(dim=-1)
		return rmse
		
	def generate_training_tfs(self, src_points, tf_est, tf_gt, acceptance_rmse=0.2):
		rmses = self.compute_rmse(tf_est, src_points, tf_gt)

		success = rmses < acceptance_rmse
		failed = ~success
		reasonable = failed & (rmses < 0.3)

		success_inds = success.nonzero().flatten()
		failed_inds = failed.nonzero().flatten()
		reasonable_inds = reasonable.nonzero().flatten()

		num = 8
		sel_ind = torch.cat([
			success_inds[torch.randperm(success_inds.shape[0])][:num], 
			failed_inds[torch.randperm(failed_inds.shape[0])][:num],
			reasonable_inds[torch.randperm(reasonable_inds.shape[0])][:num]
		])

		return tf_est[sel_ind], sel_ind
	
	def downsample_pcd_feat(self, points, feats, num_points=1024):
		"""Downsample point cloud to a fixed number of points."""
		if points.shape[0] <= num_points:
			return points, feats
		# random sample indices
		indices = torch.randperm(points.shape[0])[:num_points]
		return points[indices], feats[indices]

	def get_weight_naive(self, m, batch_sizes, k=32):
		topks = [k if i > k else i for i in batch_sizes]
		w = torch.stack([i.topk(k=k, largest=False).values.mean() for k, i in zip(topks, m.split(batch_sizes))])
		# w = torch.stack([i.mean() for k, i in zip(topks, m.split(batch_sizes))])
		w -= w.min()  # remove negative weights
		if w.max() > 0:
			w /= w.max()  # normalize weights
		return  w

	def padding(self, sequences):
		"""Pad a list of tensors to the same size."""
		padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
		B = len(sequences)
		seq_lens = list(map(len, sequences))
		padding_mask = torch.zeros((B, padded.shape[1]), dtype=torch.bool, device=padded.device)
		for i, l in enumerate(seq_lens):
			padding_mask[i, l:] = True

		padding_lens = [seq.shape[0] for seq in sequences]

		return padded, padding_mask, padding_lens


	def get_weight(self, m, batch_sizes, k=32):
		return self.get_weight_naive(m, batch_sizes, k=k)

	def forward(self, 
		ref_pts_c, ref_pts_m, ref_pts_f, 
		src_pts_c, src_pts_m, src_pts_f, 
		ref_feats_m, ref_feats_f, 
		src_feats_m, src_feats_f,
		tf_est, tf_gt=None
	):
		sel_tf_ind = None
		# downsample points and features
		ref_pts_m, ref_feats_m = self.downsample_pcd_feat(ref_pts_m, ref_feats_m, num_points=self.max_points)
		ref_pts_f, ref_feats_f = self.downsample_pcd_feat(ref_pts_f, ref_feats_f, num_points=self.max_points)
		src_pts_m, src_feats_m = self.downsample_pcd_feat(src_pts_m, src_feats_m, num_points=self.max_points)
		src_pts_f, src_feats_f = self.downsample_pcd_feat(src_pts_f, src_feats_f, num_points=self.max_points)

		if self.training:
			# generate training transforms
			tf_est, sel_tf_ind = self.generate_training_tfs(src_pts_f, tf_est, tf_gt)

		B = tf_est.shape[0]
		ref_mf_knn_ind = torch.cdist(ref_pts_m, ref_pts_f).topk(1, dim=1, largest=False).indices.flatten()
		src_mf_knn_ind = torch.cdist(src_pts_m, src_pts_f).topk(1, dim=1, largest=False).indices.flatten()
		ref_feats_f_sel = ref_feats_f[ref_mf_knn_ind]
		src_feats_f_sel = src_feats_f[src_mf_knn_ind]

		dists = torch.cdist(ref_pts_m, apply_transform(src_pts_m[None, ...], tf_est))  # [N_ref, N_src]
		mask = dists < self.radius
		if not mask.any():
			# no neighbors → default weights=1
			return torch.ones(B, device=ref_pts_m.device), sel_tf_ind
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)
		if b_idx.numel() == 0:
			# no neighbors → zero weights
			return torch.ones(ref_pts_m.size(0), device=ref_pts_m.device), sel_tf_ind
		split_sizes = [ (b_idx==i).sum().item() for i in range(B) ]

		ref_f = torch.cat([ref_feats_m, ref_feats_f_sel], dim=-1)[r_idx]
		src_f = torch.cat([src_feats_m, src_feats_f_sel], dim=-1)[s_idx]

		outputs = [self.wnet(rf, sf) for rf, sf in zip(ref_f.split(split_sizes), src_f.split(split_sizes))]
		valid_ind, valid_outputs = zip(*[(i, out) for i, out in enumerate(outputs) if out is not None])
		valid_outputs = torch.cat(valid_outputs)
		w = torch.zeros(B, device=ref_pts_m.device)

		w[[valid_ind]] = self.proj(valid_outputs).flatten()

		# k = 256
		# m = torch.einsum("nd,nd->n", ref_feats_m[r_idx], src_feats_m[s_idx])
		# w_m = self.get_weight(m, split_sizes, k=k)

		# m = torch.einsum("nd,nd->n", ref_feats_f_sel[r_idx], src_feats_f_sel[s_idx])
		# w_f = self.get_weight(m, split_sizes, k=k)
		# w = (w_f + w_m) / 2  # average weights from feature and consistency
		# # w = w_f + w_m
		return w, sel_tf_ind

