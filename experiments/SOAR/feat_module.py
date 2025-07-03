import torch
import torch.nn as nn
from geotransformer.modules.ops.transformation import apply_transform

class MLPBlock(nn.Module):
	def __init__(self, dim, out_dim):
		super().__init__()
		self.block = nn.Sequential(
			nn.Linear(dim, out_dim),
			nn.LayerNorm(out_dim),
			nn.GELU(),
			nn.Linear(out_dim, out_dim),
			nn.LayerNorm(out_dim)
		)
	def forward(self, x):
		return self.block(x)

class DistanceEmbedding(nn.Module):
	def __init__(self, num_freqs=8, dim_f=256):
		super().__init__()
		self.freqs = nn.Parameter(torch.linspace(1.0, 10.0, num_freqs), requires_grad=False)
		self.proj = nn.Linear(num_freqs * 2, dim_f)

	def forward(self, d):  # d: (..., 1)
		# sinusoidal embedding
		x = d.unsqueeze(-1) * self.freqs.view(*([1] * (d.dim())) + [-1])
		emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
		return self.proj(emb)

class FeatureAlignmentConsistencyWeighting(nn.Module):
	def __init__(self, radius=0.06, dim_f=256, num_heads=4):
		super().__init__()
		self.radius = radius
		hidden_dim = dim_f // 2
		self.model_f = MLPBlock(dim_f, out_dim=hidden_dim)
		self.model_m = MLPBlock(hidden_dim, out_dim=hidden_dim)
		self.embed_d = DistanceEmbedding(num_freqs=16, dim_f=hidden_dim)
		self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
		self.pool = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU()
		)
		
		self.model_w = nn.Sequential(
			nn.Linear(hidden_dim, 64),
			nn.GELU(),
			nn.Linear(64, 1),
			nn.Sigmoid(),
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

		sel_ind = torch.cat([
			torch.randperm(success_inds.shape[0])[:8], 
			torch.randperm(failed_inds.shape[0])[:8],
			torch.randperm(reasonable_inds.shape[0])[:8]
		])

		return tf_est[sel_ind], sel_ind


	def forward(self, ref_points, src_points, ref_feats, src_feats, tf_est, tf_gt=None):
		if self.training:
			# generate training transforms
			tf_est, sel_tf_ind = self.generate_training_tfs(src_points, tf_est, tf_gt)
		B = tf_est.shape[0]
		# compute transformed source and pairwise distance
		src_trans = apply_transform(src_points.unsqueeze(0).expand(B, -1, -1), tf_est)
		d = torch.cdist(ref_points.unsqueeze(0).expand(B, -1, -1), src_trans)

		# mask neighbors
		mask = d < self.radius
		if not mask.any():
			return torch.ones(B, device=d.device)

		# gather indices
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)
		# gather and embed
		rf = self.model_f(ref_feats.unsqueeze(0).expand(B, -1, -1)[b_idx, r_idx])
		sf = self.model_f(src_feats.unsqueeze(0).expand(B, -1, -1)[b_idx, s_idx])
		d_emb = self.embed_d(d[b_idx, r_idx, s_idx])

		# combine features
		diff = rf - sf
		feats = self.model_m(diff) + d_emb

		# reshape for attention: (N, F) -> (1, N, F) query, key, value
		feats_t = feats.unsqueeze(1)
		attn_out, _ = self.attn(feats_t, feats_t, feats_t)
		attn_out = attn_out.squeeze(1)

		# aggregate per batch via segment soft-attention
		weights = []
		for i in range(B):
			seg = attn_out[b_idx == i]
			if seg.numel() == 0:
				# no neighbors
				weights.append(torch.zeros(self.model_w[0].in_features, device=d.device))
			else:
				# attention pooling
				alpha = torch.softmax(seg.mean(dim=-1, keepdim=True), dim=0)
				pooled = (alpha * seg).sum(dim=0)
				weights.append(pooled)
		stacked = torch.stack(weights, dim=0)

		# predict final weight
		w = self.model_w(self.pool(stacked)).view(-1)
		return w, sel_tf_ind if self.training else None


class FeatureConsistencyWeighting(nn.Module):
	def __init__(self, feat_dim=256, hidden_dim=64, radius=0.06):
		"""
		feat_dim: dimensionality of per‑point features
		hidden_dim: hidden size for MLPs
		radius: neighborhood radius for local edges
		"""
		super().__init__()
		self.radius = radius
		# MLP on edge features: [f_i, f_j, (x_i - x_j)] → embedding
		self.edge_mlp = nn.Sequential(
			nn.Linear(feat_dim*2 + 3, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)
		# MLP on aggregated edge embeddings → scalar weight
		self.node_mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.max_points = 2**13

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

		num = 4
		sel_ind = torch.cat([
			torch.randperm(success_inds.shape[0])[:num], 
			torch.randperm(failed_inds.shape[0])[:num],
			torch.randperm(reasonable_inds.shape[0])[:num]
		])

		return tf_est[sel_ind], sel_ind
	
	def downsample_pcd_feat(self, points, feats, num_points=1024):
		"""Downsample point cloud to a fixed number of points."""
		if points.shape[0] <= num_points:
			return points, feats
		# random sample indices
		indices = torch.randperm(points.shape[0])[:num_points]
		return points[indices], feats[indices]

	def forward(self, ref_pts, src_pts, ref_feats, src_feats, tf_est, tf_gt=None):
		sel_tf_ind = None
		if ref_pts.shape[0] > self.max_points:
			# downsample points and features
			ref_pts, ref_feats = self.downsample_pcd_feat(ref_pts, ref_feats, num_points=self.max_points)
		if src_pts.shape[0] > self.max_points:
			src_pts, src_feats = self.downsample_pcd_feat(src_pts, src_feats, num_points=self.max_points)
		if self.training:
			# generate training transforms
			tf_est, sel_tf_ind = self.generate_training_tfs(src_pts, tf_est, tf_gt)

		B = tf_est.shape[0]
		# 1) transform source points
		src_pts_t = apply_transform(src_pts.unsqueeze(0).repeat(B,1,1), tf_est)  # user‑provided
		
		# 2) build radius graph (ref → src)
		dists = torch.cdist(ref_pts.unsqueeze(0).repeat(B,1,1), src_pts_t)  # [N_ref, N_src]
		mask = dists < self.radius
		if not mask.any():
			# no neighbors → default weights=1
			return torch.ones(B, device=d.device)
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)

		if b_idx.numel() == 0:
			# no neighbors → zero weights
			return torch.zeros(ref_pts.size(0), device=ref_pts.device)

		# 3) form edge inputs
		f_i = ref_feats[r_idx]                           # [E, F]
		f_j = src_feats[s_idx]                           # [E, F]
		rel = ref_pts[r_idx] - src_pts_t[b_idx, s_idx]   # [E, 3]
		edge_input = torch.cat([f_i, f_j, rel], dim=-1)  # [E, 2F+3]

		# 4) edge embedding
		e = self.edge_mlp(edge_input)  # [E, H]

		# 5) aggregate per ref‑point by max
		chunked = e.split([ (b_idx==i).sum().item() for i in range(B) ])
		max_feats = torch.stack([
			c.max(dim=0).values if c.numel()>0 else torch.zeros(e.size(1), device=e.device)
			for c in chunked
		])
		# 6) node weights
		w = self.node_mlp(max_feats).flatten() # [N]
		w = torch.sigmoid(w) 
		return w, sel_tf_ind










class FeatureAlignmentConsistencyWeighting(nn.Module):
	def __init__(self, radius=0.06, dim_f=256, num_heads=4):
		super().__init__()
		self.radius = radius
		hidden_dim = dim_f // 2
		self.model_f = MLPBlock(dim_f, out_dim=hidden_dim)
		self.model_m = MLPBlock(hidden_dim, out_dim=hidden_dim)
		self.embed_d = DistanceEmbedding(num_freqs=16, dim_f=hidden_dim)
		self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
		self.pool = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU()
		)
		
		self.model_w = nn.Sequential(
			nn.Linear(hidden_dim, 64),
			nn.GELU(),
			nn.Linear(64, 1),
			nn.Sigmoid(),
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

		sel_ind = torch.cat([
			torch.randperm(success_inds.shape[0])[:8], 
			torch.randperm(failed_inds.shape[0])[:8],
			torch.randperm(reasonable_inds.shape[0])[:8]
		])

		return tf_est[sel_ind], sel_ind


	def forward(self, ref_points, src_points, ref_feats, src_feats, tf_est, tf_gt=None):
		if self.training:
			# generate training transforms
			tf_est, sel_tf_ind = self.generate_training_tfs(src_points, tf_est, tf_gt)
		B = tf_est.shape[0]
		# compute transformed source and pairwise distance
		src_trans = apply_transform(src_points.unsqueeze(0).expand(B, -1, -1), tf_est)
		d = torch.cdist(ref_points.unsqueeze(0).expand(B, -1, -1), src_trans)

		# mask neighbors
		mask = d < self.radius
		if not mask.any():
			return torch.ones(B, device=d.device)

		# gather indices
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)
		# gather and embed
		rf = self.model_f(ref_feats.unsqueeze(0).expand(B, -1, -1)[b_idx, r_idx])
		sf = self.model_f(src_feats.unsqueeze(0).expand(B, -1, -1)[b_idx, s_idx])
		d_emb = self.embed_d(d[b_idx, r_idx, s_idx])

		# combine features
		diff = rf - sf
		feats = self.model_m(diff) + d_emb

		# reshape for attention: (N, F) -> (1, N, F) query, key, value
		feats_t = feats.unsqueeze(1)
		attn_out, _ = self.attn(feats_t, feats_t, feats_t)
		attn_out = attn_out.squeeze(1)

		# aggregate per batch via segment soft-attention
		weights = []
		for i in range(B):
			seg = attn_out[b_idx == i]
			if seg.numel() == 0:
				# no neighbors
				weights.append(torch.zeros(self.model_w[0].in_features, device=d.device))
			else:
				# attention pooling
				alpha = torch.softmax(seg.mean(dim=-1, keepdim=True), dim=0)
				pooled = (alpha * seg).sum(dim=0)
				weights.append(pooled)
		stacked = torch.stack(weights, dim=0)

		# predict final weight
		w = self.model_w(self.pool(stacked)).view(-1)
		return w, sel_tf_ind if self.training else None


class FeatureConsistencyWeighting(nn.Module):
	def __init__(self, feat_dim=256, hidden_dim=64, radius=0.06):
		"""
		feat_dim: dimensionality of per‑point features
		hidden_dim: hidden size for MLPs
		radius: neighborhood radius for local edges
		"""
		super().__init__()
		self.radius = radius
		# MLP on edge features: [f_i, f_j, (x_i - x_j)] → embedding
		self.edge_mlp = nn.Sequential(
			nn.Linear(feat_dim*2 + 3, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU()
		)
		# MLP on aggregated edge embeddings → scalar weight
		self.node_mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)
		self.max_points = 2**13

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

		num = 4
		sel_ind = torch.cat([
			torch.randperm(success_inds.shape[0])[:num], 
			torch.randperm(failed_inds.shape[0])[:num],
			torch.randperm(reasonable_inds.shape[0])[:num]
		])

		return tf_est[sel_ind], sel_ind
	
	def downsample_pcd_feat(self, points, feats, num_points=1024):
		"""Downsample point cloud to a fixed number of points."""
		if points.shape[0] <= num_points:
			return points, feats
		# random sample indices
		indices = torch.randperm(points.shape[0])[:num_points]
		return points[indices], feats[indices]

	def forward(self, ref_pts, src_pts, ref_feats, src_feats, tf_est, tf_gt=None):
		sel_tf_ind = None
		if ref_pts.shape[0] > self.max_points:
			# downsample points and features
			ref_pts, ref_feats = self.downsample_pcd_feat(ref_pts, ref_feats, num_points=self.max_points)
		if src_pts.shape[0] > self.max_points:
			src_pts, src_feats = self.downsample_pcd_feat(src_pts, src_feats, num_points=self.max_points)
		if self.training:
			# generate training transforms
			tf_est, sel_tf_ind = self.generate_training_tfs(src_pts, tf_est, tf_gt)

		B = tf_est.shape[0]
		# 1) transform source points
		src_pts_t = apply_transform(src_pts.unsqueeze(0).repeat(B,1,1), tf_est)  # user‑provided
		
		# 2) build radius graph (ref → src)
		dists = torch.cdist(ref_pts.unsqueeze(0).repeat(B,1,1), src_pts_t)  # [N_ref, N_src]
		mask = dists < self.radius
		if not mask.any():
			# no neighbors → default weights=1
			return torch.ones(B, device=d.device)
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)

		if b_idx.numel() == 0:
			# no neighbors → zero weights
			return torch.zeros(ref_pts.size(0), device=ref_pts.device)
 
		# 3) form edge inputs
		f_i = ref_feats[r_idx]                           # [E, F]
		f_j = src_feats[s_idx]                           # [E, F]

		rel = ref_pts[r_idx] - src_pts_t[b_idx, s_idx]   # [E, 3]
		
		edge_input = torch.cat([f_i, f_j, rel], dim=-1)  # [E, 2F+3]

		# 4) edge embedding
		e = self.edge_mlp(edge_input)  # [E, H]

		# 5) aggregate per ref‑point by max
		chunked = e.split([ (b_idx==i).sum().item() for i in range(B) ])
		max_feats = torch.stack([
			c.max(dim=0).values if c.numel()>0 else torch.zeros(e.size(1), device=e.device)
			for c in chunked
		])
		# 6) node weights
		w = self.node_mlp(max_feats).flatten() # [N]
		w = torch.sigmoid(w) 
		return w, sel_tf_ind







# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint

# class MLPBlock(nn.Module):
#     def __init__(self, dim, hidden_dim=None, norm=True):
#         super().__init__()
#         hidden_dim = hidden_dim or dim
#         layers = [nn.Linear(dim, hidden_dim)]
#         if norm:
#             layers.append(nn.LayerNorm(hidden_dim))
#         layers.append(nn.GELU())
#         layers.append(nn.Linear(hidden_dim, dim))
#         if norm:
#             layers.append(nn.LayerNorm(dim))
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         # apply checkpoint to reduce memory
#         return x + checkpoint.checkpoint(self.block, x)

# class DistanceEmbedding(nn.Module):
#     def __init__(self, num_freqs=8, dim_f=256):
#         super().__init__()
#         self.freqs = nn.Parameter(torch.linspace(1.0, 10.0, num_freqs), requires_grad=False)
#         self.proj = nn.Linear(num_freqs * 2, dim_f)

#     def forward(self, d):  # d: (K,1)
#         x = d * self.freqs.view(1, -1)      # (K, num_freqs)
#         emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
#         return self.proj(emb)

# class FeatureAlignmentConsistencyWeighting(nn.Module):
#     def __init__(self, radius, dim_f=256, max_neighbors=512):
#         super().__init__()
#         self.radius = radius
#         self.max_neighbors = max_neighbors
#         self.model_f = nn.Sequential(MLPBlock(dim_f), MLPBlock(dim_f))
#         self.model_m = MLPBlock(dim_f)
#         self.embed_d = DistanceEmbedding(num_freqs=16, dim_f=dim_f)
#         # remove global attention to save memory
#         self.pool = nn.Sequential(nn.Linear(dim_f, dim_f), nn.GELU())
#         self.model_w = nn.Sequential(
#             nn.Linear(dim_f, dim_f),
#             nn.GELU(),
#             nn.Linear(dim_f, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, ref_points, src_points, ref_feats, src_feats, tf_est):
#         B = tf_est.shape[0]
#         device = ref_points.device
#         # transform src and compute distances
#         src_trans = apply_transform(src_points.unsqueeze(0).expand(B, -1, -1), tf_est)
#         d = torch.cdist(ref_points.unsqueeze(0).expand(B, -1, -1), src_trans)

#         weights = []
#         # process per-transform to limit memory
#         for i in range(B):
#             # find neighbors for this transform
#             mask = d[i] < self.radius
#             idx = mask.nonzero(as_tuple=False)
#             K = idx.size(0)
#             if K == 0:
#                 weights.append(torch.tensor(1.0, device=device))
#                 continue
#             # sample top-k or random if too many
#             if K > self.max_neighbors:
#                 perm = torch.randperm(K, device=device)[: self.max_neighbors]
#                 idx = idx[perm]
#                 K = self.max_neighbors

#             r_idx = idx[:,0]
#             s_idx = idx[:,1]
#             rf = self.model_f(ref_feats[r_idx])
#             sf = self.model_f(src_feats[s_idx])
#             d_emb = self.embed_d(d[i, r_idx, s_idx].unsqueeze(-1))

#             diff = rf - sf
#             feats = self.model_m(diff) + d_emb
#             # simple pooling
#             pooled = feats.mean(dim=0)
#             # predict weight
#             w_i = self.model_w(self.pool(pooled.unsqueeze(0))).view(-1)
#             weights.append(w_i)

#         return torch.stack(weights, dim=0)
