import torch
import torch.nn as nn
from geotransformer.modules.ops.transformation import apply_transform


class FeatureConsistencyWeighting(nn.Module):
	def __init__(self, feat_dim=256, hidden_dim=32, radius=0.06):
		"""
		feat_dim: dimensionality of per‑point features
		hidden_dim: hidden size for MLPs
		radius: neighborhood radius for local edges
		"""
		super().__init__()
		self.radius = radius

		self.ffeat_mlp = nn.Sequential(
			nn.Linear(256, hidden_dim),
			nn.ReLU()
		)
		self.mfeat_mlp = nn.Sequential(
			nn.Linear(512, hidden_dim),
			nn.ReLU()
		)
		self.feat_m_mlp = nn.Sequential(
			nn.Linear(hidden_dim*2, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		self.feat_f_mlp = nn.Sequential(
			nn.Linear(hidden_dim*2, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)

		# MLP on edge features: [f_i, f_j, (x_i - x_j)] → embedding
		self.edge_mlp = nn.Sequential(
			nn.Linear(hidden_dim*2 + 1, hidden_dim),
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

	def forward(self, 
		ref_pts_m, ref_pts_f, 
		src_pts_m, src_pts_f, 
		ref_feats_m, ref_feats_f, 
		src_feats_m, src_feats_f,
		tf_est, tf_gt=None
	):
		sel_tf_ind = None
		# if ref_pts.shape[0] > self.max_points:
		# 	# downsample points and features
		# 	ref_pts, ref_feats = self.downsample_pcd_feat(ref_pts, ref_feats, num_points=self.max_points)
		# if src_pts.shape[0] > self.max_points:
		# 	src_pts, src_feats = self.downsample_pcd_feat(src_pts, src_feats, num_points=self.max_points)
		if self.training:
			# generate training transforms
			tf_est, sel_tf_ind = self.generate_training_tfs(src_pts_f, tf_est, tf_gt)

		B = tf_est.shape[0]
		# 1) transform source points
		src_pts_m_t = apply_transform(src_pts_m.unsqueeze(0).repeat(B,1,1), tf_est)  # user‑provided
		
		# 2) build radius graph (ref → src)
		dists = torch.cdist(ref_pts_m.unsqueeze(0).repeat(B,1,1), src_pts_m_t)  # [N_ref, N_src]
		mask = dists < self.radius
		if not mask.any():
			# no neighbors → default weights=1
			return torch.ones(B, device=d.device)
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)

		if b_idx.numel() == 0:
			# no neighbors → zero weights
			return torch.zeros(ref_pts_m.size(0), device=ref_pts_m.device)

		# 3) form edge inputs
		ref_mfeat = ref_feats_m[r_idx]                           # [E, F]
		src_mfeat = src_feats_m[s_idx]                           # [E, F]
		rel = ref_pts_m[r_idx] - src_pts_m_t[b_idx, s_idx]   # [E, 3]

		ref_mfc_knn_ind = torch.cdist(ref_pts_m, ref_pts_f).topk(3, dim=1, largest=False).indices
		src_mfc_knn_ind = torch.cdist(src_pts_m, src_pts_f).topk(3, dim=1, largest=False).indices
		ref_mfeat_knn_ffeat = ref_feats_f[ref_mfc_knn_ind[r_idx]]
		src_mfeat_knn_ffeat = src_feats_f[src_mfc_knn_ind[s_idx]]

		ref_ffeat_mean = ref_mfeat_knn_ffeat.mean(dim=1)
		src_ffeat_mean = src_mfeat_knn_ffeat.mean(dim=1)

		feat_m = torch.cat([self.mfeat_mlp(ref_mfeat), self.mfeat_mlp(src_mfeat)], dim=-1)
		feat_f = torch.cat([self.ffeat_mlp(ref_ffeat_mean), self.ffeat_mlp(src_ffeat_mean)], dim=-1)
		cp_dist = rel.square().sum(dim=-1).sqrt().unsqueeze(-1)  # [E, 1]

		edge_input = torch.cat([self.feat_m_mlp(feat_m), self.feat_f_mlp(feat_f), cp_dist], dim=-1)  # [E, 2F+3]

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

