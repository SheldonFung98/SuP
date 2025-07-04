# import torch
# import torch.nn as nn
# from geotransformer.modules.ops.transformation import apply_transform


# class FeatureConsistencyWeighting(nn.Module):
# 	def __init__(self, feat_dim=256, hidden_dim=32, radius=0.06):
# 		"""
# 		feat_dim: dimensionality of per‑point features
# 		hidden_dim: hidden size for MLPs
# 		radius: neighborhood radius for local edges
# 		"""
# 		super().__init__()
# 		self.radius = radius

# 		self.ffeat_mlp = nn.Sequential(
# 			nn.Linear(256, hidden_dim),
# 			nn.ReLU()
# 		)
# 		self.mfeat_mlp = nn.Sequential(
# 			nn.Linear(512, hidden_dim),
# 			nn.ReLU()
# 		)
# 		self.feat_m_mlp = nn.Sequential(
# 			nn.Linear(hidden_dim*2, hidden_dim),
# 			nn.ReLU(),
# 			nn.Linear(hidden_dim, hidden_dim),
# 			nn.ReLU(),
# 		)
# 		self.feat_f_mlp = nn.Sequential(
# 			nn.Linear(hidden_dim*2, hidden_dim),
# 			nn.ReLU(),
# 			nn.Linear(hidden_dim, hidden_dim),
# 			nn.ReLU(),
# 		)

# 		# MLP on edge features: [f_i, f_j, (x_i - x_j)] → embedding
# 		self.edge_mlp = nn.Sequential(
# 			nn.Linear(hidden_dim*2 + 1, hidden_dim),
# 			nn.ReLU(),
# 			nn.Linear(hidden_dim, hidden_dim),
# 			nn.ReLU()
# 		)
# 		# MLP on aggregated edge embeddings → scalar weight
# 		self.node_mlp = nn.Sequential(
# 			nn.Linear(hidden_dim, hidden_dim),
# 			nn.ReLU(),
# 			nn.Linear(hidden_dim, 1)
# 		)
# 		self.max_points = 2**13

# 	def compute_rmse(self, est_transform, src_points, gt_transform):
# 		points = src_points[None, ...].repeat(est_transform.shape[0], 1, 1)
# 		realignment_transform = torch.matmul(torch.inverse(gt_transform), est_transform)
# 		realigned_src_points_f = apply_transform(points, realignment_transform)
# 		rmse = torch.linalg.norm(realigned_src_points_f - points, dim=-1).mean(dim=-1)
# 		return rmse
		
# 	def generate_training_tfs(self, src_points, tf_est, tf_gt, acceptance_rmse=0.2):
# 		rmses = self.compute_rmse(tf_est, src_points, tf_gt)

# 		success = rmses < acceptance_rmse
# 		failed = ~success
# 		reasonable = failed & (rmses < 0.3)

# 		success_inds = success.nonzero().flatten()
# 		failed_inds = failed.nonzero().flatten()
# 		reasonable_inds = reasonable.nonzero().flatten()

# 		num = 4
# 		sel_ind = torch.cat([
# 			torch.randperm(success_inds.shape[0])[:num], 
# 			torch.randperm(failed_inds.shape[0])[:num],
# 			torch.randperm(reasonable_inds.shape[0])[:num]
# 		])

# 		return tf_est[sel_ind], sel_ind
	
# 	def downsample_pcd_feat(self, points, feats, num_points=1024):
# 		"""Downsample point cloud to a fixed number of points."""
# 		if points.shape[0] <= num_points:
# 			return points, feats
# 		# random sample indices
# 		indices = torch.randperm(points.shape[0])[:num_points]
# 		return points[indices], feats[indices]

# 	def forward(self, 
# 		ref_pts_m, ref_pts_f, 
# 		src_pts_m, src_pts_f, 
# 		ref_feats_m, ref_feats_f, 
# 		src_feats_m, src_feats_f,
# 		tf_est, tf_gt=None
# 	):
# 		sel_tf_ind = None
# 		# if ref_pts.shape[0] > self.max_points:
# 		# 	# downsample points and features
# 		# 	ref_pts, ref_feats = self.downsample_pcd_feat(ref_pts, ref_feats, num_points=self.max_points)
# 		# if src_pts.shape[0] > self.max_points:
# 		# 	src_pts, src_feats = self.downsample_pcd_feat(src_pts, src_feats, num_points=self.max_points)
# 		if self.training:
# 			# generate training transforms
# 			tf_est, sel_tf_ind = self.generate_training_tfs(src_pts_f, tf_est, tf_gt)

# 		B = tf_est.shape[0]
# 		# 1) transform source points
# 		src_pts_m_t = apply_transform(src_pts_m.unsqueeze(0).repeat(B,1,1), tf_est)  # user‑provided
		
# 		# 2) build radius graph (ref → src)
# 		dists = torch.cdist(ref_pts_m.unsqueeze(0).repeat(B,1,1), src_pts_m_t)  # [N_ref, N_src]
# 		mask = dists < self.radius
# 		if not mask.any():
# 			# no neighbors → default weights=1
# 			return torch.ones(B, device=d.device)
# 		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)

# 		if b_idx.numel() == 0:
# 			# no neighbors → zero weights
# 			return torch.zeros(ref_pts_m.size(0), device=ref_pts_m.device)

# 		# 3) form edge inputs
# 		ref_mfeat = ref_feats_m[r_idx]                       # [E, F]
# 		src_mfeat = src_feats_m[s_idx]                       # [E, F]
# 		rel = ref_pts_m[r_idx] - src_pts_m_t[b_idx, s_idx]   # [E, 3]

# 		ref_mfc_knn_ind = torch.cdist(ref_pts_m, ref_pts_f).topk(3, dim=1, largest=False).indices
# 		src_mfc_knn_ind = torch.cdist(src_pts_m, src_pts_f).topk(3, dim=1, largest=False).indices
# 		ref_mfeat_knn_ffeat = ref_feats_f[ref_mfc_knn_ind[r_idx]]
# 		src_mfeat_knn_ffeat = src_feats_f[src_mfc_knn_ind[s_idx]]

# 		ref_ffeat_mean = ref_mfeat_knn_ffeat.mean(dim=1)
# 		src_ffeat_mean = src_mfeat_knn_ffeat.mean(dim=1)

# 		feat_m = torch.cat([self.mfeat_mlp(ref_mfeat), self.mfeat_mlp(src_mfeat)], dim=-1)
# 		feat_f = torch.cat([self.ffeat_mlp(ref_ffeat_mean), self.ffeat_mlp(src_ffeat_mean)], dim=-1)
# 		cp_dist = rel.square().sum(dim=-1).sqrt().unsqueeze(-1)  # [E, 1]

# 		edge_input = torch.cat([self.feat_m_mlp(feat_m), self.feat_f_mlp(feat_f), cp_dist], dim=-1)  # [E, 2F+3]

# 		# 4) edge embedding
# 		e = self.edge_mlp(edge_input)  # [E, H]

# 		# 5) aggregate per ref‑point by max
# 		chunked = e.split([ (b_idx==i).sum().item() for i in range(B) ])
# 		max_feats = torch.stack([
# 			c.max(dim=0).values if c.numel()>0 else torch.zeros(e.size(1), device=e.device)
# 			for c in chunked
# 		])
# 		# 6) node weights
# 		w = self.node_mlp(max_feats).flatten() # [N]
# 		w = torch.sigmoid(w) 
# 		return w, sel_tf_ind


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################


import torch
import torch.nn as nn
from geotransformer.modules.ops.transformation import apply_transform


import torch
import torch.nn as nn
import torch.nn.functional as F


# VLAD operation
class NetVLAD(nn.Module):
	"""NetVLAD layer implementation"""

	def __init__(self, num_clusters=8, dim=32, alpha=1.0,
				 normalize_input=True):
		"""
		Args:
			num_clusters : int
				The number of clusters
			dim : int
				Dimension of descriptors
			alpha : float
				Parameter of initialization. Larger value is harder assignment.
			normalize_input : bool
				If true, descriptor-wise L2 normalization is applied to input.
		"""
		super().__init__()
		self.num_clusters = num_clusters
		self.dim = dim
		self.alpha = alpha
		self.normalize_input = normalize_input
		self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
		self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
		self._init_params()

	def _init_params(self):
		self.conv.weight = nn.Parameter(
			(2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
		)
		self.conv.bias = nn.Parameter(
			- self.alpha * self.centroids.norm(dim=1)
		)

	def forward(self, x):
		# x:n*f
		x = x.T[None,:,:,None]
		N, C = x.shape[:2]

		if self.normalize_input:
			x = F.normalize(x, p=2, dim=1)  # across descriptor dim

		# soft-assignment
		soft_assign = self.conv(x).view(N, self.num_clusters, -1)
		soft_assign = F.softmax(soft_assign, dim=1)

		x_flatten = x.view(N, C, -1)
		
		# calculate residuals to each clusters
		residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
			self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
		residual *= soft_assign.unsqueeze(2)
		vlad = residual.sum(dim=-1)

		vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
		vlad = vlad.view(x.size(0), -1)  # flatten
		# 1*of
		return vlad
	

class FeatureConsistencyWeighting(nn.Module):
	def __init__(self, feat_dim=256, hidden_dim=32, radius=0.1):
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
		ref_pts_c, ref_pts_m, ref_pts_f, 
		src_pts_c, src_pts_m, src_pts_f, 
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
		src_pts_c_t = apply_transform(src_pts_c.unsqueeze(0).repeat(B,1,1), tf_est)  # user‑provided
		
		# 2) build radius graph (ref → src)
		dists = torch.cdist(ref_pts_c.unsqueeze(0).repeat(B,1,1), src_pts_c_t)  # [N_ref, N_src]
		mask = dists < self.radius
		if not mask.any():
			# no neighbors → default weights=1
			return torch.ones(B, device=d.device)
		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)

		if b_idx.numel() == 0:
			# no neighbors → zero weights
			return torch.zeros(ref_pts_m.size(0), device=ref_pts_m.device)

		# 3) form edge inputs
		ref_mfeat = ref_feats_m[r_idx]                       # [E, F]
		src_mfeat = src_feats_m[s_idx]                       # [E, F]
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


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################

# Sinusoidal positional encoding for 1-d distance to 128-d embedding
class SinusoidalDistanceEmbedding(nn.Module):
	def __init__(self, emb_dim=128):
		super().__init__()
		self.emb_dim = emb_dim
		div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / emb_dim))
		self.register_buffer('div_term', div_term)

	def forward(self, x):
		# x: [N, 1] or [N]
		x = x.view(-1, 1).float()
		pe = torch.zeros(x.size(0), self.emb_dim, device=x.device)
		pe[:, 0::2] = torch.sin(x * self.div_term)
		pe[:, 1::2] = torch.cos(x * self.div_term)
		return pe

class WeightingNet(nn.Module):

	def __init__(self, num_clusters=8, dim=128, alpha=1.0,
				 normalize_input=True):
		"""
		Args:
			num_clusters : int
				The number of clusters
			dim : int
				Dimension of descriptors
			alpha : float
				Parameter of initialization. Larger value is harder assignment.
			normalize_input : bool
				If true, descriptor-wise L2 normalization is applied to input.
		"""
		super().__init__()
		self.num_clusters = num_clusters
		self.dim = dim
		self.alpha = alpha
		self.normalize_input = normalize_input
		self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
		self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

		self._init_params()

	def _init_params(self):
		self.conv.weight = nn.Parameter(
			(2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
		)
		self.conv.bias = nn.Parameter(
			- self.alpha * self.centroids.norm(dim=1)
		)

	def forward(self, x):

		# x:n*f
		x = x.T[None,:,:,None]
		N, C = x.shape[:2]

		if self.normalize_input:
			x = F.normalize(x, p=2, dim=1)  # across descriptor dim

		# soft-assignment
		soft_assign = self.conv(x).view(N, self.num_clusters, -1)
		soft_assign = F.softmax(soft_assign, dim=1)

		x_flatten = x.view(N, C, -1)
		
		# calculate residuals to each clusters
		residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
			self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
		residual *= soft_assign.unsqueeze(2)
		vlad = residual.sum(dim=-1)

		vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
		vlad = vlad.view(x.size(0), -1)  # flatten
		# 1*of
		return vlad


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
		self.mlp_coarse = nn.Sequential(
			nn.Linear(512, 64),
			nn.ReLU()
		)
		self.mlp_fine = nn.Sequential(
			nn.Linear(256, 64),
			nn.ReLU()
		)

		self.dist_emb = SinusoidalDistanceEmbedding(emb_dim=128)

		self.proj = nn.Sequential(
			nn.Linear(1024, 1),
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

	# def get_weight(self, m, batch_sizes, k=32):
	# 	topks = [k if i > k else i for i in batch_sizes]
	# 	x_paded, padding_mask, padding_lens = self.padding([i.topk( k=k, largest=False).values for k, i in zip(topks, m.split(batch_sizes))])
	# 	self.net(x_paded)
	# 	w = torch.stack([i.topk(k=k, largest=False).values.mean() for k, i in zip(topks, m.split(batch_sizes))])
	# 	w /= w.max()  # normalize weights
	# 	return w

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



		ref_fm = self.mlp_coarse(ref_feats_m)
		src_fm = self.mlp_coarse(src_feats_m)
		ref_ff = self.mlp_fine(ref_feats_f_sel)
		src_ff = self.mlp_fine(src_feats_f_sel)
		ref_f = torch.cat([ref_fm, ref_ff], dim=-1)[r_idx]
		src_f = torch.cat([src_fm, src_ff], dim=-1)[s_idx]
		d = self.dist_emb(dists[b_idx, r_idx, s_idx])

		feat_diff = ref_f - src_f + d
		o = torch.cat([self.wnet(x) for x in feat_diff.split(split_sizes)], dim=0)  # [N, 1]
		w = self.proj(o).flatten()

		# k = 256
		# m = torch.einsum("nd,nd->n", ref_feats_m[r_idx], src_feats_m[s_idx])
		# w_m = self.get_weight(m, split_sizes, k=k)

		# m = torch.einsum("nd,nd->n", ref_feats_f_sel[r_idx], src_feats_f_sel[s_idx])
		# w_f = self.get_weight(m, split_sizes, k=k)
		# # w = (w_f + w_m) / 2  # average weights from feature and consistency
		# w = w_f + w_m
		return w, sel_tf_ind

