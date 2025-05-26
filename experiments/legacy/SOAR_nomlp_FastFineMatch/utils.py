import numpy as np
import torch

from geotransformer.modules.ops import (
	grid_subsample, radius_search, grid_subsample_dps, 
	point_to_node_partition, index_select
)
from torch_cluster import fps, knn
from geotransformer.modules.ops import pairwise_distance
import torch.nn.functional as F
from pointscope import PointScopeClient as PSC
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.geotransformer import SuperPointMatching
from geotransformer.modules.registration import weighted_procrustes

def point2anchor_partition(points, anchor_num=5, anchor_max_points=None):
	fps_indices = fps(points, ratio=anchor_num / points.shape[0], random_start=False)
	anchors = points[fps_indices]
	point_to_node, node_masks, node_knn_indices, node_knn_masks = point_to_node_partition(
		points, anchors, anchor_max_points
	)
	return point_to_node, node_masks, node_knn_indices, node_knn_masks

	padded_points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
	blocks = index_select(padded_points, node_knn_indices, dim=0)
	blocks = [blocks[i][node_knn_masks[i]] for i in range(blocks.shape[0])]
	return blocks
	if False:
		from pointscope import PointScopeClient as PSC
		psc = PSC().vedo(subplot=2)
		psc.add_pcd(points).draw_at(1)
		for i in blocks:
			psc.add_pcd(i)
		psc.show()


def pcd_division_knn_radius(points, centroids=None, fragment_num=5):
	if centroids is None:
		centroids = fps(points, ratio=fragment_num / points.shape[0])
	anchors = points[centroids]
	neighbor_indices = radius_search(
		anchors,
		points, 
		torch.tensor([anchors.shape[0]]),
		torch.tensor([points.shape[0]]), 
		3.5, 
		180
	)
	return None, None, neighbor_indices, neighbor_indices!=points.shape[0]
	padded_points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
	blocks = index_select(padded_points, neighbor_indices, dim=0)
	return blocks

	if False:
		from pointscope import PointScopeClient as PSC
		psc = PSC().vedo(subplot=2)
		psc.add_pcd(points).draw_at(1)
		for i in blocks:
			psc.add_pcd(i)
		psc.show()

	if False:
		from pointscope import PointScopeClient as PSC
		psc = PSC().vedo(subplot=2)
		psc.add_pcd(points).draw_at(1)
		for i in blocks:
			psc.add_pcd(i)
		psc.show()


def padding(tensor_list):
	"""
	args:
		tensor_list: list[torch.Tensor]
	return: 
		padded_tensor: torch.Tensor
		tensor_mask: torch.Tensor
	"""
	max_len = max([tensor.shape[0] for tensor in tensor_list])
	padded_tensor = torch.zeros((len(tensor_list), max_len, tensor_list[0].shape[1]), device=tensor_list[0].device)
	tensor_mask = torch.zeros((len(tensor_list), max_len), device=tensor_list[0].device)
	for i, tensor in enumerate(tensor_list):
		padded_tensor[i, :tensor.shape[0], :] = tensor
		tensor_mask[i, :tensor.shape[0]] = 1
	return padded_tensor, tensor_mask

def unpad(tensor, tensor_mask):
	"""
	args:
		tensor: torch.Tensor
		tensor_mask: torch.Tensor
	return: 
		unpadded_tensor: list[torch.Tensor]
	"""
	unpadded_tensor = []
	for i in range(tensor.shape[0]):
		unpadded_tensor.append(tensor[i, :int(tensor_mask[i].sum()), :])
	return unpadded_tensor


def weighted_farthest_point_sample(xyz, npoint, weights):
	"""
	Input:
		xyz: pointcloud data, [B, N, 3]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [B, npoint]
	"""
	device = xyz.device
	B, N, C = xyz.shape
	centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e10
	# farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
	farthest = weights.max(dim=1).indices
	batch_indices = torch.arange(B, dtype=torch.long).to(device)
	weights = weights - weights.min(dim=1, keepdim=True).values
	weights = weights / weights.max(dim=1, keepdim=True).values
	for i in range(npoint):
		centroids[:, i] = farthest
		centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
		dist = torch.sum((xyz - centroid) ** 2, -1).sqrt() * weights
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, -1)[1]
	return centroids


def pcd_overlap_est(ref_feats, src_feats, topk=5, mutual_topk=True):

	matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
	ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
	src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
	matching_scores = ref_matching_scores * src_matching_scores

	ref_length, src_length = matching_scores.shape
	# correspondences from reference side
	ref_topk_scores, ref_topk_indices = matching_scores.topk(k=topk, dim=1)
	ref_indices = torch.arange(ref_length).to(matching_scores.device).view(ref_length, 1).expand(-1, topk)  # (B, N, K)
	ref_score_mat = torch.zeros_like(matching_scores)
	ref_score_mat[ref_indices, ref_topk_indices] = ref_topk_scores

	# correspondences from source side
	src_topk_scores, src_topk_indices = matching_scores.topk(k=topk, dim=0)
	src_indices = torch.arange(src_length).to(matching_scores.device).view(1, src_length).expand(topk, -1)  # (B, N, K)
	src_score_mat = torch.zeros_like(matching_scores)
	src_score_mat[src_topk_indices, src_indices] = src_topk_scores
	# merge results from two sides
	if mutual_topk:
		overlap_est = ref_score_mat * src_score_mat
	else:
		overlap_est = torch.logical_or(ref_score_mat, src_score_mat)

	overlap_est = overlap_est - overlap_est.min()
	ref_overlap_est, src_overlap_est = overlap_est.sum(dim=1), overlap_est.sum(dim=0)
	ref_overlap_est = ref_overlap_est / ref_overlap_est.max()
	src_overlap_est = src_overlap_est / src_overlap_est.max()

	return ref_overlap_est, src_overlap_est

def hierarchical_weight_merging(points_c, weight_c, points_m, weight_m, points_f, weight_f):

	points_m2f_ind = torch.cdist(points_m, points_f).topk(k=3, dim=1, largest=False).indices
	weights_m2f = weight_f[points_m2f_ind].mean(dim=1)

	points_c2m_ind = torch.cdist(points_c, points_m).topk(k=5, dim=1, largest=False).indices
	weight_mf = weight_m * weights_m2f
	weights_c2m = weight_mf[points_c2m_ind].mean(dim=1)

	weight = weight_c * weights_c2m

	return weight


def hierarchical_overlap_est(
	ref_points_c, ref_feats_c, src_points_c, src_feats_c,
	ref_points_m, ref_feats_m, src_points_m, src_feats_m,
	ref_points_f, ref_feats_f, src_points_f, src_feats_f):
	ref_c_overlap_est, src_c_overlap_est = pcd_overlap_est(
		F.normalize(ref_feats_c, p=2, dim=-1),
		F.normalize(src_feats_c, p=2, dim=-1)
	)
	ref_m_overlap_est, src_m_overlap_est = pcd_overlap_est(
		F.normalize(ref_feats_m, p=2, dim=-1),
		F.normalize(src_feats_m, p=2, dim=-1)
	)
	ref_f_overlap_est, src_f_overlap_est = pcd_overlap_est(
		F.normalize(ref_feats_f, p=2, dim=-1),
		F.normalize(src_feats_f, p=2, dim=-1)
	)
	ref_hc_overlap_est = hierarchical_weight_merging(
		ref_points_c, ref_c_overlap_est,
		ref_points_m, ref_m_overlap_est,
		ref_points_f, ref_f_overlap_est,
	)
	src_hc_overlap_est = hierarchical_weight_merging(
		src_points_c, src_c_overlap_est,
		src_points_m, src_m_overlap_est,
		src_points_f, src_f_overlap_est,
	)
	
	return ref_hc_overlap_est, src_hc_overlap_est


def compute_feature_base_consistency(
	ref_points, src_points, ref_feats, src_feats, tf_est, radius=0.1, focal=6, top_ratio=0.8, feat_dist_thr=1.6):
	# r2s_dist, r2s_ind = torch.cdist(ref_points, apply_transform(src_points, tf_est)).topk(k=k, dim=1, largest=False)
	r2s_dist = torch.cdist(ref_points, apply_transform(src_points, tf_est))
	neighbour_inds = (r2s_dist < radius).nonzero()


	# if neighbour_inds.shape[0] == 0:
	#     return None
	feat_dist = (ref_feats[neighbour_inds[:, 0]] - src_feats[neighbour_inds[:, 1]]).square().sum(dim=1)
	# feat_dist = feat_dist[feat_dist > feat_dist_thr]
	# if feat_dist.shape[0] == 0:
	#     return None
	# topk = int(feat_dist.shape[0] * top_ratio)
	# if topk >= 1:
	#     feat_dist = feat_dist.topk(k=topk, dim=0, largest=True).values
	# topk = 3
	# feat_sim = 1 - (torch.exp((0.0001*(feat_dist**focal)))-1).topk(k=topk, dim=0, largest=True).values.sum()
	# return feat_sim

	# matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
	# ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
	# src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
	# matching_scores = ref_matching_scores * src_matching_scores
	# feat_sim = matching_scores[r2s_dist < radius].sum()
	# topk = 3
	# feat_sim = matching_scores[r2s_dist < radius].topk(k=topk, largest=False).values.sum()
	# m = matching_scores[r2s_dist < radius]
	m = torch.exp(-feat_dist)
	min_llh = torch.exp(-0.05*(((m.topk(3,largest=False).values-m.mean())**2)/(2*m.std()**2)))
	min_llh *= 1/torch.sqrt(2*3.14*m.std()**2)
	return min_llh.mean()

	# return feat_sim.clamp(min=0, max=1)
	PSC().vedo()\
		.add_pcd(ref_points)\
		.add_pcd(apply_transform(src_points, tf_est))\
		.add_lines(ref_points[neighbour_inds[:,0]], apply_transform(src_points, tf_est)[neighbour_inds[:,1]]).show()

	feat_sim_color = feat_sim[:, None].repeat(1, 3)
	feat_sim_color[:, 1:] = 0
	PSC().vedo()\
		.add_pcd(ref_points)\
		.add_pcd(apply_transform(src_points, tf_est))\
		.add_lines(ref_points[neighbour_inds[:,0]], apply_transform(src_points, tf_est)[neighbour_inds[:,1]], colors=feat_sim_color).show()
	

def compute_feature_base_consistency(
	ref_points, src_points, 
	ref_feats, src_feats, 
	tf_est, radius=0.1, alpha=0.05, top_k=3):
	r2s_dist = torch.cdist(ref_points, apply_transform(src_points, tf_est))
	neighbour_inds = (r2s_dist < radius).nonzero()
	feat_dist = (ref_feats[neighbour_inds[:, 0]] - src_feats[neighbour_inds[:, 1]]).square().sum(dim=1)

	if feat_dist.shape[0] < top_k:
		return 0

	m = torch.exp(-feat_dist)
	min_llh = torch.exp(-alpha*(((m.topk(top_k,largest=False).values-m.mean())**2)/(2*m.std()**2)))
	min_llh *= 1/torch.sqrt(2*3.14*m.std()**2)
	return min_llh.mean()

	PSC().vedo()\
		.add_pcd(ref_points)\
		.add_pcd(apply_transform(src_points, tf_est))\
		.add_lines(ref_points[neighbour_inds[:,0]], apply_transform(src_points, tf_est)[neighbour_inds[:,1]]).show()

	feat_sim_color = feat_sim[:, None].repeat(1, 3)
	feat_sim_color[:, 1:] = 0
	PSC().vedo()\
		.add_pcd(ref_points)\
		.add_pcd(apply_transform(src_points, tf_est))\
		.add_lines(ref_points[neighbour_inds[:,0]], apply_transform(src_points, tf_est)[neighbour_inds[:,1]], colors=feat_sim_color).show()
	
def compute_feature_base_consistency_(
	ref_points, src_points, 
	ref_feats, src_feats, 
	tf_est, radius=0.1, alpha=0.05, top_k=3):
	r2s_dist = torch.cdist(ref_points, apply_transform(src_points, tf_est))
	neighbour_inds = (r2s_dist < radius).nonzero()
	if neighbour_inds.shape[0] == 0:
		return 0
	m = torch.einsum("nd,nd->n", ref_feats[neighbour_inds[:, 0]], src_feats[neighbour_inds[:, 1]])
	# min_llh = torch.exp(-alpha*(((m.topk(top_k,largest=False).values-m.mean())**2)/(2*m.std()**2)))
	# min_llh *= 1/torch.sqrt(2*3.14*m.std()**2)
	# return min_llh.mean()
	return m.mean()

	PSC().vedo()\
		.add_pcd(torch.cat([ref_points, apply_transform(src_points, tf_est)]))\
		.add_feat(torch.cat([ref_feats, src_feats]))\
		.add_lines(ref_points[neighbour_inds[:,0]], apply_transform(src_points, tf_est)[neighbour_inds[:,1]]).show()
	c = torch.zeros(neighbour_inds.shape[0], 3)
	c[:, 0] = (m/m.max()).clamp(min=0, max=1)
	PSC().vedo()\
		.add_pcd(torch.cat([ref_points, apply_transform(src_points, tf_est)]))\
		.add_feat(torch.cat([ref_feats, src_feats]))\
		.add_lines(ref_points[neighbour_inds[:,0]], apply_transform(src_points, tf_est)[neighbour_inds[:,1]], colors=c).show()


def blockwise_overlap_est(ref_feats_b, src_feats_b, topk=3, mutual_topk=True):
	# Block-wise Overlap Estimation
	# matching_scores = torch.exp(-pairwise_distance(ref_feats_b, src_feats_b, normalized=True))
	# ref_matching_scores = matching_scores / matching_scores.sum(dim=-1, keepdim=True)
	# src_matching_scores = matching_scores / matching_scores.sum(dim=-2, keepdim=True)
	# matching_scores = ref_matching_scores * src_matching_scores
	# block_overlap_est = matching_scores.topk(k=topk, largest=True, dim=2).values.mean(dim=-1).mean(dim=-1)
	# if block_overlap_est.max() != block_overlap_est.min():
	#     block_overlap_est_norm = block_overlap_est - block_overlap_est.min()
	#     block_overlap_est_norm /= block_overlap_est_norm.max()
	#     block_overlap_est = block_overlap_est_norm
	matching_scores = torch.exp(-pairwise_distance(ref_feats_b, src_feats_b, normalized=True))
	ref_matching_scores = matching_scores / matching_scores.sum(dim=-1, keepdim=True)
	src_matching_scores = matching_scores / matching_scores.sum(dim=-2, keepdim=True)
	matching_scores = ref_matching_scores * src_matching_scores
	b_size, ref_length, src_length = matching_scores.shape
	batch_indices = torch.arange(b_size).to(matching_scores.device)
	# correspondences from reference side
	ref_topk_scores, ref_topk_indices = matching_scores.topk(k=topk, dim=-1)
	ref_batch_indices = batch_indices.view(b_size, 1, 1).expand(-1, ref_length, topk)  # (B, N, K)
	ref_indices = torch.arange(ref_length).to(matching_scores.device).view(1, ref_length, 1).expand(1, -1, topk)  # (B, N, K)
	ref_score_mat = torch.zeros_like(matching_scores)
	ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
	# correspondences from source side
	src_topk_scores, src_topk_indices = matching_scores.topk(k=topk, dim=-2)
	src_batch_indices = batch_indices.view(b_size, 1, 1).expand(-1, topk, src_length)  # (B, K, N)
	src_indices = torch.arange(src_length).to(matching_scores.device).view(1, src_length).expand(1, topk, -1)  # (B, N, K)
	src_score_mat = torch.zeros_like(matching_scores)
	src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
	# merge results from two sides
	if mutual_topk:
		overlap_est = ref_score_mat * src_score_mat
	else:
		overlap_est = torch.logical_or(ref_score_mat, src_score_mat)
	
	overlap_est = overlap_est.sum(dim=2).sum(dim=1)
	if overlap_est.max() != overlap_est.min():
		overlap_est_norm = overlap_est - overlap_est.min()
		overlap_est_norm /= overlap_est_norm.max()
	elif overlap_est.max() > 0:
		overlap_est_norm = overlap_est / overlap_est.max()
	else:
		overlap_est_norm = overlap_est
	return overlap_est_norm



class PartialRegistrationGenerator(torch.nn.Module):
	def __init__(self, pair_num=24, anchor_num=8, num_correspondences=256):
		super().__init__()
		self.anchor_num = anchor_num
		self.num_correspondences = num_correspondences
		self.pair_num = pair_num

	def blockwise_overlap_est(self, ref_feats_b, src_feats_b, topk=3, mutual_topk=False):
		# Block-wise Overlap Estimation
		r_len, ref_sizes, _ = ref_feats_b.shape
		s_len, src_sizes, _ = src_feats_b.shape
		ref_feats_b = ref_feats_b.view(r_len * ref_sizes, -1)
		src_feats_b = src_feats_b.view(s_len * src_sizes, -1)
		matching_scores = torch.exp(-pairwise_distance(ref_feats_b, src_feats_b, normalized=True))
		# matching_scores = matching_scores.view(r_len*s_len, ref_sizes, src_sizes)
		ref_matching_scores = matching_scores / matching_scores.sum(dim=-1, keepdim=True)
		src_matching_scores = matching_scores / matching_scores.sum(dim=-2, keepdim=True)
		matching_scores = ref_matching_scores * src_matching_scores
		matching_scores = matching_scores.view(r_len*s_len, ref_sizes, src_sizes)
		block_overlap_est = matching_scores.topk(k=topk, largest=True).values.mean(dim=1).mean(dim=1).reshape(r_len, s_len)
		return block_overlap_est

	def sample(self, points, anchor_num=8, k=180):
		if k > points.shape[0]:
			k = points.shape[0]
		centroids = fps(points, ratio=anchor_num / points.shape[0])[:anchor_num]
		anchors = points[centroids]
		knn_inds = torch.cdist(anchors, points).topk(k=k, dim=1, largest=False).indices
		return points[knn_inds], knn_inds
	
	def matching(self, ref_feats, src_feats, dual_normalization=True, num_correspondences=256):
		bsize, src_size, _ = src_feats.shape
		# select top-k proposals
		matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
		if dual_normalization:
			ref_matching_scores = matching_scores / matching_scores.sum(dim=2, keepdim=True)
			src_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
			matching_scores = ref_matching_scores * src_matching_scores
		num_correspondences = min(num_correspondences, matching_scores[0].numel())
		corr_scores, corr_indices = matching_scores.view(bsize, -1).topk(k=num_correspondences, largest=True, dim=1)
		ref_corr_indices = corr_indices // src_size
		src_corr_indices = corr_indices % src_size
		return ref_corr_indices, src_corr_indices, corr_scores
	
	def select(self, x, inds):
		inds = inds.unsqueeze(-1).expand(-1, -1, x.shape[-1])
		return torch.gather(x, 1, inds)
	
	def corrs_weights(self, ref_corrs, src_corrs, corr_scores, transform_ests):
		corr_scores /= corr_scores.max(dim=1, keepdim=True).values
		corr_deviation = corr_scores * (ref_corrs - apply_transform(src_corrs, transform_ests)).square().sum(dim=-1)
		weights = torch.exp(-corr_deviation.topk(k=3, dim=1).values.mean(dim=1))
		return weights

	def forward(self, ref_points, src_points, ref_feats, src_feats, T=None):
		ref_points_sp, ref_sp_ind = self.sample(ref_points, self.anchor_num)
		ref_feats_sp = ref_feats[ref_sp_ind]
		src_points_sp, src_sp_ind = self.sample(src_points, self.anchor_num)
		src_feats_sp = src_feats[src_sp_ind]
		
		block_overlap_prior_est = self.blockwise_overlap_est(ref_feats_sp, src_feats_sp, topk=3, mutual_topk=True)
		_, sel_ind = block_overlap_prior_est.view(-1).topk(k=self.pair_num, largest=True)
		ref_sel_ind = sel_ind // block_overlap_prior_est.shape[1]
		src_sel_ind = sel_ind % block_overlap_prior_est.shape[1]
		ov_weights = block_overlap_prior_est[ref_sel_ind, src_sel_ind]

		ref_points_sp = ref_points_sp[ref_sel_ind]
		src_points_sp = src_points_sp[src_sel_ind]
		ref_feats_sp = ref_feats_sp[ref_sel_ind]
		src_feats_sp = src_feats_sp[src_sel_ind]
		ref_sp_ind = ref_sp_ind[ref_sel_ind]
		src_sp_ind = src_sp_ind[src_sel_ind]

		ref_corr_indices, src_corr_indices, corr_scores = self.matching(ref_feats_sp, src_feats_sp, self.num_correspondences)
		ref_corrs = self.select(ref_points_sp, ref_corr_indices)
		src_corrs = self.select(src_points_sp, src_corr_indices)
		Ts = weighted_procrustes(src_corrs, ref_corrs, weights=corr_scores, return_transform=True)
		corr_weights = self.corrs_weights(ref_corrs, src_corrs, corr_scores, Ts)
		ref_node_corr_indices = self.select(ref_sp_ind[..., None], ref_corr_indices).squeeze(-1)
		src_node_corr_indices = self.select(src_sp_ind[..., None], src_corr_indices).squeeze(-1)
		return Ts, corr_weights, ov_weights, ref_node_corr_indices, src_node_corr_indices, corr_scores
		# Visualization
		view_num = 9
		view_inds = corr_weights.topk(k=view_num, dim=0).indices
		scores = corr_scores[view_inds]
		c = torch.zeros(*scores.shape, 3)
		c[:, :, 0] = (scores/scores.max(dim=1).values[:, None]).clamp(min=0, max=1)
		psc = PSC().vedo(subplot=view_num)
		src = apply_transform(src_points_sp[view_inds], Ts[view_inds])
		ref = ref_points_sp[view_inds]
		ref_corr_ind = ref_corr_indices[view_inds]
		src_corr_ind = src_corr_indices[view_inds]
		for i in range(view_num):
			psc.draw_at(i).add_pcd(ref[i]) \
				.add_pcd(src[i]) \
				.add_lines(ref[i][ref_corr_ind[i]], src[i][src_corr_ind[i]], colors=c[i])
		psc.show()



class FeatureAlignmentConsistencyWeighting(torch.nn.Module):
	def __init__(self, radius, dim_f=256):
		super().__init__()
		self.radius = radius
		self.model_f = torch.nn.Sequential(
			torch.nn.Linear(dim_f, dim_f),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_f, dim_f),
			torch.nn.ReLU(),
		)
		self.model_m = torch.nn.Sequential(
			torch.nn.Linear(dim_f, dim_f),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_f, dim_f),
			torch.nn.ReLU(),
		)
		self.model_d = torch.nn.Sequential(
			torch.nn.Linear(1, dim_f),
			torch.nn.ReLU(),
		)
		self.model_w = torch.nn.Sequential(
			torch.nn.Linear(dim_f, dim_f),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_f, 1),
			torch.nn.Sigmoid(),
		)

	def forward(self, ref_points, src_points, ref_feats, src_feats, tf_est):
		B = tf_est.shape[0]
		# compute pairwise distances
		d = torch.cdist(
			ref_points.unsqueeze(0).repeat(B,1,1),
			apply_transform(src_points.unsqueeze(0).repeat(B,1,1), tf_est)
		)
		mask = d < self.radius
		if not mask.any():
			# no neighbors → default weights=1
			return torch.ones(B, device=d.device)

		b_idx, r_idx, s_idx = mask.nonzero(as_tuple=True)
		# gather features
		rf = ref_feats.unsqueeze(0).repeat(B,1,1)[b_idx, r_idx]
		sf = src_feats.unsqueeze(0).repeat(B,1,1)[b_idx, s_idx]
		fd = self.model_d(d[b_idx, r_idx, s_idx].unsqueeze(-1))
		fm = self.model_m(self.model_f(rf) - self.model_f(sf)) * fd

		# split into B chunks and take max per chunk
		# ensure chunk count = B
		chunked = fm.split([ (b_idx==i).sum().item() for i in range(B) ])
		# if any chunk is empty, treat its max as zero before sigmoid
		max_feats = torch.stack([
			c.max(dim=0).values if c.numel()>0 else torch.zeros(fm.size(1), device=fm.device)
			for c in chunked
		])
		w_m = self.model_w(max_feats).flatten()
		return w_m


class PartialRegistrationGenerator(torch.nn.Module):
	def __init__(self, pair_num=24, anchor_num=8, num_correspondences=256):
		super().__init__()
		self.anchor_num = anchor_num
		self.num_correspondences = num_correspondences
		self.pair_num = pair_num
		# self.facw = FeatureAlignmentConsistencyWeighting(radius=0.1)

	def blockwise_overlap_est(self, ref_feats_b, src_feats_b, topk=3):
		# Block-wise Overlap Estimation
		ref_sizes = [i.size(0) for i in ref_feats_b]
		src_sizes = [i.size(0) for i in src_feats_b]
		ref_feats_b_cat = torch.cat(ref_feats_b, dim=0)
		src_feats_b_cat = torch.cat(src_feats_b, dim=0)
		matching_scores = torch.exp(-pairwise_distance(ref_feats_b_cat, src_feats_b_cat, normalized=True))
		ref_matching_scores = matching_scores / matching_scores.sum(dim=-1, keepdim=True)
		src_matching_scores = matching_scores / matching_scores.sum(dim=-2, keepdim=True)
		matching_scores = ref_matching_scores * src_matching_scores
		r_len = len(ref_sizes)
		s_len = len(src_sizes)
		# block_overlap_est = torch.stack([x.mean() for i in torch.split(matching_scores, ref_sizes) for x in torch.split(i, src_sizes, dim=1)]).reshape(b_len, b_len)
		block_overlap_est = torch.stack([
			x.topk(k=topk, largest=True).values.mean() 
			for i in torch.split(matching_scores, ref_sizes) 
			for x in torch.split(i, src_sizes, dim=1)
		]).reshape(r_len, s_len)
		return block_overlap_est

	def sample(self, points, anchor_num=8, k=180):
		if k > points.shape[0]:
			k = points.shape[0]
		centroids = fps(points, ratio=anchor_num / points.shape[0])
		anchors = points[centroids]
		knn_dist, knn_inds = torch.cdist(anchors, points).topk(k=k, dim=1, largest=False)
		ind_mask = knn_dist < 3.5
		knn_inds[~ind_mask] = points.shape[0]
		return knn_inds, ind_mask

	def matching(self, ref_feats, src_feats, dual_normalization=True):
		src_size, _ = src_feats.shape
		# select top-k proposals
		matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
		if dual_normalization:
			ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
			src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
			matching_scores = ref_matching_scores * src_matching_scores
		num_correspondences = min(self.num_correspondences, matching_scores.numel())
		corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
		ref_corr_indices = corr_indices // matching_scores.shape[1]
		src_corr_indices = corr_indices % matching_scores.shape[1]
		return ref_corr_indices, src_corr_indices, corr_scores
	
	def fast_matching(self, ref_feats, src_feats, ref_sp_ind, src_sp_ind, ref_sp_mask, src_sp_mask, dual_normalization=True):
		b_size, ref_size = ref_sp_ind.shape
		_, src_size = src_sp_ind.shape

		# select top-k proposals
		matching_scores = torch.zeros(ref_feats.shape[0]+1, src_feats.shape[0]+1).to(ref_feats.device)
		matching_scores[:-1, :-1] = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
		matching_scores = matching_scores.unsqueeze(0).expand(b_size, -1, -1)
		
		ref_sp_inds = torch.cat([torch.arange(b_size)[:, None, None].repeat(1, ref_size, 1).to(ref_sp_ind.device), ref_sp_ind[..., None]], dim=-1).view(-1, 2)
		src_sp_inds = torch.cat([torch.arange(b_size)[:, None, None].repeat(1, src_size, 1).to(src_sp_ind.device), src_sp_ind[..., None]], dim=-1).view(-1, 2)
		matching_scores = matching_scores[ref_sp_inds[:, 0], ref_sp_inds[:, 1]].view(b_size, ref_size, -1)
		ref_sp_mask_b = ref_sp_mask.view(b_size, ref_size, 1).expand(-1, -1, matching_scores.shape[2])
		matching_scores = matching_scores.permute(0, 2, 1)[src_sp_inds[:, 0], src_sp_inds[:, 1]].view(b_size, src_size, ref_size)
		src_sp_mask_b = src_sp_mask.view(b_size, src_size, 1).expand(-1, -1, matching_scores.shape[2])
		matching_scores = matching_scores.permute(0, 2, 1).contiguous()

		if dual_normalization:
			ref_matching_scores = matching_scores / matching_scores.sum(dim=2, keepdim=True)
			src_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
			matching_scores = ref_matching_scores * src_matching_scores
		matching_scores[matching_scores.isnan()] = 0
		num_correspondences = min(self.num_correspondences, matching_scores[0].numel())
		corr_scores, corr_indices = matching_scores.view(b_size, -1).topk(k=num_correspondences, largest=True)
		ref_corr_indices = corr_indices // src_size
		src_corr_indices = corr_indices % src_size

		ref_node_corr_indices = self.select(ref_sp_ind[..., None], ref_corr_indices).squeeze(-1)
		src_node_corr_indices = self.select(src_sp_ind[..., None], src_corr_indices).squeeze(-1)

		return ref_node_corr_indices, src_node_corr_indices, corr_scores
	
	def overlap_estimation(self, matching_scores, ref_sp_ind, src_sp_ind, topk=3):
		r_len, ref_size = ref_sp_ind.shape
		s_len, src_size = src_sp_ind.shape

		combine_inds = torch.stack([
			ref_sp_ind.view(-1)[:, None].repeat(1, src_sp_ind.numel()), 
			src_sp_ind.view(-1)[None, :].repeat(ref_sp_ind.numel(), 1), 
		], dim=-1)
		ov_matching_scores = matching_scores[combine_inds[..., 0], combine_inds[..., 1]]
	
		ref_ov_matching_scores = ov_matching_scores / ov_matching_scores.sum(dim=-1, keepdim=True)
		src_ov_matching_scores = ov_matching_scores / ov_matching_scores.sum(dim=-2, keepdim=True)
		ov_matching_scores = ref_ov_matching_scores * src_ov_matching_scores
		block_overlap_est = torch.stack([
			x.topk(k=topk, largest=True).values.mean() 
			for i in torch.split(ov_matching_scores, ref_size) 
			for x in torch.split(i, src_size, dim=1)
		]).reshape(r_len, s_len)
		return block_overlap_est
	
	def fast_overlap_matching(self, ref_feats, src_feats, ref_sp_ind, src_sp_ind, ref_sp_mask, src_sp_mask, dual_normalization=True):
		# select top-k proposals
		matching_scores = torch.zeros(ref_feats.shape[0]+1, src_feats.shape[0]+1).to(ref_feats.device)
		matching_scores[:-1, :-1] = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
		oest = self.overlap_estimation(matching_scores, ref_sp_ind, src_sp_ind)
		_, sel_ind = oest.view(-1).topk(k=self.pair_num, largest=True)
		# if self.training:
		# 	_, sel_ind = block_overlap_prior_est.view(-1).topk(k=block_overlap_prior_est.numel(), largest=True)
		# else:
		# 	_, sel_ind = block_overlap_prior_est.view(-1).topk(k=self.pair_num, largest=True)
		ref_sel_ind = sel_ind // oest.shape[1]
		src_sel_ind = sel_ind % oest.shape[1]

		ov_weights = oest[ref_sel_ind, src_sel_ind]
		ref_sp_ind = ref_sp_ind[ref_sel_ind]
		src_sp_ind = src_sp_ind[src_sel_ind]
		ref_sp_mask = ref_sp_mask[ref_sel_ind]
		src_sp_mask = src_sp_mask[src_sel_ind]

		b_size, ref_size = ref_sp_ind.shape
		_, src_size = src_sp_ind.shape
		matching_scores = matching_scores.unsqueeze(0).expand(b_size, -1, -1)

		ref_sp_inds = torch.cat([torch.arange(b_size)[:, None, None].repeat(1, ref_size, 1).to(ref_sp_ind.device), ref_sp_ind[..., None]], dim=-1).view(-1, 2)
		src_sp_inds = torch.cat([torch.arange(b_size)[:, None, None].repeat(1, src_size, 1).to(src_sp_ind.device), src_sp_ind[..., None]], dim=-1).view(-1, 2)
		matching_scores = matching_scores[ref_sp_inds[:, 0], ref_sp_inds[:, 1]].view(b_size, ref_size, -1)
		ref_sp_mask_b = ref_sp_mask.view(b_size, ref_size, 1).expand(-1, -1, matching_scores.shape[2])
		matching_scores = matching_scores.permute(0, 2, 1)[src_sp_inds[:, 0], src_sp_inds[:, 1]].view(b_size, src_size, ref_size)
		src_sp_mask_b = src_sp_mask.view(b_size, src_size, 1).expand(-1, -1, matching_scores.shape[2])
		matching_scores = matching_scores.permute(0, 2, 1).contiguous()

		if dual_normalization:
			ref_matching_scores = matching_scores / matching_scores.sum(dim=2, keepdim=True)
			src_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
			matching_scores = ref_matching_scores * src_matching_scores
		matching_scores[matching_scores.isnan()] = 0
		num_correspondences = min(self.num_correspondences, matching_scores[0].numel())
		corr_scores, corr_indices = matching_scores.view(b_size, -1).topk(k=num_correspondences, largest=True)
		ref_corr_indices = corr_indices // src_size
		src_corr_indices = corr_indices % src_size

		ref_node_corr_indices = self.select(ref_sp_ind[..., None], ref_corr_indices).squeeze(-1)
		src_node_corr_indices = self.select(src_sp_ind[..., None], src_corr_indices).squeeze(-1)

		return ref_node_corr_indices, src_node_corr_indices, corr_scores, ov_weights

	def select(self, x, inds):
		inds = inds.unsqueeze(-1).expand(-1, -1, x.shape[-1])
		return torch.gather(x, 1, inds)
	
	def index_select(self, x, inds, mask):
		x_padded = torch.cat([x, torch.zeros_like(x[:1])], dim=0)
		x_sel = index_select(x_padded, inds, dim=0)
		x_blocks = [x_sel[i][mask[i]] for i in range(x_sel.shape[0])]
		return x_blocks

	def corrs_weights(self, ref_corrs, src_corrs, corr_scores, transform_ests):
		# corr_scores /= corr_scores.max(dim=1, keepdim=True).values
		corr_deviation = (ref_corrs - apply_transform(src_corrs, transform_ests)).square().sum(dim=-1).sqrt().mean(dim=-1)
		# weights = torch.exp(-corr_deviation.topk(k=3, dim=1).values.mean(dim=1))
		# weights = torch.exp(-corr_deviation.topk(k=3, dim=1).values.mean(dim=1))
		# weights = (1- corr_deviation.mean(dim=1)) - corr_deviation.std(dim=1)
		return torch.exp(-2*corr_deviation)
	
	def corr_err(self, ref_corrs, src_corrs, transform_ests):
		return (ref_corrs - apply_transform(src_corrs, transform_ests)).square().sum(dim=-1).sqrt()
	
	def solve_procrustes(self, ref_corrs, src_corrs, corr_scores, iterations=10):
		# Solve Procrustes
		for i in range(iterations):
			T = weighted_procrustes(src_corrs, ref_corrs, weights=corr_scores, return_transform=True)
			cerr = self.corr_err(ref_corrs, src_corrs, T)
			corr_scores = corr_scores * torch.exp(-cerr)
		return T, corr_scores

	def forward(self, ref_points, src_points, ref_feats, src_feats, aux=None):
		# Debug
		with torch.no_grad():
			failed_ind = [1, 8, 36, 39, 44, 49, 53, 56, 63, 80, 89, 97]
						#[1, 8, 24, 39, 44, 53, 56, 61, 63, 74, 80, 86, 89,
			# AUX data
			gt_tf, index = aux
			if index in failed_ind:
				pass
			if index == 852:
				pass

		ref_sp_ind, ref_sp_mask = self.sample(ref_points, self.anchor_num)
		# ref_feats_sp = self.index_select(ref_feats, ref_sp_ind, ref_sp_mask)
		# ref_points_sp = self.index_select(ref_points, ref_sp_ind, ref_sp_mask)

		src_sp_ind, src_sp_mask = self.sample(src_points, self.anchor_num)
		# src_points_sp = self.index_select(src_points, src_sp_ind, src_sp_mask)
		# src_feats_sp = self.index_select(src_feats, src_sp_ind, src_sp_mask)

		# block_overlap_prior_est = self.blockwise_overlap_est(ref_feats_sp, src_feats_sp, topk=3)
		# if self.training:
		# 	_, sel_ind = block_overlap_prior_est.view(-1).topk(k=block_overlap_prior_est.numel(), largest=True)
		# else:
		# 	_, sel_ind = block_overlap_prior_est.view(-1).topk(k=self.pair_num, largest=True)
		# ref_sel_ind = sel_ind // block_overlap_prior_est.shape[1]
		# src_sel_ind = sel_ind % block_overlap_prior_est.shape[1]

		# ov_weights = block_overlap_prior_est[ref_sel_ind, src_sel_ind]

		# ref_points_sp, _ = padding([ref_points_sp[i] for i in ref_sel_ind])
		# src_points_sp, _ = padding([src_points_sp[i] for i in src_sel_ind])
		# ref_feats_sp, _ = padding([ref_feats_sp[i] for i in ref_sel_ind])
		# src_feats_sp, _ = padding([src_feats_sp[i] for i in src_sel_ind])

		ref_corr_indices, src_corr_indices, corr_scores, ov_weights = self.fast_overlap_matching(
			ref_feats, 
			src_feats, 
			ref_sp_ind, 
			src_sp_ind,
			ref_sp_mask,
			src_sp_mask,
		)

		# ref_corr_indices, src_corr_indices, corr_scores = self.matching(ref_feats_sp, src_feats_sp)
		# corrs_list = [self.matching(rf, sf) for rf, sf in zip(ref_feats_sp, src_feats_sp)]
		# ref_corr_indices = torch.stack([i[0] for i in corrs_list])
		# src_corr_indices = torch.stack([i[1] for i in corrs_list])
		# corr_scores = torch.stack([i[2] for i in corrs_list])
		# ref_corrs = torch.stack([ref_points_sp[ind][i] for ind, i in enumerate(ref_corr_indices)])
		# src_corrs = torch.stack([src_points_sp[ind][i] for ind, i in enumerate(src_corr_indices)])
		# Ts = weighted_procrustes(src_corrs, ref_corrs, weights=corr_scores, return_transform=True)
		# Ts, corr_scores = self.solve_procrustes(ref_corrs, src_corrs, corr_scores)

		# corr_weights = self.corrs_weights(ref_corrs, src_corrs, corr_scores, Ts)
		corr_weights = None

		# inlier_mask = (ref_corrs - apply_transform(src_corrs, Ts)).square().sum(dim=-1).sqrt() < 0.1
		# ref_node_corr_indices = self.select(ref_sp_ind[..., None], ref_corr_indices).squeeze(-1)
		# src_node_corr_indices = self.select(src_sp_ind[..., None], src_corr_indices).squeeze(-1)
		
		# ref_node_corr_indices = torch.split(ref_node_corr_indices.view(-1)[inlier_mask.view(-1)], inlier_mask.sum(dim=1).tolist())
		# src_node_corr_indices = torch.split(src_node_corr_indices.view(-1)[inlier_mask.view(-1)], inlier_mask.sum(dim=1).tolist())
		# corr_scores = torch.split(corr_scores.view(-1)[inlier_mask.view(-1)], inlier_mask.sum(dim=1).tolist())
		# w_facw = self.facw(
		# 	ref_points, 
		# 	src_points,
		# 	ref_feats,
		# 	src_feats,
		# 	Ts
		# )
		w_facw = 0
		Ts = None

		# return Ts, corr_weights, ov_weights, w_facw, ref_node_corr_indices, src_node_corr_indices, corr_scores
		return Ts, corr_weights, ov_weights, w_facw, ref_corr_indices, src_corr_indices, corr_scores
		# Visualize top-k pairs
		view_num = 9
		view_inds = corr_weights.topk(k=view_num, dim=0, largest=True).indices
		scores = corr_scores[view_inds]
		c = torch.zeros(*scores.shape, 3)
		c[:, :, 0] = (scores/scores.max(dim=1).values[:, None]).clamp(min=0, max=1)
		src = apply_transform(src_points_sp[view_inds], Ts[view_inds])
		src_gt = apply_transform(src_points_sp[view_inds], gt_tf)
		ref = ref_points_sp[view_inds]
		ref_corr_ind = ref_corr_indices[view_inds]
		src_corr_ind = src_corr_indices[view_inds]
		psc = PSC().vedo(subplot=view_num)
		for i in range(view_num):
			psc.draw_at(i).add_pcd(ref[i]) \
				.add_pcd(src[i]).add_pcd(src_gt[i]) \
				.add_lines(ref[i][ref_corr_ind[i]], src[i][src_corr_ind[i]], colors=c[i])
		psc.show()

		# Visualize top-k pairs with features
		view_num = 3
		view_inds = corr_weights.topk(k=view_num, dim=0, largest=True).indices
		scores = corr_scores[view_inds]
		c = torch.zeros(*scores.shape, 3)
		c[:, :, 0] = (scores/scores.max(dim=1).values[:, None]).clamp(min=0, max=1)
		src = apply_transform(src_points_sp[view_inds], Ts[view_inds])
		ref = ref_points_sp[view_inds]
		ref_corr_ind = ref_corr_indices[view_inds]
		src_corr_ind = src_corr_indices[view_inds]
		psc = PSC().vedo(subplot=view_num+1)
		i = 0
		psc.add_pcd(torch.cat([ref_points, apply_transform(src_points, gt_tf)])).add_feat(torch.cat([ref_feats, src_feats]))
		for i in range(view_num):
			psc.draw_at(i+1).add_pcd(torch.cat([ref[i], src[i]])) \
				.add_feat(torch.cat([ref_feats_sp[view_inds][i], src_feats_sp[view_inds][i]])) \
				.add_lines(ref[i][ref_corr_ind[i]], src[i][src_corr_ind[i]], colors=c[i])
		psc.show()

		# Visualize specific pairs with features
		view_inds = 0
		scores = corr_scores[view_inds]
		c = torch.zeros(*scores.shape, 3)
		c[:, 0] = (scores/scores.max()).clamp(min=0, max=1)
		# src = apply_transform(src_points_sp[view_inds], Ts[view_inds])
		src = apply_transform(src_points_sp[view_inds], gt_tf)
		ref = ref_points_sp[view_inds]
		ref_corr_ind = ref_corr_indices[view_inds]
		src_corr_ind = src_corr_indices[view_inds]
		psc = PSC().vedo(subplot=4)
		psc.add_pcd(ref_points).add_pcd(apply_transform(src_points, gt_tf)).draw_at(1)
		psc.add_pcd(torch.cat([ref_points, apply_transform(src_points, gt_tf)])).add_feat(torch.cat([ref_feats, src_feats]))
		psc.draw_at(2).add_pcd(torch.cat([ref, src])) \
			.add_feat(torch.cat([ref_feats_sp[view_inds], src_feats_sp[view_inds]])) \
			.add_lines(ref[ref_corr_ind], src[src_corr_ind], colors=c)
		psc.draw_at(3).add_pcd(torch.cat([ref, apply_transform(src_points_sp[view_inds], gt_tf)])) \
			.add_lines(ref[ref_corr_ind], src[src_corr_ind], colors=c)
		psc.show()


		# Visualize specific pairs with features new
		view_inds = 0
		scores = corr_scores[view_inds]
		c = torch.zeros(*scores.shape, 3)
		c[:, 0] = (scores/scores.max()).clamp(min=0, max=1)
		src = apply_transform(src_points, gt_tf)
		ref = ref_points
		ref_corr_ind = ref_corr_indices[view_inds]
		src_corr_ind = src_corr_indices[view_inds]
		psc = PSC().vedo(subplot=4)
		psc.add_pcd(ref_points).add_pcd(apply_transform(src_points, gt_tf)).draw_at(1)
		# psc.add_pcd(torch.cat([ref_points, apply_transform(src_points, gt_tf)])).add_feat(torch.cat([ref_feats, src_feats]))
		psc.draw_at(3).add_pcd(torch.cat([ref, src])) \
			.add_lines(ref[ref_corr_ind], src[src_corr_ind], colors=c)
		psc.show()

