import numpy as np
import torch

from geotransformer.modules.ops import (
	grid_subsample, radius_search, grid_subsample_dps, 
	point_to_node_partition, index_select
)
from geotransformer.modules.ops import pairwise_distance
import torch.nn.functional as F
# from pointscope import PointScopeClient as PSC
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.geotransformer import SuperPointMatching
from geotransformer.modules.registration import weighted_procrustes


def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: point cloud data, [B, N, 3]
        npoint: number of points to sample
    Return:
        centroids: sampled point indices, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)  # sampled point indices
    distance = torch.full((B, N), 1e10, device=device)  # store minimum distance to sampled points
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)  # initial farthest point indices
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest].unsqueeze(1)  # [B, 1, 3]
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
        
    return centroids


class PartialRegistrationGenerator(torch.nn.Module):
	def __init__(self, pair_num=24, anchor_num=6, num_correspondences=256):
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

	def sample(self, points, anchor_num=8, k=128, radius=3.5):
		if k > points.shape[0]:
			k = points.shape[0]
			
		centroids = furthest_point_sample(points.unsqueeze(dim=0), npoint=anchor_num).squeeze(dim=0)
		anchors = points[centroids]
		knn_dist, knn_inds = torch.cdist(anchors, points).topk(k=k, dim=1, largest=False)
		ind_mask = knn_dist < radius
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
		oest[oest.isnan()] = 0
		if self.training:
			_, sel_ind = oest.view(-1).topk(k=oest.numel(), largest=True)
		else:
			_, sel_ind = oest.view(-1).topk(k=self.pair_num, largest=True)
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
		# visualize sub-clouds
		vis_ind = 0
		ref_points_color = torch.zeros_like(ref_points)
		ref_points_color[ref_sp_ind[vis_ind], 0] = 1
		PSC().vedo().add_pcd(ref_points).add_color(ref_points_color).show()

		# visualize all sub-clouds
		psc = PSC().vedo(subplot=self.anchor_num)
		for i in range(self.anchor_num):
			ref_points_color = torch.zeros_like(ref_points)
			ref_points_color[ref_sp_ind[i], 0] = 1
			psc.draw_at(i).add_pcd(ref_points).add_color(ref_points_color)
		psc.show()

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

