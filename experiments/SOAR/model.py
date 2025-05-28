import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
	GeometricTransformer,
	GeometricTransformerSOAR,
	SuperPointMatching,
	SuperPointTargetGenerator,
	LocalGlobalRegistration,
)
import numpy as np
from backbone import KPConvFPN
import time
import utils
try:
	from pointscope import PointScopeClient as PSC
except ImportError:
	pass
from geotransformer.modules.ops import pairwise_distance
from Laplacian_TS import pair2globalT_cycle
from geotransformer.modules.ops.transformation import apply_transform

# from feat_module import FeatureAlignmentConsistencyWeighting, FeatureConsistencyWeighting
from fcw_module import FeatureConsistencyWeighting
from geotransformer.modules.registration import weighted_procrustes


class GeoTransformer(nn.Module):
	def __init__(self, cfg):
		super(GeoTransformer, self).__init__()
		self.num_points_in_patch = cfg.model.num_points_in_patch
		self.matching_radius = cfg.model.ground_truth_matching_radius

		self.backbone = KPConvFPN(
			cfg.backbone.input_dim+3,
			cfg.backbone.output_dim,
			cfg.backbone.init_dim,
			cfg.backbone.kernel_size,
			cfg.backbone.init_radius,
			cfg.backbone.init_sigma,
			cfg.backbone.group_norm,
		)

		self.transformer = GeometricTransformer(
			cfg.geotransformer.input_dim,
			cfg.geotransformer.output_dim,
			cfg.geotransformer.hidden_dim,
			cfg.geotransformer.num_heads,
			cfg.geotransformer.blocks,
			cfg.geotransformer.sigma_d,
			cfg.geotransformer.sigma_a,
			cfg.geotransformer.angle_k,
			reduction_a=cfg.geotransformer.reduction_a,
			sigma_hd=cfg.geotransformer.sigma_hd,
		)

		self.coarse_target = SuperPointTargetGenerator(
			cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
		)

		self.coarse_matching = SuperPointMatching(
			cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
		)

		self.fine_matching = LocalGlobalRegistration(
			cfg.fine_matching.topk,
			cfg.fine_matching.acceptance_radius,
			mutual=cfg.fine_matching.mutual,
			confidence_threshold=cfg.fine_matching.confidence_threshold,
			use_dustbin=cfg.fine_matching.use_dustbin,
			use_global_score=cfg.fine_matching.use_global_score,
			correspondence_threshold=cfg.fine_matching.correspondence_threshold,
			correspondence_limit=cfg.fine_matching.correspondence_limit,
			num_refinement_steps=cfg.fine_matching.num_refinement_steps,
		)

		self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

		self.prg = utils.PartialRegistrationGenerator()
		self.facw = FeatureConsistencyWeighting()

	def blockwise_transform_est(self, 
		ref_node_corr_indices,
		src_node_corr_indices,
		node_corr_scores,
		ref_feats_f,
		src_feats_f,
		ref_node_knn_points,
		src_node_knn_points,
		ref_node_knn_indices,
		src_node_knn_indices,
		ref_node_knn_masks,
		src_node_knn_masks,
		feats_f,
		output_dict,
	):
		# ref_node_masks = torch.ones(ref_feats_c_norm.shape[0], dtype=torch.bool).to(ref_feats_c_norm.device)
		# src_node_masks = torch.ones(src_feats_c_norm.shape[0], dtype=torch.bool).to(ref_feats_c_norm.device)
		# 6. Select topk nearest node correspondences
		with torch.no_grad():
			# ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
			#     ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
			# )

			output_dict['ref_node_corr_indices'] = ref_node_corr_indices
			output_dict['src_node_corr_indices'] = src_node_corr_indices

			# 7 Random select ground truth node correspondences during training
			# if self.training:
			#     ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
			#         gt_node_corr_indices, gt_node_corr_overlaps
			#     )

		# 7.2 Generate batched node points & feats
		ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
		src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
		ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
		src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
		ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
		src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

		ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
		src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
		ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
		src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

		output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
		output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
		output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
		output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

		# 8. Optimal transport
		matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
		matching_scores = matching_scores / feats_f.shape[1] ** 0.5
		matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

		output_dict['matching_scores'] = matching_scores

		# 9. Generate final correspondences during testing
		with torch.no_grad():
			if not self.fine_matching.use_dustbin:
				matching_scores = matching_scores[:, :-1, :-1]

			ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
				ref_node_corr_knn_points,
				src_node_corr_knn_points,
				ref_node_corr_knn_masks,
				src_node_corr_knn_masks,
				matching_scores,
				node_corr_scores,
			)
			output_dict['ref_corr_points'] = ref_corr_points
			output_dict['src_corr_points'] = src_corr_points
			output_dict['corr_scores'] = corr_scores

		return ref_corr_points, src_corr_points, corr_scores, estimated_transform

	def fast_fine_match(
		self,
		ref_node_corr_indices,
		src_node_corr_indices,
		node_corr_scores,
		ref_feats_f,
		src_feats_f,
		ref_node_knn_points,
		src_node_knn_points,
		ref_node_knn_indices,
		src_node_knn_indices,
		ref_node_knn_masks,
		src_node_knn_masks,
		feats_f,
		output_dict
		):
		batch_size = ref_node_corr_indices.shape[0]
		corr_indices_stack = torch.stack([ref_node_corr_indices.view(-1), src_node_corr_indices.view(-1)]).T
		corr_indices_cat = (corr_indices_stack[:, 0] * 1000 + corr_indices_stack[:, 1])
		corr_indices_cat_unique = corr_indices_cat.unique()
		batch2unique_indices = (corr_indices_cat_unique[None, :].eq(corr_indices_cat[:, None])).nonzero(as_tuple=True)[1]
		batch2unique_indices = batch2unique_indices.view(batch_size, -1)

		corr_indices_unique_stack = torch.stack([corr_indices_cat_unique // 1000, corr_indices_cat_unique % 1000], dim=1)
		ref_node_corr_indices_unique = corr_indices_unique_stack[:, 0]
		src_node_corr_indices_unique = corr_indices_unique_stack[:, 1]

		ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices_unique]  # (P, K)
		src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices_unique]  # (P, K)
		ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices_unique]  # (P, K)
		src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices_unique]  # (P, K)
		ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices_unique]  # (P, K, 3)
		src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices_unique]  # (P, K, 3)

		ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
		src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
		ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
		src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

		output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
		output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
		output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
		output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

		# 8. Optimal transport
		matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
		matching_scores = matching_scores / feats_f.shape[1] ** 0.5
		matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

		output_dict['matching_scores'] = matching_scores
		
		output_list = []
		with torch.no_grad():
			if not self.fine_matching.use_dustbin:
				matching_scores = matching_scores[..., :-1, :-1]
			for ind, b2u_inds in enumerate(batch2unique_indices):
				matching_scores_b = matching_scores[b2u_inds]
				node_corr_scores_b = node_corr_scores[ind]
				ref_node_corr_knn_points_b = ref_node_corr_knn_points[b2u_inds]
				src_node_corr_knn_points_b = src_node_corr_knn_points[b2u_inds]
				ref_node_corr_knn_masks_b = ref_node_corr_knn_masks[b2u_inds]
				src_node_corr_knn_masks_b = src_node_corr_knn_masks[b2u_inds]
				ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
					ref_node_corr_knn_points_b,
					src_node_corr_knn_points_b,
					ref_node_corr_knn_masks_b,
					src_node_corr_knn_masks_b,
					matching_scores_b,
					node_corr_scores_b,
				)
				output_list.append((ref_corr_points, src_corr_points, corr_scores, estimated_transform))

		return output_list

	def forward(self, data_dict):
		output_dict = {}

		# Downsample point clouds
		feats = data_dict['features'].detach()
		transform = data_dict['transform'].detach()

		ref_length_c = data_dict['lengths'][-1][0].item()
		ref_length_m = data_dict['lengths'][2][0].item()
		ref_length_f = data_dict['lengths'][1][0].item()
		ref_length = data_dict['lengths'][0][0].item()
		points_c = data_dict['points'][-1].detach()
		points_m = data_dict['points'][2].detach()
		points_f = data_dict['points'][1].detach()
		points = data_dict['points'][0].detach()
		ref_points_c = points_c[:ref_length_c]
		src_points_c = points_c[ref_length_c:]
		ref_points_m = points_m[:ref_length_m]
		src_points_m = points_m[ref_length_m:]
		ref_points_f = points_f[:ref_length_f]
		src_points_f = points_f[ref_length_f:]
		ref_points = points[:ref_length]
		src_points = points[ref_length:]

		output_dict['ref_points_c'] = ref_points_c
		output_dict['src_points_c'] = src_points_c
		output_dict['ref_points_f'] = ref_points_f
		output_dict['src_points_f'] = src_points_f
		output_dict['ref_points'] = ref_points
		output_dict['src_points'] = src_points

		hsv_c = data_dict['hsv'][-1].detach()
		hsv_f = data_dict['hsv'][1].detach()
		hsv = data_dict['hsv'][0].detach()

		ref_hsv_c = hsv_c[:ref_length_c]
		src_hsv_c = hsv_c[ref_length_c:]
		ref_hsv_f = hsv_f[:ref_length_f]
		src_hsv_f = hsv_f[ref_length_f:]
		ref_hsv = hsv[:ref_length]
		src_hsv = hsv[ref_length:]
		output_dict['ref_hsv_c'] = ref_hsv_c
		output_dict['src_hsv_c'] = src_hsv_c
		output_dict['ref_hsv_f'] = ref_hsv_f
		output_dict['src_hsv_f'] = src_hsv_f
		output_dict['ref_hsv'] = ref_hsv
		output_dict['src_hsv'] = src_hsv

		# 1. Generate ground truth node correspondences
		_, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
			ref_points_f, ref_points_c, self.num_points_in_patch
		)
		_, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
			src_points_f, src_points_c, self.num_points_in_patch
		)

		ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
		src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
		ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
		src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

		gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
			ref_points_c,
			src_points_c,
			ref_node_knn_points,
			src_node_knn_points,
			transform,
			self.matching_radius,
			ref_masks=ref_node_masks,
			src_masks=src_node_masks,
			ref_knn_masks=ref_node_knn_masks,
			src_knn_masks=src_node_knn_masks,
		)

		output_dict['gt_node_corr_indices'] = gt_node_corr_indices
		output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

		# 2. CEFE
		feats_list = self.backbone(feats, data_dict)

		feats_c = feats_list[-1]
		feats_m = feats_list[1]
		feats_f = feats_list[0]

		ref_feats_c = feats_c[:ref_length_c]
		src_feats_c = feats_c[ref_length_c:]

		ref_feats_m = feats_m[:ref_length_m]
		src_feats_m = feats_m[ref_length_m:]

		ref_feats_f = feats_f[:ref_length_f]
		src_feats_f = feats_f[ref_length_f:]

		output_dict['ref_feats_f'] = ref_feats_f
		output_dict['src_feats_f'] = src_feats_f

		ref_feats_c, src_feats_c = self.transformer(
			ref_points_c.unsqueeze(0),
			src_points_c.unsqueeze(0),
			ref_feats_c.unsqueeze(0),
			src_feats_c.unsqueeze(0),
			ref_color=ref_hsv_c.unsqueeze(0),
			src_color=src_hsv_c.unsqueeze(0),
		)

		ref_feats_c = ref_feats_c.squeeze(0)
		src_feats_c = src_feats_c.squeeze(0)

		ref_feats_c_norm = F.normalize(ref_feats_c, p=2, dim=-1)
		src_feats_c_norm = F.normalize(src_feats_c, p=2, dim=-1)
		output_dict['ref_feats_c'] = ref_feats_c_norm
		output_dict['src_feats_c'] = src_feats_c_norm

		Ts, Ts_weights, ov_weights, w_facw, ref_node_corr_indices, src_node_corr_indices, node_corr_scores = \
			self.prg(ref_points_c, src_points_c, ref_feats_c_norm, src_feats_c_norm, aux=(transform, data_dict['index']))


		d_num = node_corr_scores.shape[0]
		weights = torch.zeros(d_num, 5).to(ref_feats_c_norm.device)

		match_res = self.fast_fine_match(
			ref_node_corr_indices,
			src_node_corr_indices,
			node_corr_scores,
			ref_feats_f,
			src_feats_f,
			ref_node_knn_points,
			src_node_knn_points,
			ref_node_knn_indices,
			src_node_knn_indices,
			ref_node_knn_masks,
			src_node_knn_masks,
			feats_f,
			output_dict
		)
		Ts = torch.eye(4, 4).unsqueeze(0).repeat(d_num, 1, 1).to(ref_feats_f.device)

		for i in range(d_num):

			ref_corr_points, src_corr_points, corr_scores, estimated_transform = match_res[i]
			Ts[i] = estimated_transform
		# 	# 8.2 Update weights
		# 	# weights[i, 0] = Ts_weights[i]
			weights[i, 0] = ov_weights[i]
			corr_weight = (apply_transform(src_corr_points, estimated_transform) - ref_corr_points).square().sum(dim=1).sqrt().mean()
			weights[i, 1] = torch.exp(-2*corr_weight)
			wfbs_m = utils.compute_feature_base_consistency_(
				ref_points_m,
				src_points_m,
				ref_feats_m,
				src_feats_m,
				estimated_transform,
				radius=0.06, alpha=0.08, top_k=1
			)
			weights[i, 3] = 0 if isinstance(wfbs_m, int) else wfbs_m.float()
		
		w_facw, sel_tf_ind = self.facw(
			ref_points_m, ref_points_f,
			src_points_m, src_points_f,
			ref_feats_m, ref_feats_f,
			src_feats_m, src_feats_f,
			Ts, tf_gt=transform if self.training else None,
		)
		if self.training:
			weights = weights[sel_tf_ind]
			Ts = Ts[sel_tf_ind]

		weights[:, 4] = w_facw
		output_dict['w_facw'] = w_facw
		output_dict['all_weights'] = weights
		# w1 = weights[:, 1]
		# w1 = w1 / w1.max()
		# w3 = weights[:, 3]
		# w3 = w3 / w3.max()
		# w4 = weights[:, 4]
		# w4 = w4 / w4.max()
		# merged_weights = weights[:, 0] * weights[:, 1] * weights[:, 2] * weights[:, 3] * weights[:, 4]
		# merged_weights = 0.4 * w1 + w3 + w4
		# merged_weights = 0.4 * w1 + w3 + w4
		# with torch.no_grad():
		# 	# merged_weights = w_facw * Ts_weights * ov_weights
		# 	# merged_weights = w_facw * weights[:, 0] * weights[:, 1] * weights[:, 3] * weights[:, 4]
		# 	# merged_weights = w_facw * weights[:, 3] * weights[:, 4]
		# 	pass
		w3 = weights[:, 3]
		w3 = w3 / w3.max()
		with torch.no_grad():
			merged_weights = 0.4 * weights[:, 1] + weights[:, 4] + w3

		merged_weights[merged_weights.isnan()] = 0
		sel_weights, sel_ind = merged_weights.view(-1).topk(3)
		output_dict['estimated_transforms'] = Ts
		output_dict['sel_estimated_transform'] = Ts[sel_ind]

		if self.training:
			# top-1
			final_sel = merged_weights.max(dim=0).indices
			output_dict['estimated_transform'] = Ts[final_sel]
			ref_corr_points, src_corr_points, corr_scores, _ = match_res[final_sel]
			output_dict['ref_corr_points'] = ref_corr_points
			output_dict['src_corr_points'] = src_corr_points
			output_dict['corr_scores'] = corr_scores
			output_dict['ref_node_corr_indices'] = ref_node_corr_indices[final_sel]
			output_dict['src_node_corr_indices'] = src_node_corr_indices[final_sel]
		else:
			# Close coarse node unique merge
			dist_thr = 0.1
			sel_weights, sel_ind = merged_weights.view(-1).topk(8)
			ref_node_corr_indices_sel = ref_node_corr_indices[sel_ind]
			src_node_corr_indices_sel = src_node_corr_indices[sel_ind]
			Ts_sel = Ts[sel_ind]
			node_corr_scores_sel = node_corr_scores[sel_ind]
			ref_points_node_corr = ref_points_c[None, ...].repeat(ref_node_corr_indices_sel.shape[0], 1, 1).gather(
				dim=1, index=ref_node_corr_indices_sel[..., None].expand(-1, -1, 3)
			)
			src_points_node_corr = src_points_c[None, ...].repeat(src_node_corr_indices_sel.shape[0], 1, 1).gather(
				dim=1, index=src_node_corr_indices_sel[..., None].expand(-1, -1, 3)
			)
			src_points_node_corr_tf = apply_transform(src_points_node_corr, Ts_sel)
			corr_dists = (ref_points_node_corr - src_points_node_corr_tf).square().sum(dim=2).sqrt()
			corr_mask = corr_dists < dist_thr

			ref_node_corr_indices_merged = ref_node_corr_indices_sel[corr_mask]
			src_node_corr_indices_merged = src_node_corr_indices_sel[corr_mask]
			corr_node_scores_merged = (node_corr_scores_sel * sel_weights[:, None])[corr_mask]

			corr_indices_cat_unique = (
				ref_node_corr_indices_merged.view(-1) * 1000 + \
				src_node_corr_indices_merged.view(-1) + \
				corr_node_scores_merged.view(-1)
			).unique()
			ref_node_corr_indices_unique = (corr_indices_cat_unique // 1000).long()
			src_node_corr_indices_unique = (corr_indices_cat_unique % 1000).long()
			corr_node_scores_merged_unique = corr_indices_cat_unique - corr_indices_cat_unique.int()

			_, _, _, estimated_transform = self.blockwise_transform_est(
				ref_node_corr_indices_unique,
				src_node_corr_indices_unique,
				corr_node_scores_merged_unique,
				ref_feats_f,
				src_feats_f,
				ref_node_knn_points,
				src_node_knn_points,
				ref_node_knn_indices,
				src_node_knn_indices,
				ref_node_knn_masks,
				src_node_knn_masks,
				feats_f,
				output_dict
			)
			output_dict['estimated_transform'] = estimated_transform

		if True and not compute_recall(estimated_transform, transform, src_points_f):
			failed_ind.append(data_dict['index'])
			# psc = PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, T).add_pcd(src_points_f, transform).show()
		if True and not torch.stack([compute_recall(tf, transform, src_points_f, 0.2) for tf in Ts]).any():
			pass

		if False:
			psc = PSC().vedo().add_pcd(ref_points_f)
			for i in Ts:
				psc.add_pcd(src_points_f, i)
			psc.show()
			psc = PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, estimated_transform).show()
			psc = PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, estimated_transform).add_pcd(src_points_f, transform).show()
			psc = PSC().vedo().add_pcd(ref_points_c).add_pcd(src_points_c, estimated_transform).add_pcd(src_points_c, transform).show()

			# Visualize correspondences
			ind = 0
			sel_weights, sel_ind = merged_weights.view(-1).topk(8)
			ref_corr_points, src_corr_points, corr_scores, tf = [match_res[i] for i in sel_ind][ind]
			src_corr_points_tf = apply_transform(src_corr_points, tf)
			corr_dists = (src_corr_points_tf - ref_corr_points).square().sum(dim=1).sqrt()
			corr_mask = corr_dists < 0.1
			score_color = torch.zeros_like(ref_corr_points)
			corr_scores /= corr_scores.max()
			score_color[:, 0] = corr_scores
			psc = PSC().vedo(subplot=2)\
					.add_pcd(ref_points_f).add_pcd(src_points_f, tf)\
					.add_lines(src_corr_points_tf, ref_corr_points, colors=score_color)
			if corr_mask.any():
				psc.draw_at(1).add_pcd(ref_points_f).add_pcd(src_points_f, tf)\
					.add_lines(src_corr_points_tf[corr_mask], ref_corr_points[corr_mask], colors=score_color[corr_mask])
			psc.show()

			# Visualize merged fine correspondences
			dist_thr = 0.1
			sel_weights, sel_ind = merged_weights.view(-1).topk(8)
			ref_corr_merged, src_corr_merged, corr_scores_merged = [], [], []
			for ind, each_res in enumerate([match_res[i] for i in sel_ind]):
				ref_corr_points, src_corr_points, corr_scores, tf = each_res
				src_corr_points_tf = apply_transform(src_corr_points, tf)
				corr_dists = (src_corr_points_tf - ref_corr_points).square().sum(dim=1).sqrt()
				corr_mask = corr_dists < dist_thr
				if corr_mask.any():
					ref_corr_merged.append(ref_corr_points[corr_mask])
					src_corr_merged.append(src_corr_points[corr_mask])
					corr_scores_merged.append(corr_scores[corr_mask] * sel_weights[ind])
			if len(ref_corr_merged) > 0:
				ref_corr_merged = torch.cat(ref_corr_merged, dim=0)
				src_corr_merged = torch.cat(src_corr_merged, dim=0)
				corr_scores_merged = torch.cat(corr_scores_merged, dim=0)
				src_corr_points_tf = apply_transform(src_corr_merged, transform)
				score_color = torch.zeros_like(ref_corr_merged)
				score_color[:, 0] = corr_scores_merged / corr_scores_merged.max()
				PSC().vedo()\
					.add_pcd(ref_points_f).add_pcd(src_points_f, transform)\
					.add_lines(src_corr_points_tf, ref_corr_merged, colors=score_color).show()

			# Visualize merged coarse correspondences
			dist_thr = 0.08
			reweight = False
			sel_weights, sel_ind = merged_weights.view(-1).topk(8)
			ref_node_corr_indices_sel = ref_node_corr_indices[sel_ind]
			src_node_corr_indices_sel = src_node_corr_indices[sel_ind]
			ref_points_node_corr = ref_points_c[None, ...].repeat(ref_node_corr_indices_sel.shape[0], 1, 1).gather(
				dim=1, index=ref_node_corr_indices_sel[..., None].expand(-1, -1, 3)
			)
			src_points_node_corr = src_points_c[None, ...].repeat(src_node_corr_indices_sel.shape[0], 1, 1).gather(
				dim=1, index=src_node_corr_indices_sel[..., None].expand(-1, -1, 3)
			)
			src_points_node_corr_tf = apply_transform(src_points_node_corr, Ts[sel_ind])
			corr_dists = (ref_points_node_corr - src_points_node_corr_tf).square().sum(dim=2).sqrt()
			corr_mask = corr_dists < dist_thr

			ref_node_corr_indices_merged = ref_node_corr_indices_sel[corr_mask]
			src_node_corr_indices_merged = src_node_corr_indices_sel[corr_mask]
			corr_node_scores_merged = (node_corr_scores[sel_ind] * (sel_weights[:, None] if reweight else 1))[corr_mask]
			score_color = torch.zeros(corr_node_scores_merged.shape[0], 3).to(corr_node_scores_merged.device)
			score_color[:, 0] = corr_node_scores_merged / corr_node_scores_merged.max()
			PSC().vedo()\
				.add_pcd(ref_points_c).add_pcd(src_points_c, transform)\
				.add_lines(ref_points_c[ref_node_corr_indices_merged], apply_transform(src_points_c, transform)[src_node_corr_indices_merged], colors=score_color).show()

		return output_dict
failed_ind = []


def compute_recall(est_transform, gt_transform, src_points, acceptance_rmse=0.2):
	realignment_transform = torch.matmul(torch.inverse(gt_transform), est_transform)
	realigned_src_points_f = apply_transform(src_points, realignment_transform)
	rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
	return torch.lt(rmse, acceptance_rmse).float()

def cal(w: torch.Tensor, RT: torch.Tensor) -> torch.Tensor:
	"""
	Compute the weighted‐mean rigid transform in PyTorch.

	Parameters
	----------
	w : Tensor, shape (N,)
		Non-negative weights.
	RT : Tensor, shape (N,4,4)
		Sequence of N homogeneous transforms [R|T;0|1].

	Returns
	-------
	RT_mean : Tensor, shape (4,4)
		The weighted‐mean rigid transform.
	"""
	# ensure float
	w = w.to(dtype=RT.dtype)

	# 1) extract rotations Rs and translations Ts
	Rs = RT[:, :3, :3]            # (N,3,3)
	Ts = RT[:, :3, 3]             # (N,3)

	# 2) weighted rotation mean
	M = torch.tensordot(w, Rs, dims=([0],[0]))     # :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
	U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # :contentReference[oaicite:10]{index=10}
	D = torch.diag(torch.tensor([1.0, 1.0, torch.det(U @ Vh)], dtype=RT.dtype, device=RT.device))
	R_mean = U @ D @ Vh

	# 3) weighted translation mean
	T_mean = torch.tensordot(w, Ts, dims=([0],[0])) / w.sum()  # :contentReference[oaicite:11]{index=11}

	# 4) assemble
	RT_mean = torch.eye(4, dtype=RT.dtype, device=RT.device)
	RT_mean[:3, :3] = R_mean
	RT_mean[:3, 3] = T_mean

	return RT_mean


def create_model(config):
	model = GeoTransformer(config)
	return model


def main():
	from config import make_cfg

	cfg = make_cfg()
	model = create_model(cfg)
	print(model.state_dict().keys())
	print(model)


if __name__ == '__main__':
	main()
