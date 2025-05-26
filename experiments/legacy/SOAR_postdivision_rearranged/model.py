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
from pointscope import PointScopeClient as PSC
from geotransformer.modules.ops import pairwise_distance
from Laplacian_TS import pair2globalT_cycle
from geotransformer.modules.ops.transformation import apply_transform


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

        # self.facw = utils.FeatureAlignmentConsistencyWeighting(radius=0.1)
        self.prg = utils.PartialRegistrationGenerator()

    def pcd_overlap_est(self, ref_feats, src_feats, topk=3, mutual_topk=True):
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
        src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
        matching_scores = ref_matching_scores * src_matching_scores
        ref_overlap_est = matching_scores.topk(k=topk, largest=True, dim=1).values.mean(dim=1)
        src_overlap_est = matching_scores.topk(k=topk, largest=True, dim=0).values.mean(dim=0)
        # block_overlap_est_norm = block_overlap_est - block_overlap_est.min()
        # block_overlap_est_norm /= block_overlap_est_norm.max()
        # # correspondences from reference side
        # ref_topk_scores, ref_topk_indices = matching_scores.topk(k=topk, dim=1)
        # # correspondences from source side
        # src_topk_scores, src_topk_indices = matching_scores.topk(k=topk, dim=0)
        return ref_overlap_est, src_overlap_est


    def blockwise_overlap_est_list(self, ref_feats_b, src_feats_b, topk=3, mutual_topk=False):
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
        # overlap_est = torch.zeros(len(ref_sizes), len(src_sizes))
        # for x, i in enumerate(torch.split(matching_scores, ref_sizes)):
        #     for y, j in enumerate(torch.split(i, src_sizes, dim=1)):
        #         overlap_est[x, y] = j.topk(k=topk, largest=True).values.mean()
        # block_overlap_est_norm = block_overlap_est - block_overlap_est.min()
        # block_overlap_est_norm /= block_overlap_est_norm.max()
        return block_overlap_est

    def blockwise_overlap_est(self, ref_feats_b, src_feats_b, topk=3, mutual_topk=True):
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
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

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

        # ref_overlap_est, src_overlap_est = self.pcd_overlap_est(F.normalize(ref_feats_c, p=2, dim=-1), F.normalize(src_feats_c, p=2, dim=-1))
        # ref_overlap_est, src_overlap_est = utils.hierarchical_overlap_est(
        #     ref_points_c, ref_feats_c, src_points_c, src_feats_c,
        #     ref_points_m, ref_feats_m, src_points_m, src_feats_m,
        #     ref_points_f, ref_feats_f, src_points_f, src_feats_f
        # )
        Ts, Ts_weights, ov_weights, ref_node_corr_indices, src_node_corr_indices, corr_scores = \
            self.prg(ref_points_c, src_points_c, ref_feats_c_norm, src_feats_c_norm, aux=(transform, data_dict['index']))

        division_num = min(24, Ts.shape[0])
        # sel_ind = Ts_weights.topk(division_num, largest=True).indices
        # Ts = Ts[sel_ind]
        # Ts_weights = Ts_weights[sel_ind]
        # ov_weights = ov_weights[sel_ind]
        # ref_node_corr_indices = ref_node_corr_indices[sel_ind]
        # src_node_corr_indices = src_node_corr_indices[sel_ind]
        # corr_scores = corr_scores[sel_ind]
        d_num = Ts.shape[0]

        weights = torch.zeros(d_num, 5).to(ref_feats_f.device)
        # Ts = torch.eye(4, 4).repeat(d_num, 1, 1).to(ref_feats_f.device)

        # for i in range(d_num):

        #     ref_node_corr_indices_ = ref_node_corr_indices[i]
        #     src_node_corr_indices_ = src_node_corr_indices[i]
        #     corr_scores_ = corr_scores[i]

        #     ref_corr_points, src_corr_points, _, estimated_transform = self.blockwise_transform_est(
        #         ref_node_corr_indices_,
        #         src_node_corr_indices_,
        #         corr_scores_,
        #         ref_feats_f,
        #         src_feats_f,
        #         ref_node_knn_points,
        #         src_node_knn_points,
        #         ref_node_knn_indices,
        #         src_node_knn_indices,
        #         ref_node_knn_masks,
        #         src_node_knn_masks,
        #         feats_f,
        #         output_dict
        #     )
        #     Ts[i] = estimated_transform

        #     # 8.2 Update weights
        #     # weights[i, 0] = Ts_weights[i]
        #     weights[i, 0] = ov_weights[i]
        #     corr_weight = (apply_transform(src_corr_points, estimated_transform) - ref_corr_points).square().sum(dim=1).sqrt().mean()
        #     weights[i, 1] = torch.exp(-2*corr_weight)

        #     wfbs = utils.compute_feature_base_consistency(
        #         ref_points_c,
        #         src_points_c,
        #         ref_feats_c_norm,
        #         src_feats_c_norm,
        #         estimated_transform,
        #         radius=0.1, alpha=0.05, top_k=3
        #     )
        #     weights[i, 2] = wfbs.float()
        #     wfbs_m = utils.compute_feature_base_consistency_(
        #         ref_points_m,
        #         src_points_m,
        #         ref_feats_m,
        #         src_feats_m,
        #         estimated_transform,
        #         radius=0.06, alpha=0.08, top_k=1
        #     )
        #     weights[i, 3] = wfbs_m.float()
        #     wfbs_f = utils.compute_feature_base_consistency_(
        #         ref_points_f,
        #         src_points_f,
        #         ref_feats_f,
        #         src_feats_f,
        #         estimated_transform,
        #         radius=0.06, alpha=0.08, top_k=1
        #     )
        #     weights[i, 4] = wfbs_f.float()

            # wfbs = utils.compute_feature_base_consistency(
            #     ref_points_c,
            #     src_points_c,
            #     ref_feats_c_norm,
            #     src_feats_c_norm,
            #     Ts[i],
            #     radius=0.1, alpha=0.05, top_k=3
            # )
            # weights[i, 2] = wfbs
            
        for i in range(division_num):
            # 8.2 Update weights
            weights[i, 0] = Ts_weights[i]
            weights[i, 1] = ov_weights[i]

            wfbs = utils.compute_feature_base_consistency(
                ref_points_c,
                src_points_c,
                ref_feats_c_norm,
                src_feats_c_norm,
                Ts[i],
                radius=0.1, alpha=0.05, top_k=3
            )
            weights[i, 2] = wfbs
            wfbs_m = utils.compute_feature_base_consistency_(
                ref_points_m,
                src_points_m,
                ref_feats_m,
                src_feats_m,
                Ts[i],
                radius=0.06, alpha=0.08, top_k=1
            )
            weights[i, 3] = wfbs_m
            wfbs_f = utils.compute_feature_base_consistency_(
                ref_points_f,
                src_points_f,
                ref_feats_f,
                src_feats_f,
                Ts[i],
                radius=0.06, alpha=0.08, top_k=1
            )
            weights[i, 4] = wfbs_f

        # w_facw = self.facw(
        #     ref_points_c, 
        #     src_points_c,
        #     ref_feats_c_norm,
        #     src_feats_c_norm,
        #     Ts
        # )
        # output_dict['w_facw'] = w_facw
        
        # merged_weights = weights[:, 1] * weights[:, 2] * weights[:, 3] * weights[:, 4]
        merged_weights = weights[:, 0] * weights[:, 1] * weights[:, 2] * weights[:, 3] * weights[:, 4]

        sel_weights, sel_ind = merged_weights.view(-1).topk(3)
        sel_Ts = Ts.view(-1, 4, 4)[sel_ind]
        # src_corr_points_sel = [output_dict['src_corr_points'][i] for i in sel_ind]
        # ref_corr_points_sel = [output_dict['ref_corr_points'][i] for i in sel_ind]
        T = Ts[sel_ind]
        output_dict['estimated_transforms'] = Ts
        output_dict['sel_estimated_transform'] = T

        # sel_ind = merged_weights.max(dim=0).indices
        sel_weights, sel_ind = merged_weights.view(-1).topk(4)
        ref_node_corr_indices = ref_node_corr_indices[sel_ind]
        src_node_corr_indices = src_node_corr_indices[sel_ind]
        corr_indices_stack = torch.stack([ref_node_corr_indices.view(-1), src_node_corr_indices.view(-1)]).T
        corr_indices_cat_unique = (corr_indices_stack[:, 0] * 1000 + corr_indices_stack[:, 1]).unique()
        corr_indices_unique_stack = torch.stack([corr_indices_cat_unique // 1000, corr_indices_cat_unique % 1000], dim=1)
        ref_node_corr_indices_unique = corr_indices_unique_stack[:, 0]
        src_node_corr_indices_unique = corr_indices_unique_stack[:, 1]

        corr_scores = corr_scores[sel_ind]
        # ref_node_corr_indices = torch.cat([ref_node_corr_indices[i] for i in sel_ind])
        # src_node_corr_indices = torch.cat([src_node_corr_indices[i] for i in sel_ind])
        # corr_scores = torch.cat([corr_scores[i] for i in sel_ind])

        _, _, _, estimated_transform = self.blockwise_transform_est(
            ref_node_corr_indices_unique,
            src_node_corr_indices_unique,
            corr_scores,
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
