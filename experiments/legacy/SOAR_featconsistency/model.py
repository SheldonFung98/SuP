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
        ref_feats_c_norm,
        src_feats_c_norm,
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
        ref_node_masks = torch.ones(ref_feats_c_norm.shape[0], dtype=torch.bool).to(ref_feats_c_norm.device)
        src_node_masks = torch.ones(src_feats_c_norm.shape[0], dtype=torch.bool).to(ref_feats_c_norm.device)
        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'].append(ref_node_corr_indices)
            output_dict['src_node_corr_indices'].append(src_node_corr_indices)

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

        output_dict['ref_node_corr_knn_points'].append(ref_node_corr_knn_points)
        output_dict['src_node_corr_knn_points'].append(src_node_corr_knn_points)
        output_dict['ref_node_corr_knn_masks'].append(ref_node_corr_knn_masks)
        output_dict['src_node_corr_knn_masks'].append(src_node_corr_knn_masks)

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'].append(matching_scores)

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
            output_dict['ref_corr_points'].append(ref_corr_points)
            output_dict['src_corr_points'].append(src_corr_points)
            output_dict['corr_scores'].append(corr_scores)

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

        # ref_overlap_est, src_overlap_est = self.pcd_overlap_est(F.normalize(ref_feats_c, p=2, dim=-1), F.normalize(src_feats_c, p=2, dim=-1))
        ref_overlap_est, src_overlap_est = utils.hierarchical_overlap_est(
            ref_points_c, ref_feats_c, src_points_c, src_feats_c,
            ref_points_m, ref_feats_m, src_points_m, src_feats_m,
            ref_points_f, ref_feats_f, src_points_f, src_feats_f
        )

        ref_centroids = utils.weighted_farthest_point_sample(ref_points_c[None, ...], 16, ref_overlap_est[None, ...])
        src_centroids = utils.weighted_farthest_point_sample(src_points_c[None, ...], 16, src_overlap_est[None, ...])

        if False:
            # Visualization unconditional features from KPConv on dense point cloud
            pcds = torch.cat([ref_points_f.cpu(), apply_transform(src_points_f, transform).cpu()+torch.tensor([2,0,0])])
            ref_feats_f = feats_f[:ref_length_f]
            src_feats_f = feats_f[ref_length_f:]
            ref_overlap_est, src_overlap_est = self.pcd_overlap_est(F.normalize(ref_feats_f, p=2, dim=-1), F.normalize(src_feats_f, p=2, dim=-1))
            ref_overlap_est = ref_overlap_est - ref_overlap_est.min()
            ref_overlap_est /= ref_overlap_est.max()
            src_overlap_est = src_overlap_est - src_overlap_est.min()
            src_overlap_est /= src_overlap_est.max()
            PSC().vedo(subplot=3)\
                .add_pcd(pcds).add_feat(feats_f)\
                .draw_at(1)\
                .add_pcd(ref_points_f).add_color(ref_overlap_est[:, None].repeat(1, 3))\
                .draw_at(2)\
                .add_pcd(apply_transform(src_points_f, transform)).add_color(src_overlap_est[:, None].repeat(1, 3))\
                .show()
            
            # Visualization unconditional features from KPConv on sparse point cloud
            pcds = torch.cat([ref_points_c.cpu(), apply_transform(src_points_c, transform).cpu()+torch.tensor([2,0,0])])
            ref_feats_c = feats_c[:ref_length_c]
            src_feats_c = feats_c[ref_length_c:]
            ref_overlap_est, src_overlap_est = self.pcd_overlap_est(F.normalize(ref_feats_c, p=2, dim=-1), F.normalize(src_feats_c, p=2, dim=-1))
            ref_overlap_est = ref_overlap_est - ref_overlap_est.min()
            ref_overlap_est /= ref_overlap_est.max()
            src_overlap_est = src_overlap_est - src_overlap_est.min()
            src_overlap_est /= src_overlap_est.max()
            PSC().vedo(subplot=3)\
                .add_pcd(pcds).add_feat(feats_c)\
                .draw_at(1)\
                .add_pcd(ref_points_c).add_color(ref_overlap_est[:, None].repeat(1, 3))\
                .add_pcd(ref_points_c[ref_centroids[0]]).add_color(torch.tensor([[1,0,0]]).repeat(ref_centroids.shape[1], 1))\
                .draw_at(2)\
                .add_pcd(apply_transform(src_points_c, transform)).add_color(src_overlap_est[:, None].repeat(1, 3))\
                .add_pcd(apply_transform(src_points_c[src_centroids[0]], transform)).add_color(torch.tensor([[1,0,0]]).repeat(src_centroids.shape[1], 1))\
                .show()


            pcds = torch.cat([ref_points_m.cpu(), apply_transform(src_points_m, transform).cpu()+torch.tensor([2,0,0])])
            feats_m = feats_list[1]
            ref_feats_m = feats_m[:ref_length_m]
            src_feats_m = feats_m[ref_length_m:]
            ref_overlap_est, src_overlap_est = self.pcd_overlap_est(F.normalize(ref_feats_m, p=2, dim=-1), F.normalize(src_feats_m, p=2, dim=-1))
            ref_overlap_est = ref_overlap_est - ref_overlap_est.min()
            ref_overlap_est /= ref_overlap_est.max()
            src_overlap_est = src_overlap_est - src_overlap_est.min()
            src_overlap_est /= src_overlap_est.max()
            PSC().vedo(subplot=3)\
                .add_pcd(pcds).add_feat(feats_m)\
                .draw_at(1)\
                .add_pcd(ref_points_m).add_color(ref_overlap_est[:, None].repeat(1, 3))\
                .draw_at(2)\
                .add_pcd(apply_transform(src_points_m, transform)).add_color(src_overlap_est[:, None].repeat(1, 3))\
                .show()


            ref_length_m = data_dict['lengths'][2][0].item()
            points_m = data_dict['points'][2].detach()
            ref_points_m = points_m[:ref_length_m]
            src_points_m = points_m[ref_length_m:]
            feats_m = feats_list[1]
            ref_feats_m = feats_m[:ref_length_m]
            src_feats_m = feats_m[ref_length_m:]
            pcds = torch.cat([ref_points_m.cpu(), apply_transform(src_points_m, transform).cpu()+torch.tensor([2,0,0])])
            pcds_f = torch.cat([ref_points_f.cpu(), apply_transform(src_points_f, transform).cpu()+torch.tensor([2,0,0])])
            pcds_m = torch.cat([ref_points_m.cpu(), apply_transform(src_points_m, transform).cpu()+torch.tensor([2,0,0])])
            pcds_c = torch.cat([ref_points_c.cpu(), apply_transform(src_points_c, transform).cpu()+torch.tensor([2,0,0])])
            ref_feats_c = feats_c[:ref_length_c]
            src_feats_c = feats_c[ref_length_c:]
            overlap_est_f = torch.cat(self.pcd_overlap_est(F.normalize(ref_feats_f, p=2, dim=-1), F.normalize(src_feats_f, p=2, dim=-1)))
            overlap_est_m = torch.cat(self.pcd_overlap_est(F.normalize(ref_feats_m, p=2, dim=-1), F.normalize(src_feats_m, p=2, dim=-1)))
            overlap_est_c = torch.cat(self.pcd_overlap_est(F.normalize(ref_feats_c, p=2, dim=-1), F.normalize(src_feats_c, p=2, dim=-1)))
            overlap_est_f = overlap_est_f - overlap_est_f.min()
            overlap_est_f /= overlap_est_f.max()
            overlap_est_m = overlap_est_m - overlap_est_m.min()
            overlap_est_m /= overlap_est_m.max()
            overlap_est_c = overlap_est_c - overlap_est_c.min()
            overlap_est_c /= overlap_est_c.max()
            PSC().vedo(subplot=6)\
                .add_pcd(pcds_f).add_feat(feats_f)\
                .draw_at(1)\
                .add_pcd(pcds_m).add_feat(feats_m)\
                .draw_at(2)\
                .add_pcd(pcds_c).add_feat(feats_c)\
                .draw_at(3)\
                .add_pcd(pcds_f).add_color(overlap_est_f[:, None].repeat(1, 3))\
                .draw_at(4)\
                .add_pcd(pcds_m).add_color(overlap_est_m[:, None].repeat(1, 3))\
                .draw_at(5)\
                .add_pcd(pcds_c).add_color(overlap_est_c[:, None].repeat(1, 3))\
                .add_pcd(apply_transform(src_points_c[src_centroids[0]], transform)).add_color(torch.tensor([[1,0,0]]).repeat(src_centroids.shape[1], 1))\
                .show()

        # ref_fc_norm = F.normalize(ref_feats_c, p=2, dim=1)
        # src_fc_norm = F.normalize(src_feats_c, p=2, dim=1)
        # p_overlap_est = self.pcd_overlap_est(ref_fc_norm, src_fc_norm)

        # _, _, ref_anchor_knn_indices, ref_anchor_knn_masks = utils.point2anchor_partition(ref_points_c)
        # _, _, ref_anchor_knn_indices, ref_anchor_knn_masks = utils.pcd_division_knn_radius(ref_points_c.cpu())
        _, _, ref_anchor_knn_indices, ref_anchor_knn_masks = utils.pcd_division_knn_radius(ref_points_c.cpu(), ref_centroids[0].cpu())
        ref_anchor_knn_indices = ref_anchor_knn_indices.to(ref_feats_c.device)
        ref_anchor_knn_masks = ref_anchor_knn_masks.to(ref_feats_c.device)

        ref_padded_c_points = torch.cat([ref_points_c, torch.zeros_like(ref_points_c[:1])], dim=0)
        ref_pc_blocks = index_select(ref_padded_c_points, ref_anchor_knn_indices, dim=0)
        ref_pc_blocks = [ref_pc_blocks[i][ref_anchor_knn_masks[i]] for i in range(ref_pc_blocks.shape[0])]

        ref_padded_c_feats = torch.cat([ref_feats_c, torch.zeros_like(ref_feats_c[:1])], dim=0)
        ref_fc_blocks = index_select(ref_padded_c_feats, ref_anchor_knn_indices, dim=0)
        ref_fc_blocks = [ref_fc_blocks[i][ref_anchor_knn_masks[i]] for i in range(ref_fc_blocks.shape[0])]

        ref_padded_c_hsv = torch.cat([ref_hsv_c, torch.zeros_like(ref_hsv_c[:1])], dim=0)
        ref_hsv_blocks = index_select(ref_padded_c_hsv, ref_anchor_knn_indices, dim=0)
        ref_hsv_blocks = [ref_hsv_blocks[i][ref_anchor_knn_masks[i]] for i in range(ref_hsv_blocks.shape[0])]

        ref_node_knn_indices_padded = torch.cat([ref_node_knn_indices, torch.zeros_like(ref_node_knn_indices[:1])], dim=0)
        ref_block_knn_indices = index_select(ref_node_knn_indices_padded, ref_anchor_knn_indices, dim=0)
        ref_block_knn_indices = [ref_block_knn_indices[i][ref_anchor_knn_masks[i]] for i in range(ref_anchor_knn_masks.shape[0])]

        ref_node_knn_masks_padded = torch.cat([ref_node_knn_masks, torch.zeros_like(ref_node_knn_masks[:1])], dim=0)
        ref_block_knn_masks = index_select(ref_node_knn_masks_padded, ref_anchor_knn_indices, dim=0)
        ref_block_knn_masks = [ref_block_knn_masks[i][ref_anchor_knn_masks[i]] for i in range(ref_anchor_knn_masks.shape[0])]

        ref_node_knn_points_padded = torch.cat([ref_node_knn_points, torch.zeros_like(ref_node_knn_points[:1])], dim=0)
        ref_block_knn_points = index_select(ref_node_knn_points_padded, ref_anchor_knn_indices, dim=0)
        ref_block_knn_points = [ref_block_knn_points[i][ref_anchor_knn_masks[i]] for i in range(ref_anchor_knn_masks.shape[0])]

        ref_overlap_est_padded = torch.cat([ref_overlap_est, torch.zeros_like(ref_overlap_est[:1])], dim=0)
        ref_overlap_est_blocks = index_select(ref_overlap_est_padded, ref_anchor_knn_indices, dim=0)
        ref_overlap_est_blocks = [ref_overlap_est_blocks[i][ref_anchor_knn_masks[i]] for i in range(ref_anchor_knn_masks.shape[0])]
        # ref_overlap_est_blocks = torch.stack(ref_overlap_est_blocks, dim=0)

        # _, _, src_anchor_knn_indices, src_anchor_knn_masks = utils.point2anchor_partition(src_points_c)
        # _, _, src_anchor_knn_indices, src_anchor_knn_masks = utils.pcd_division_knn_radius(src_points_c.cpu())
        _, _, src_anchor_knn_indices, src_anchor_knn_masks = utils.pcd_division_knn_radius(src_points_c.cpu(), src_centroids[0].cpu())
        src_anchor_knn_indices = src_anchor_knn_indices.to(src_feats_c.device)
        src_anchor_knn_masks = src_anchor_knn_masks.to(src_feats_c.device)

        src_padded_c_points = torch.cat([src_points_c, torch.zeros_like(src_points_c[:1])], dim=0)
        src_pc_blocks = index_select(src_padded_c_points, src_anchor_knn_indices, dim=0)
        src_pc_blocks = [src_pc_blocks[i][src_anchor_knn_masks[i]] for i in range(src_pc_blocks.shape[0])]

        src_padded_c_feats = torch.cat([src_feats_c, torch.zeros_like(src_feats_c[:1])], dim=0)
        src_fc_blocks = index_select(src_padded_c_feats, src_anchor_knn_indices, dim=0)
        src_fc_blocks = [src_fc_blocks[i][src_anchor_knn_masks[i]] for i in range(src_fc_blocks.shape[0])]

        src_padded_c_hsv = torch.cat([src_hsv_c, torch.zeros_like(src_hsv_c[:1])], dim=0)
        src_hsv_blocks = index_select(src_padded_c_hsv, src_anchor_knn_indices, dim=0)
        src_hsv_blocks = [src_hsv_blocks[i][src_anchor_knn_masks[i]] for i in range(src_hsv_blocks.shape[0])]

        src_node_knn_indices_padded = torch.cat([src_node_knn_indices, torch.zeros_like(src_node_knn_indices[:1])], dim=0)
        src_block_knn_indices = index_select(src_node_knn_indices_padded, src_anchor_knn_indices, dim=0)
        src_block_knn_indices = [src_block_knn_indices[i][src_anchor_knn_masks[i]] for i in range(src_anchor_knn_masks.shape[0])]

        src_node_knn_masks_padded = torch.cat([src_node_knn_masks, torch.zeros_like(src_node_knn_masks[:1])], dim=0)
        src_block_knn_masks = index_select(src_node_knn_masks_padded, src_anchor_knn_indices, dim=0)
        src_block_knn_masks = [src_block_knn_masks[i][src_anchor_knn_masks[i]] for i in range(src_anchor_knn_masks.shape[0])]

        src_node_knn_points_padded = torch.cat([src_node_knn_points, torch.zeros_like(src_node_knn_points[:1])], dim=0)
        src_block_knn_points = index_select(src_node_knn_points_padded, src_anchor_knn_indices, dim=0)
        src_block_knn_points = [src_block_knn_points[i][src_anchor_knn_masks[i]] for i in range(src_anchor_knn_masks.shape[0])]

        src_overlap_est_padded = torch.cat([src_overlap_est, torch.zeros_like(src_overlap_est[:1])], dim=0)
        src_overlap_est_blocks = index_select(src_overlap_est_padded, src_anchor_knn_indices, dim=0)
        src_overlap_est_blocks = [src_overlap_est_blocks[i][src_anchor_knn_masks[i]] for i in range(src_anchor_knn_masks.shape[0])]
        # src_overlap_est_blocks = torch.stack(src_overlap_est_blocks, dim=0)

        ref_fc_blocks_norm = [F.normalize(i.squeeze(0), p=2, dim=1) for i in ref_fc_blocks]
        src_fc_blocks_norm = [F.normalize(i.squeeze(0), p=2, dim=1) for i in src_fc_blocks]
        block_overlap_prior_est = self.blockwise_overlap_est_list(ref_fc_blocks_norm, src_fc_blocks_norm)
        # block_overlap_prior_est = torch.stack([i.sum() for i in src_overlap_est_blocks]) + torch.stack([i.sum() for i in ref_overlap_est_blocks])
        # sel_ind = (block_overlap_prior_est >= block_overlap_prior_est.view(-1).topk(k=6).values[-1]).nonzero()
        # ref_sel_ind, src_sel_ind = sel_ind[:, 0], sel_ind[:, 1]
        _, sel_ind = block_overlap_prior_est.view(-1).topk(k=16, largest=True)
        ref_sel_ind = sel_ind // block_overlap_prior_est.shape[1]
        src_sel_ind = sel_ind % block_overlap_prior_est.shape[1]

        ref_pc_blocks, ref_pc_b_mask = utils.padding([ref_pc_blocks[i] for i in ref_sel_ind])
        src_pc_blocks, src_pc_b_mask = utils.padding([src_pc_blocks[i] for i in src_sel_ind])
        ref_fc_blocks, ref_fc_b_mask = utils.padding([ref_fc_blocks[i] for i in ref_sel_ind])
        src_fc_blocks, src_fc_b_mask = utils.padding([src_fc_blocks[i] for i in src_sel_ind])
        ref_hsv_blocks, ref_hsv_b_mask = utils.padding([ref_hsv_blocks[i] for i in ref_sel_ind])
        src_hsv_blocks, src_hsv_b_mask = utils.padding([src_hsv_blocks[i] for i in src_sel_ind])

        ref_feats_c, src_feats_c = self.transformer(
            ref_pc_blocks,
            src_pc_blocks,
            ref_fc_blocks,
            src_fc_blocks,
            ref_color=ref_hsv_blocks,
            src_color=src_hsv_blocks,
        )

        ref_block_knn_points    = [ref_block_knn_points[i] for i in ref_sel_ind]
        src_block_knn_points    = [src_block_knn_points[i] for i in src_sel_ind]
        ref_block_knn_indices   = [ref_block_knn_indices[i] for i in ref_sel_ind]
        src_block_knn_indices   = [src_block_knn_indices[i] for i in src_sel_ind]
        ref_block_knn_masks     = [ref_block_knn_masks[i] for i in ref_sel_ind]
        src_block_knn_masks     = [src_block_knn_masks[i] for i in src_sel_ind]

        ref_feats_c_norm = F.normalize(ref_feats_c, p=2, dim=-1)
        src_feats_c_norm = F.normalize(src_feats_c, p=2, dim=-1)
        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # Block-wise Overlap Estimation
        block_overlap_est = self.blockwise_overlap_est(ref_feats_c_norm, src_feats_c_norm)
        if False:
            rind = 0
            sind = 2
            ref_feat = torch.cat([ref_feats_c_norm[rind],ref_feats_c_norm[1]])
            # src_feat = torch.cat(src_feats_c_norm, dim=0)
            src_feat = src_feats_c_norm[sind]
            ref = torch.cat([ref_pc_blocks[rind], ref_pc_blocks[1]])
            # src = apply_transform(torch.cat(src_pc_blocks, dim=0), transform) + torch.tensor([0.5, 0.5, 0.5]).to(ref.device)
            src = apply_transform(src_pc_blocks[sind], transform) + torch.tensor([0.5, 0.5, 0.5]).to(ref.device)
            PSC().vedo().add_pcd(torch.cat([ref, src])).add_feat(torch.cat([ref_feat, src_feat])).show()

        if False:
            psc = PSC().vedo(subplot=3).add_pcd(ref_points_c).add_pcd(src_points_c, transform).draw_at(1)
            ind = 0
            for i in range(ref_pc_blocks.shape[0]):
                psc.add_pcd(ref_pc_blocks[i])
                color = torch.zeros_like(ref_pc_blocks[i])
                if ind == i:
                    color[:, 0] = 1
                    psc.add_color(color)
                else:
                    psc.add_color(color)
            psc.draw_at(2)
            for i in range(src_pc_blocks.shape[0]):
                psc.add_pcd(src_pc_blocks[i])
                color = torch.zeros_like(src_pc_blocks[i])
                color[:, 0] = block_overlap_est[i]
                psc.add_color(color)
            psc.show()


        # fragment_num = len(ref_feats_c_norm) + len(src_feats_c_norm)
        # weights = torch.zeros(fragment_num, fragment_num).to(ref_feats_f.device)
        # Ts = torch.eye(4, 4).repeat(10, 10, 1, 1).to(ref_feats_f.device)


        # block_tfs_est = [
        #     self.blockwise_transform_est(
        #         ref_feats_c_norm[i],
        #         src_feats_c_norm[j],
        #         ref_feats_f,
        #         src_feats_f,
        #         ref_block_knn_points[i],
        #         src_block_knn_points[j],
        #         ref_block_knn_indices[i],
        #         src_block_knn_indices[j],
        #         ref_block_knn_masks[i],
        #         src_block_knn_masks[j],
        #         feats_f,
        #     ) 
        #     for i in range(len(ref_feats_c_norm)) for j in range(len(src_feats_c_norm))
        # ] # [[ref_corr_points, src_corr_points, corr_scores, estimated_transform], ...]

        output_dict['ref_node_corr_indices'] = []
        output_dict['src_node_corr_indices'] = []

        output_dict['ref_node_corr_knn_points'] = []
        output_dict['src_node_corr_knn_points'] = []
        output_dict['ref_node_corr_knn_masks'] = []
        output_dict['src_node_corr_knn_masks'] = []

        output_dict['ref_corr_points'] = []
        output_dict['src_corr_points'] = []
        output_dict['corr_scores'] = []
        output_dict['matching_scores'] = []

        # weights = torch.zeros(ref_feats_c_norm.shape[0]).to(ref_feats_f.device)
        weights = torch.zeros(ref_feats_c_norm.shape[0], 3).to(ref_feats_f.device)
        weights[:, 1] = 1
        Ts = torch.eye(4, 4).repeat(ref_feats_c_norm.shape[0], 1, 1).to(ref_feats_f.device)

        for i in range(ref_feats_c_norm.shape[0]):
            # 8.1 Generate final correspondences during testing
            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.blockwise_transform_est(
                ref_feats_c_norm[i],
                src_feats_c_norm[i],
                ref_feats_f,
                src_feats_f,
                ref_block_knn_points[i],
                src_block_knn_points[i],
                ref_block_knn_indices[i],
                src_block_knn_indices[i],
                ref_block_knn_masks[i],
                src_block_knn_masks[i],
                feats_f,
                output_dict,
            )
            # 8.2 Update weights
            # weights[i, j+5] = block_overlap_est[i, j]
            # weights[i+5, j] = block_overlap_est[i, j]
            # Ts[i, j+5] = estimated_transform
            # Ts[i+5, j] = estimated_transform.inverse()
            weights[i, 0] = block_overlap_est[i]

            wfbs = [utils.compute_feature_base_consistency(
                ref_pc_blocks[j],
                src_pc_blocks[j],
                ref_feats_c_norm[j],
                src_feats_c_norm[j],
                estimated_transform
            ) for j in range(ref_feats_c.shape[0])]


            wfbs = [utils.compute_feature_base_consistency(
                ref_pc_blocks[j],
                src_pc_blocks[j],
                ref_feats_c_norm[j],
                src_feats_c_norm[j],
                estimated_transform
            ) for j in range(ref_feats_c.shape[0])]

            wfbs = [i for i in wfbs if i is not None]
            if len(wfbs):
                w_fb = torch.stack(wfbs).min()
                weights[i, 1] = w_fb
            corr_weight = (apply_transform(src_corr_points, estimated_transform) - ref_corr_points).square().sum(dim=1).sqrt().mean()
            weights[i, 2] = torch.exp(-2*corr_weight)
            # if corr_scores.shape[0] > 0:
            #     corr_weight = (apply_transform(src_corr_points, estimated_transform) - ref_corr_points).square().sum(dim=1).sqrt().mean()
            #     weights[i] *= torch.exp(-2*corr_weight)

            Ts[i] = estimated_transform
        
        merged_weights = weights[:, 0]*weights[:, 1]#*weights[:, 2]
        # weights[:5, :5] = 1
        # weights[-5:, -5:] = 1
        N_cyclegraph = 100
        # Tglobals, weights_out = pair2globalT_cycle(weights.cpu(), Ts.cpu(), N_cyclegraph)

        sel_weights, sel_ind = merged_weights.view(-1).topk(2)
        sel_Ts = Ts.view(-1, 4, 4)[sel_ind]
        src_corr_points_sel = [output_dict['src_corr_points'][i] for i in sel_ind]
        ref_corr_points_sel = [output_dict['ref_corr_points'][i] for i in sel_ind]
        output_dict['estimated_transforms'] = Ts
        output_dict['sel_estimated_transform'] = sel_Ts

        sel_weights = torch.exp(2*(sel_weights-1))
        # T = cal(sel_weights, sel_Ts)

        T = Ts[merged_weights.max(dim=0).indices]

        # w = torch.exp(2*(sel_weights-1))
        # T = (w[..., None, None].repeat(1,1,4,4) * sel_Ts).view(-1, 4,4).sum(dim=0) / w.sum()

        # for i in range(N_cyclegraph):
        #     # Calculate axis-angle bias between each Ts and T
        #     R_est = sel_Ts.view(-1, 4, 4)[:, :3, :3]
        #     R_gt = T[:3, :3].unsqueeze(0).repeat(R_est.size(0), 1, 1)
        #     R_diff = torch.matmul(R_est, R_gt.transpose(1, 2))
        #     trace = R_diff.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        #     angle_bias = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))  # Axis-angle bias
        #     # bias = [angle_bias[j] * (apply_transform(src_corr_points_sel[j], T) - ref_corr_points_sel[j]).square().sum(dim=1).sqrt().mean() for j in range(angle_bias.shape[0])]
        #     # bias = torch.stack(bias)
        #     bias = angle_bias
        #     # angle_bias = angle_bias.reshape(5,2)
        #     sel_weights = sel_weights * torch.exp(-bias * (2*(i+1)/(N_cyclegraph*(N_cyclegraph+1))))
        #     # w = torch.exp(2*(sel_weights-1))
        #     # T = (w[..., None, None].repeat(1,1,4,4) * sel_Ts).view(-1, 4,4).sum(dim=0) / w.sum()
        #     T = cal(sel_weights, sel_Ts)

        if False:
            psc = PSC().vedo().add_pcd(ref_points_f)
            for i in Tglobals:
                psc.add_pcd(src_points_f, i)
            psc.show()

            psc = PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, T).show()
            psc = PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, T).add_pcd(src_points_f, transform).show()
        
        if True and not compute_recall(T, transform, src_points_f):
            pass
            # psc = PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, T).add_pcd(src_points_f, transform).show()

        output_dict['ref_corr_points'] = torch.cat(output_dict['ref_corr_points'])
        output_dict['src_corr_points'] = torch.cat(output_dict['src_corr_points'])
        output_dict['corr_scores'] = torch.cat(output_dict['corr_scores'])

        output_dict['ref_node_corr_indices'] = torch.stack(output_dict['ref_node_corr_indices'])
        output_dict['src_node_corr_indices'] = torch.stack(output_dict['src_node_corr_indices'])

        output_dict['estimated_transform'] = T
        return output_dict


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
