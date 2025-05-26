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
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
import numpy as np
from backbone import KPConvFPN
import time
from pointscope import PointScopeClient as PSC
from geotransformer.modules.ops.transformation import apply_transform
from torch_cluster import fps
from geotransformer.modules.ops.radius_search import radius_search
from geotransformer.modules.ops.index_select import index_select
import os


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

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        anchor_num = data_dict['anchor_num']
        length_c = data_dict['lengths'][-1]
        length_f = data_dict['lengths'][1]
        length = data_dict['lengths'][0]

        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()
        
        points_c_blocks = torch.split(points_c, length_c.tolist(), dim=0)
        ref_points_c = points_c_blocks[:anchor_num]
        src_points_c = points_c_blocks[anchor_num:]
        points_f_blocks = torch.split(points_f, length_f.tolist(), dim=0)
        ref_points_f = points_f_blocks[:anchor_num]
        src_points_f = points_f_blocks[anchor_num:]
        points_blocks = torch.split(points, length.tolist(), dim=0)
        ref_points = points_blocks[:anchor_num]
        src_points = points_blocks[anchor_num:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        hsv_c = data_dict['hsv'][-1].detach()
        hsv_f = data_dict['hsv'][1].detach()
        hsv = data_dict['hsv'][0].detach()

        hsv_c_blocks = torch.split(hsv_c, length_c.tolist(), dim=0)
        ref_hsv_c = hsv_c_blocks[:anchor_num]
        src_hsv_c = hsv_c_blocks[anchor_num:]
        hsv_f_blocks = torch.split(hsv_f, length_f.tolist(), dim=0)
        ref_hsv_f = hsv_f_blocks[:anchor_num]
        src_hsv_f = hsv_f_blocks[anchor_num:]
        hsv_blocks = torch.split(hsv, length.tolist(), dim=0)
        ref_hsv = hsv_blocks[:anchor_num]
        src_hsv = hsv_blocks[anchor_num:]

        output_dict['ref_hsv_c'] = ref_hsv_c
        output_dict['src_hsv_c'] = src_hsv_c
        output_dict['ref_hsv_f'] = ref_hsv_f
        output_dict['src_hsv_f'] = src_hsv_f
        output_dict['ref_hsv'] = ref_hsv
        output_dict['src_hsv'] = src_hsv

        if False:
            psc = PSC().vedo()
            for i in ref_points_f:
                psc.add_pcd(i)
            for i in src_points_f:
                psc.add_pcd(i+torch.tensor([0.5, 0.5, 0.5]).cuda(), transform)
            psc.show()

        if False:
            gt_transform = data_dict['transform'].detach()
            PSC().vedo().add_pcd(ref_points_f).add_pcd(src_points_f, gt_transform).show()

            # Use FPS to sample 5 points from ref_points_f
            fragment_num = 5
            fps_indices = fps(ref_points_f, ratio=fragment_num / ref_points_f.shape[0])
            sampled_ref_points_f = ref_points_f[fps_indices]

            ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
            neighbor_indices = radius_search(
                sampled_ref_points_f.cpu(), 
                ref_points_f.cpu(), 
                torch.tensor([sampled_ref_points_f.shape[0]]),
                torch.tensor([ref_length_f]), 
                3.2, 
                2000
            )
            ref_points_f_blocks = index_select(ref_padded_points_f, neighbor_indices.cuda(), dim=0)


            # Use FPS to sample 5 points from src_points_f
            fps_indices_src = fps(src_points_f, ratio=fragment_num / src_points_f.shape[0])
            sampled_src_points_f = src_points_f[fps_indices_src]

            src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
            neighbor_indices_src = radius_search(
                sampled_src_points_f.cpu(),
                src_points_f.cpu(),
                torch.tensor([sampled_src_points_f.shape[0]]),
                torch.tensor([src_points_f.shape[0]]),
                3.2,
                2000
            )
            src_points_f_blocks = index_select(src_padded_points_f, neighbor_indices_src.cuda(), dim=0)

            psc = PSC().vedo(subplot=2).add_pcd(ref_points_f).add_pcd(src_points_f, gt_transform).draw_at(1)
            for i in ref_points_f_blocks:
                psc.add_pcd(i+torch.rand(1, 3).cuda()/10)
            for i in src_points_f_blocks:
                psc.add_pcd(i + torch.rand(1, 3).cuda() / 10, gt_transform)
            psc.show()

            # Save each block in ref_points_f_blocks to .ply files
            output_dir = "blocks"
            os.makedirs(output_dir, exist_ok=True)

            for i, block in enumerate(ref_points_f_blocks):
                block_np = block.cpu().numpy()
                filename = os.path.join(output_dir, f"ref_block_{i}.ply")
                with open(filename, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {block_np.shape[0]}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    for point in block_np:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            for i, block in enumerate(src_points_f_blocks):
                block_np = block.cpu().numpy()
                filename = os.path.join(output_dir, f"src_block_{i}.ply")
                with open(filename, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {block_np.shape[0]}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    for point in block_np:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")



            fragment_num = 5
            # Use FPS to sample 5 points from ref_points
            fps_indices_ref = fps(ref_points, ratio=fragment_num / ref_points.shape[0])
            sampled_ref_points = ref_points[fps_indices_ref]

            ref_padded_points = torch.cat([ref_points, torch.zeros_like(ref_points[:1])], dim=0)
            neighbor_indices_ref = radius_search(
                sampled_ref_points.cpu(),
                ref_points.cpu(),
                torch.tensor([sampled_ref_points.shape[0]]),
                torch.tensor([ref_length]),
                3.2,
                8000
            )
            ref_points_blocks = index_select(ref_padded_points, neighbor_indices_ref.cuda(), dim=0)

            # Use FPS to sample 5 points from src_points
            fps_indices_src = fps(src_points, ratio=fragment_num / src_points.shape[0])
            sampled_src_points = src_points[fps_indices_src]

            src_padded_points = torch.cat([src_points, torch.zeros_like(src_points[:1])], dim=0)
            neighbor_indices_src = radius_search(
                sampled_src_points.cpu(),
                src_points.cpu(),
                torch.tensor([sampled_src_points.shape[0]]),
                torch.tensor([src_points.shape[0]]),
                3.2,
                8000
            )
            src_points_blocks = index_select(src_padded_points, neighbor_indices_src.cuda(), dim=0)

            psc = PSC().vedo(subplot=2).add_pcd(ref_points).add_pcd(src_points, gt_transform).draw_at(1)
            for i in ref_points_blocks:
                psc.add_pcd(i + torch.rand(1, 3).cuda() / 10)
            for i in src_points_blocks:
                psc.add_pcd(i + torch.rand(1, 3).cuda() / 10, gt_transform)
            psc.show()

            output_dir = "blocks"
            os.makedirs(output_dir, exist_ok=True)
            # Save each block in ref_points_blocks to .ply files
            for i, block in enumerate(ref_points_blocks):
                block_np = block.cpu().numpy()
                filename = os.path.join(output_dir, f"ref_block_{i}_points.ply")
                with open(filename, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {block_np.shape[0]}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    for point in block_np:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")

            for i, block in enumerate(src_points_blocks):
                block_np = block.cpu().numpy()
                filename = os.path.join(output_dir, f"src_block_{i}_points.ply")
                with open(filename, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {block_np.shape[0]}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")
                    for point in block_np:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")


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
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]


        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
            ref_color=ref_hsv_c.unsqueeze(0),
            src_color=src_hsv_c.unsqueeze(0),
        )


        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)


        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

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
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


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
