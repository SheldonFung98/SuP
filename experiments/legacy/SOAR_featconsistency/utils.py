import numpy as np
import torch

from geotransformer.modules.ops import (
    grid_subsample, radius_search, grid_subsample_dps, 
    point_to_node_partition, index_select
)
from torch_cluster import fps
from geotransformer.modules.ops import pairwise_distance
import torch.nn.functional as F
from pointscope import PointScopeClient as PSC
from geotransformer.modules.ops.transformation import apply_transform


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
        64
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
    # neighbour_inds = (r2s_dist < radius).nonzero()


    # if neighbour_inds.shape[0] == 0:
    #     return None
    # feat_dist = (ref_feats[neighbour_inds[:, 0]] - src_feats[neighbour_inds[:, 1]]).square().sum(dim=1)
    # feat_dist = feat_dist[feat_dist > feat_dist_thr]
    # if feat_dist.shape[0] == 0:
    #     return None
    # topk = int(feat_dist.shape[0] * top_ratio)
    # if topk >= 1:
    #     feat_dist = feat_dist.topk(k=topk, dim=0, largest=True).values
    # topk = 3
    # feat_sim = 1 - (torch.exp((0.0001*(feat_dist**focal)))-1).topk(k=topk, dim=0, largest=True).values.sum()
    # return feat_sim

    matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
    # ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
    # src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
    # matching_scores = ref_matching_scores * src_matching_scores
    # feat_sim = matching_scores[r2s_dist < radius].sum()
    # topk = 3
    # feat_sim = matching_scores[r2s_dist < radius].topk(k=topk, largest=False).values.sum()
    return ~(matching_scores[r2s_dist < radius]<0.1).any()

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