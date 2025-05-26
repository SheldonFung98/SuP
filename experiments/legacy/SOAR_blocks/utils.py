import numpy as np
import torch

from geotransformer.modules.ops import (
    grid_subsample, radius_search, grid_subsample_dps, 
    point_to_node_partition, index_select
)
from torch_cluster import fps


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

def pcd_division_knn_radius(points, fragment_num=5):
    fps_indices = fps(points, ratio=fragment_num / points.shape[0])
    anchors = points[fps_indices]
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