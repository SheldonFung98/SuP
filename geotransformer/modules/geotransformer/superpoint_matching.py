import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance


class SuperPointMatching(nn.Module):
	def __init__(self, num_correspondences, dual_normalization=True):
		super(SuperPointMatching, self).__init__()
		self.num_correspondences = num_correspondences
		self.dual_normalization = dual_normalization

	def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
		r"""Extract superpoint correspondences.

		Args:
			ref_feats (Tensor): features of the superpoints in reference point cloud.
			src_feats (Tensor): features of the superpoints in source point cloud.
			ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
			src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

		Returns:
			ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
			src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
			corr_scores (Tensor): scores of the correspondences.
		"""
		if ref_masks is None:
			ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
		if src_masks is None:
			src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
		# remove empty patch
		ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
		src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
		ref_feats = ref_feats[ref_indices]
		src_feats = src_feats[src_indices]
		# select top-k proposals
		matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
		if self.dual_normalization:
			ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
			src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
			matching_scores = ref_matching_scores * src_matching_scores
		num_correspondences = min(self.num_correspondences, matching_scores.numel())
		corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
		ref_sel_indices = corr_indices // matching_scores.shape[1]
		src_sel_indices = corr_indices % matching_scores.shape[1]
		# recover original indices
		ref_corr_indices = ref_indices[ref_sel_indices]
		src_corr_indices = src_indices[src_sel_indices]


		return ref_corr_indices, src_corr_indices, corr_scores



class SuperPointMatchingBatch(nn.Module):
	def __init__(self, num_correspondences, dual_normalization=True):
		super().__init__()
		self.num_correspondences = num_correspondences
		self.dual_normalization = dual_normalization

	def forward(self, ref_feats, src_feats, ref_masks=None, src_masks=None):
		r"""Extract superpoint correspondences.

		Args:
			ref_feats (Tensor): features of the superpoints in reference point cloud.
			src_feats (Tensor): features of the superpoints in source point cloud.
			ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
			src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).

		Returns:
			ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
			src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
			corr_scores (Tensor): scores of the correspondences.
		"""
		if ref_masks is None:
			ref_masks = torch.ones_like(ref_feats[:, :, 0], dtype=torch.bool)
		if src_masks is None:
			src_masks = torch.ones_like(src_feats[:, :, 0], dtype=torch.bool)

		# construct matching_scores_mask from ref_masks and src_masks
		matching_scores_mask = torch.einsum('ij,ik->ijk', ref_masks, src_masks)
		bsize, src_size, _ = src_feats.shape
		# select top-k proposals
		matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
		if self.dual_normalization:
			ref_matching_scores = matching_scores / matching_scores.sum(dim=2, keepdim=True)
			src_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
			matching_scores = ref_matching_scores * src_matching_scores
		num_correspondences = min(self.num_correspondences, matching_scores[0].numel())
		# fill the matching_scores with -inf where the mask is 0
		matching_scores = matching_scores.masked_fill(matching_scores_mask == 0, -float('inf'))
		corr_scores, corr_indices = matching_scores.view(bsize, -1).topk(k=num_correspondences, largest=True, dim=1)
		ref_corr_indices = corr_indices // src_size
		src_corr_indices = corr_indices % src_size
		return ref_corr_indices, src_corr_indices, corr_scores
