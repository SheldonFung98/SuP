import argparse
import os.path as osp
import time
import glob
import sys
import json

import torch
import numpy as np

from geotransformer.engine import Logger
from geotransformer.modules.registration import weighted_procrustes
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences
from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)
from geotransformer.datasets.registration.threedmatch.utils import (
    get_num_fragments,
    get_scene_abbr,
    get_gt_logs_and_infos,
    compute_transform_error,
    write_log_file,
)

from config import make_cfg
import pandas as pd
from easydict import EasyDict as edict


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'first', 'second', 'third', 'forth'],
                        required=True, help='test benchmark')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd'], required=True, help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser


def eval_one_epoch(args, cfg, logger):
    features_root = osp.join(cfg.feature_dir, args.benchmark)
    benchmark = args.benchmark

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter('precision')
    coarse_matching_meter.register_meter('PMR>0')
    coarse_matching_meter.register_meter('PMR>=0.1')
    coarse_matching_meter.register_meter('PMR>=0.3')
    coarse_matching_meter.register_meter('PMR>=0.5')
    coarse_matching_meter.register_meter('scene_precision')
    coarse_matching_meter.register_meter('scene_PMR>0')
    coarse_matching_meter.register_meter('scene_PMR>=0.1')
    coarse_matching_meter.register_meter('scene_PMR>=0.3')
    coarse_matching_meter.register_meter('scene_PMR>=0.5')

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter('recall')
    fine_matching_meter.register_meter('inlier_ratio')
    fine_matching_meter.register_meter('overlap')
    fine_matching_meter.register_meter('scene_recall')
    fine_matching_meter.register_meter('scene_inlier_ratio')
    fine_matching_meter.register_meter('scene_overlap')

    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('mean_rre')
    registration_meter.register_meter('mean_rte')
    registration_meter.register_meter('median_rre')
    registration_meter.register_meter('median_rte')
    registration_meter.register_meter('scene_recall')
    registration_meter.register_meter('scene_rre')
    registration_meter.register_meter('scene_rte')

    scene_coarse_matching_result_dict = {}
    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    scene_roots = sorted(glob.glob(osp.join(features_root, '*')))
    weights = []
    for scene_root in scene_roots:
        coarse_matching_meter.reset_meter('scene_precision')
        coarse_matching_meter.reset_meter('scene_PMR>0')
        coarse_matching_meter.reset_meter('scene_PMR>=0.1')
        coarse_matching_meter.reset_meter('scene_PMR>=0.3')
        coarse_matching_meter.reset_meter('scene_PMR>=0.5')

        fine_matching_meter.reset_meter('scene_recall')
        fine_matching_meter.reset_meter('scene_inlier_ratio')
        fine_matching_meter.reset_meter('scene_overlap')

        registration_meter.reset_meter('scene_recall')
        registration_meter.reset_meter('scene_rre')
        registration_meter.reset_meter('scene_rte')

        scene_name = osp.basename(scene_root)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)
        bench_name = benchmark if benchmark in ['3DMatch', '3DLoMatch'] else '3DLoMatch'
        gt_root = osp.join(cfg.data.dataset_root, 'metadata', 'benchmarks', bench_name, scene_name)
        gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_root, num_fragments)

        estimated_transforms = []

        file_names = sorted(
            glob.glob(osp.join(scene_root, '*.npz')),
            key=lambda x: [int(i) for i in osp.basename(x).split('.')[0].split('_')],
        )
        hist = {
            "success": [],
            "overlap": [],
        }
        for file_name in file_names:
            ref_frame, src_frame = [int(x) for x in osp.basename(file_name).split('.')[0].split('_')]

            data_dict = np.load(file_name)

            ref_points_c = data_dict['ref_points_c']
            src_points_c = data_dict['src_points_c']
            ref_node_corr_indices = data_dict['ref_node_corr_indices']
            src_node_corr_indices = data_dict['src_node_corr_indices']

            ref_corr_points = data_dict['ref_corr_points']
            src_corr_points = data_dict['src_corr_points']
            corr_scores = data_dict['corr_scores']

            gt_node_corr_indices = data_dict['gt_node_corr_indices']
            transform = data_dict['transform']
            pcd_overlap = data_dict['overlap']

            if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
                sel_indices = np.argsort(-corr_scores)[: args.num_corr]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]

            message = '{}, id0: {}, id1: {}, OV: {:.3f}'.format(scene_abbr, ref_frame, src_frame, pcd_overlap)

            # 1. evaluate correspondences
            # 1.1 evaluate coarse correspondences
            coarse_matching_result_dict = evaluate_sparse_correspondences(
                ref_points_c, src_points_c, ref_node_corr_indices, src_node_corr_indices, gt_node_corr_indices
            )

            coarse_precision = coarse_matching_result_dict['precision']

            coarse_matching_meter.update('scene_precision', coarse_precision)
            coarse_matching_meter.update('scene_PMR>0', float(coarse_precision > 0))
            coarse_matching_meter.update('scene_PMR>=0.1', float(coarse_precision >= 0.1))
            coarse_matching_meter.update('scene_PMR>=0.3', float(coarse_precision >= 0.3))
            coarse_matching_meter.update('scene_PMR>=0.5', float(coarse_precision >= 0.5))

            # 1.2 evaluate fine correspondences
            fine_matching_result_dict = evaluate_correspondences(
                ref_corr_points, src_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
            )

            inlier_ratio = fine_matching_result_dict['inlier_ratio']
            overlap = fine_matching_result_dict['overlap']

            fine_matching_meter.update('scene_inlier_ratio', inlier_ratio)
            fine_matching_meter.update('scene_overlap', overlap)
            fine_matching_meter.update('scene_recall', float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

            message += ', c_PIR: {:.3f}'.format(coarse_precision)
            message += ', f_IR: {:.3f}'.format(inlier_ratio)
            message += ', f_OV: {:.3f}'.format(overlap)
            message += ', f_RS: {:.3f}'.format(fine_matching_result_dict['residual'])
            message += ', f_NU: {}'.format(fine_matching_result_dict['num_corr'])

            # 2. evaluate registration
            if args.method == 'lgr':
                estimated_transform = data_dict['estimated_transform']
            elif args.method == 'ransac':
                estimated_transform = registration_with_ransac_from_correspondences(
                    src_corr_points,
                    ref_corr_points,
                    distance_threshold=cfg.ransac.distance_threshold,
                    ransac_n=cfg.ransac.num_points,
                    num_iterations=cfg.ransac.num_iterations,
                )
            elif args.method == 'svd':
                with torch.no_grad():
                    ref_corr_points = torch.from_numpy(ref_corr_points).cuda()
                    src_corr_points = torch.from_numpy(src_corr_points).cuda()
                    corr_scores = torch.from_numpy(corr_scores).cuda()
                    estimated_transform = weighted_procrustes(
                        src_corr_points, ref_corr_points, corr_scores, return_transform=True
                    )
                    estimated_transform = estimated_transform.detach().cpu().numpy()
            else:
                raise ValueError(f'Unsupported registration method: {args.method}.')

            estimated_transforms.append(
                dict(
                    test_pair=[ref_frame, src_frame],
                    num_fragments=num_fragments,
                    transform=estimated_transform,
                )
            )

            if gt_indices[ref_frame, src_frame] != -1:
                # evaluate transform (realignment error)
                gt_index = gt_indices[ref_frame, src_frame]
                transform = gt_logs[gt_index]['transform']
                covariance = gt_infos[gt_index]['covariance']
                error = compute_transform_error(transform, covariance, estimated_transform)
                message += ', r_RMSE: {:.3f}'.format(np.sqrt(error))
                accepted = error < cfg.eval.rmse_threshold ** 2
                registration_meter.update('scene_recall', float(accepted))
                if accepted:
                    rre, rte = compute_registration_error(transform, estimated_transform)
                    registration_meter.update('scene_rre', rre)
                    registration_meter.update('scene_rte', rte)
                    message += ', r_RRE: {:.3f}'.format(rre)
                    message += ', r_RTE: {:.3f}'.format(rte)
                hist['success'].append(accepted)
                hist['overlap'].append(pcd_overlap)

                weights.append(edict(
                    estimated_transform = data_dict['estimated_transform'],
                    all_est_transform = data_dict['all_est_transform'],
                    # w_facw = data_dict['w_facw'],
                    sel_est_transform = data_dict['sel_est_transform'],
                    all_weights = data_dict['all_weights'],
                    transform = transform,
                    covariance = covariance,
                    accepted = accepted,
                ))
            if args.verbose:
                logger.info(message)

        logger.info(f'Scene_name: {scene_name}')
        # est_log = osp.join(cfg.registration_dir, benchmark, scene_name, 'est.log')
        # write_log_file(est_log, estimated_transforms)

        # create_overlap_histogram(hist)
    
    evaluate_weights(weights)


def evaluate_weights(weights):
    w_facws = []
    errors = []
    weights_all = []
    aest_transform = []
    success = []
    for weight in weights:
        estimated_transform = weight.estimated_transform
        transform = weight.transform
        covariance = weight.covariance
        all_est_transform = weight.all_est_transform
        # w_facw = weight.w_facw
        sel_est_transform = weight.sel_est_transform
        all_weights = weight.all_weights

        errors.append(torch.Tensor([compute_transform_error(transform, covariance, i) for i in all_est_transform]))
        # w_facws.append(torch.from_numpy(w_facw))
        weights_all.append(torch.from_numpy(all_weights))
        aest_transform.append(torch.from_numpy(all_est_transform))
        success.append(weight.accepted)

    errors = torch.stack(errors)
    accepted = errors < 0.04
    # w_facws = torch.stack(w_facws)
    weights_all = torch.stack(weights_all)
    aest_transform = torch.stack(aest_transform)
    success = torch.tensor(success)

    rescale = True
    w0 = weights_all[..., 0]
    w0[w0.isnan()] = 0
    # w0 = w0 - w0.min(dim=1).values[..., None]
    # w0 = w0 / w0.max(dim=1).values[..., None]
    w1 = weights_all[..., 1]
    # w1 -= w1.min(dim=1).values[..., None]
    if rescale:
        w1 /= w1.max(dim=1).values[..., None]
    w2 = weights_all[..., 2]
    # w2 = w2 - w2.min(dim=1).values[..., None]
    if rescale:
        w2 = w2 / w2.max(dim=1).values[..., None]
    w3 = weights_all[..., 3]
    # w3 = w3 - w3.min(dim=1).values[..., None]
    if rescale:
        w3 = w3 / w3.max(dim=1).values[..., None]
    w4 = weights_all[..., 4]
    # w4 = w4 - w4.min(dim=1).values[..., None]
    if rescale:
        w4 = w4 / w4.max(dim=1).values[..., None]

    x = w0 + 0.4 * w1 + w4 + w3  # [1726,24]
    
    b = {}
    for i in range(x.shape[1]):
        a = accepted.gather(dim=1, index=x.topk(i+1, dim=1).indices)
        b[i] = a.sum(dim=1).unique(return_counts=True)[1][-i:].sum()/a.shape[0]


    # x = w2  # [1726,24]
    y = errors        # [1726,24]
    # compute mean & std per column
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x_std  = x.std(dim=0,  unbiased=False, keepdim=True)
    y_std  = y.std(dim=0,  unbiased=False, keepdim=True)

    x_z = (x - x_mean) / x_std    # [1726,24]
    y_z = (y - y_mean) / y_std    # [1726,24]

    # elementwise product, then average over samples
    corrs = (x_z * y_z).mean(dim=0)   # [24]
    relation = corrs.mean()
    # cov = torch.cov(torch.stack([x_z.view(-1), y_z.view(-1)]))

    # x =  0.4 * w1 + w4 + w3  # [1726,24]
    x = w0 + w1 + w4 + w3  # [1726,24]
    # x = w0 * w1 * w4 **2 * w3  # [1726,24]
    # x = w0 * w1 * w2 * w4 * w3  # [1726,24]
    # x =  w1 + w4 + w3  # [1726,24]
    # x = 0.4*w1 + w4 + w3  # [1726,24]
    x = 0.4 * w1 + w4 + w3  # [1726,24]

    a = accepted.gather(dim=1, index=x.topk(8, dim=1).indices)
    a.sum(dim=1).unique(return_counts=True)[1][-7:].sum()/a.shape[0]


    accepted.gather(dim=1, index=x.topk(1, dim=1).indices).float().mean()

    w1_sel_ind = w1.topk(6, dim=1).indices
    accepted_w1_sel = accepted.gather(dim=1, index=w1_sel_ind)

    w4_w1sel = w3.gather(dim=0, index=w1_sel_ind)

    w4_w1sel_sel_ind = w4_w1sel.topk(1, dim=1).indices
    accepted_w1sel_w4sel = accepted_w1_sel.gather(dim=1, index=w4_w1sel_sel_ind)

    w4_top16 = w4.topk(16, dim=1).indices
    w3_top16 = w3.topk(16, dim=1).indices
    eq_inds = ((w3_top16[:, None, :] - w4_top16[:, :, None]) == 0).nonzero()
    w4_top16[eq_inds[:, 0], eq_inds[:, 1]]

def create_overlap_histogram(hist):
    # import matplotlib.pyplot as plt
    # # Create a 2D histogram for success vs overlap
    # success_overlap_histogram_path = osp.join('success_vs_overlap_histogram.png')
    # plt.figure()
    # plt.hist2d(hist['overlap'], hist['success'], bins=[100, 2], range=[[0.1, 1.0], [0, 1]], cmap='Blues')
    # plt.colorbar(label='Frequency')
    # plt.xlabel('Overlap')
    # plt.ylabel('Registration Success')
    # plt.yticks([0, 1], ['Failure', 'Success'])
    # plt.title('Histogram of Success vs Overlap')
    # plt.savefig(success_overlap_histogram_path)
    # plt.close()
    # Create a DataFrame from the histogram data
    df = pd.DataFrame({
        'Overlap': hist['overlap'],
        'Success': hist['success']
    })
    # Group the data by overlap ranges
    overlap_bins = pd.cut(df['Overlap'], bins=np.arange(0.3, 1.05, 0.05), right=False)
    # overlap_bins = pd.cut(df['Overlap'], bins=np.arange(0.1, 0.32, 0.02), right=False)
    grouped = df.groupby(overlap_bins)['Success'].mean()

def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.log_dir, 'eval-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    logger = Logger(log_file=log_file)

    message = 'Command executed: ' + ' '.join(sys.argv)
    logger.info(message)
    message = 'Configs:\n' + json.dumps(cfg, indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, logger)


if __name__ == '__main__':
    main()
