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

            # Evaluate re-alignment error
            # if ref_frame + 1 < src_frame:
            #     evaluate transform (realignment error)
            #     src_points_f = data_dict['src_points_f']
            #     error = compute_realignment_error(src_points_f, transform, estimated_transform)
            #     message += ', r_RMSE: {:.3f}'.format(error)
            #     accepted = error < config.eval_rmse_threshold
            #     registration_meter.update('scene_recall', float(accepted))
            #     if accepted:
            #         rre, rte = compute_registration_error(transform, estimated_transform)
            #         registration_meter.update('scene_rre', rre)
            #         registration_meter.update('scene_rte', rte)
            #         message += ', r_RRE: {:.3f}, r_RTE: {:.3f}'.format(rre, rte)

            if args.verbose:
                logger.info(message)

        est_log = osp.join(cfg.registration_dir, benchmark, scene_name, 'est.log')
        write_log_file(est_log, estimated_transforms)

        logger.info(f'Scene_name: {scene_name}')

        # 1. print correspondence evaluation results (one scene)
        # 1.1 coarse level statistics
        coarse_precision = coarse_matching_meter.mean('scene_precision')
        coarse_matching_recall_0 = coarse_matching_meter.mean('scene_PMR>0')
        coarse_matching_recall_1 = coarse_matching_meter.mean('scene_PMR>=0.1')
        coarse_matching_recall_3 = coarse_matching_meter.mean('scene_PMR>=0.3')
        coarse_matching_recall_5 = coarse_matching_meter.mean('scene_PMR>=0.5')
        coarse_matching_meter.update('precision', coarse_precision)
        coarse_matching_meter.update('PMR>0', coarse_matching_recall_0)
        coarse_matching_meter.update('PMR>=0.1', coarse_matching_recall_1)
        coarse_matching_meter.update('PMR>=0.3', coarse_matching_recall_3)
        coarse_matching_meter.update('PMR>=0.5', coarse_matching_recall_5)
        scene_coarse_matching_result_dict[scene_abbr] = {
            'precision': coarse_precision,
            'PMR>0': coarse_matching_recall_0,
            'PMR>=0.1': coarse_matching_recall_1,
            'PMR>=0.3': coarse_matching_recall_3,
            'PMR>=0.5': coarse_matching_recall_5,
        }

        # 1.2 fine level statistics
        recall = fine_matching_meter.mean('scene_recall')
        inlier_ratio = fine_matching_meter.mean('scene_inlier_ratio')
        overlap = fine_matching_meter.mean('scene_overlap')
        fine_matching_meter.update('recall', recall)
        fine_matching_meter.update('inlier_ratio', inlier_ratio)
        fine_matching_meter.update('overlap', overlap)
        scene_fine_matching_result_dict[scene_abbr] = {'recall': recall, 'inlier_ratio': inlier_ratio}

        message = '  Correspondence, '
        message += ', c_PIR: {:.3f}'.format(coarse_precision)
        message += ', c_PMR>0: {:.3f}'.format(coarse_matching_recall_0)
        message += ', c_PMR>=0.1: {:.3f}'.format(coarse_matching_recall_1)
        message += ', c_PMR>=0.3: {:.3f}'.format(coarse_matching_recall_3)
        message += ', c_PMR>=0.5: {:.3f}'.format(coarse_matching_recall_5)
        message += ', f_FMR: {:.3f}'.format(recall)
        message += ', f_IR: {:.3f}'.format(inlier_ratio)
        message += ', f_OV: {:.3f}'.format(overlap)
        logger.info(message)

        # 2. print registration evaluation results (one scene)
        recall = registration_meter.mean('scene_recall')
        mean_rre = registration_meter.mean('scene_rre')
        mean_rte = registration_meter.mean('scene_rte')
        median_rre = registration_meter.median('scene_rre')
        median_rte = registration_meter.median('scene_rte')
        registration_meter.update('recall', recall)
        registration_meter.update('mean_rre', mean_rre)
        registration_meter.update('mean_rte', mean_rte)
        registration_meter.update('median_rre', median_rre)
        registration_meter.update('median_rte', median_rte)

        scene_registration_result_dict[scene_abbr] = {
            'recall': recall,
            'mean_rre': mean_rre,
            'mean_rte': mean_rte,
            'median_rre': median_rre,
            'median_rte': median_rte,
        }

        message = '  Registration'
        message += ', RR: {:.3f}'.format(recall)
        message += ', mean_RRE: {:.3f}'.format(mean_rre)
        message += ', mean_RTE: {:.3f}'.format(mean_rte)
        message += ', median_RRE: {:.3f}'.format(median_rre)
        message += ', median_RTE: {:.3f}'.format(median_rte)
        logger.info(message)

    if args.test_epoch is not None:
        logger.critical('Epoch {}'.format(args.test_epoch))

    # 1. print correspondence evaluation results
    message = '  Coarse Matching'
    message += ', PIR: {:.3f}'.format(coarse_matching_meter.mean('precision'))
    message += ', PMR>0: {:.3f}'.format(coarse_matching_meter.mean('PMR>0'))
    message += ', PMR>=0.1: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.1'))
    message += ', PMR>=0.3: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.3'))
    message += ', PMR>=0.5: {:.3f}'.format(coarse_matching_meter.mean('PMR>=0.5'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_coarse_matching_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', PIR: {:.3f}'.format(result_dict['precision'])
        message += ', PMR>0: {:.3f}'.format(result_dict['PMR>0'])
        message += ', PMR>=0.1: {:.3f}'.format(result_dict['PMR>=0.1'])
        message += ', PMR>=0.3: {:.3f}'.format(result_dict['PMR>=0.3'])
        message += ', PMR>=0.5: {:.3f}'.format(result_dict['PMR>=0.5'])
        logger.critical(message)

    message = '  Fine Matching'
    message += ', FMR: {:.3f}'.format(fine_matching_meter.mean('recall'))
    message += ', IR: {:.3f}'.format(fine_matching_meter.mean('inlier_ratio'))
    message += ', OV: {:.3f}'.format(fine_matching_meter.mean('overlap'))
    message += ', std: {:.3f}'.format(fine_matching_meter.std('recall'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_fine_matching_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', FMR: {:.3f}'.format(result_dict['recall'])
        message += ', IR: {:.3f}'.format(result_dict['inlier_ratio'])
        logger.critical(message)

    # 2. print registration evaluation results
    message = '  Registration'
    message += ', RR: {:.3f}'.format(registration_meter.mean('recall'))
    message += ', mean_RRE: {:.3f}'.format(registration_meter.mean('mean_rre'))
    message += ', mean_RTE: {:.3f}'.format(registration_meter.mean('mean_rte'))
    message += ', median_RRE: {:.3f}'.format(registration_meter.mean('median_rre'))
    message += ', median_RTE: {:.3f}'.format(registration_meter.mean('median_rte'))
    logger.critical(message)
    for scene_abbr, result_dict in scene_registration_result_dict.items():
        message = '    {}'.format(scene_abbr)
        message += ', RR: {:.3f}'.format(result_dict['recall'])
        message += ', mean_RRE: {:.3f}'.format(result_dict['mean_rre'])
        message += ', mean_RTE: {:.3f}'.format(result_dict['mean_rte'])
        message += ', median_RRE: {:.3f}'.format(result_dict['median_rre'])
        message += ', median_RTE: {:.3f}'.format(result_dict['median_rte'])
        logger.critical(message)


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



"""
ColorPCR

[2025-05-09 18:51:35] [CRIT]   Coarse Matching, PIR: 0.631, PMR>0: 0.995, PMR>=0.1: 0.955, PMR>=0.3: 0.844, PMR>=0.5: 0.699
[2025-05-09 18:51:35] [CRIT]     Kitchen, PIR: 0.463, PMR>0: 0.996, PMR>=0.1: 0.901, PMR>=0.3: 0.728, PMR>=0.5: 0.484
[2025-05-09 18:51:35] [CRIT]     Home_1, PIR: 0.633, PMR>0: 0.997, PMR>=0.1: 0.969, PMR>=0.3: 0.865, PMR>=0.5: 0.747
[2025-05-09 18:51:35] [CRIT]     Home_2, PIR: 0.620, PMR>0: 0.974, PMR>=0.1: 0.943, PMR>=0.3: 0.804, PMR>=0.5: 0.709
[2025-05-09 18:51:35] [CRIT]     Hotel_1, PIR: 0.776, PMR>0: 1.000, PMR>=0.1: 0.982, PMR>=0.3: 0.959, PMR>=0.5: 0.899
[2025-05-09 18:51:35] [CRIT]     Hotel_2, PIR: 0.684, PMR>0: 0.994, PMR>=0.1: 0.937, PMR>=0.3: 0.854, PMR>=0.5: 0.766
[2025-05-09 18:51:35] [CRIT]     Hotel_3, PIR: 0.579, PMR>0: 1.000, PMR>=0.1: 0.980, PMR>=0.3: 0.776, PMR>=0.5: 0.490
[2025-05-09 18:51:35] [CRIT]     Study, PIR: 0.685, PMR>0: 1.000, PMR>=0.1: 0.983, PMR>=0.3: 0.921, PMR>=0.5: 0.800
[2025-05-09 18:51:35] [CRIT]     MIT_Lab, PIR: 0.607, PMR>0: 1.000, PMR>=0.1: 0.944, PMR>=0.3: 0.847, PMR>=0.5: 0.694
[2025-05-09 18:51:35] [CRIT]   Fine Matching, FMR: 0.965, IR: 0.510, OV: 0.873, std: 0.024
[2025-05-09 18:51:35] [CRIT]     Kitchen, FMR: 0.916, IR: 0.320
[2025-05-09 18:51:35] [CRIT]     Home_1, FMR: 0.976, IR: 0.533
[2025-05-09 18:51:35] [CRIT]     Home_2, FMR: 0.970, IR: 0.558
[2025-05-09 18:51:35] [CRIT]     Hotel_1, FMR: 0.995, IR: 0.620
[2025-05-09 18:51:35] [CRIT]     Hotel_2, FMR: 0.943, IR: 0.555
[2025-05-09 18:51:35] [CRIT]     Hotel_3, FMR: 0.959, IR: 0.451
[2025-05-09 18:51:35] [CRIT]     Study, FMR: 0.988, IR: 0.561
[2025-05-09 18:51:35] [CRIT]     MIT_Lab, FMR: 0.972, IR: 0.481
[2025-05-09 18:51:35] [CRIT]   Registration, RR: 0.882, mean_RRE: 2.538, mean_RTE: 0.075, median_RRE: 1.974, median_RTE: 0.058
[2025-05-09 18:51:35] [CRIT]     Kitchen, RR: 0.807, mean_RRE: 3.875, mean_RTE: 0.087, median_RRE: 3.087, median_RTE: 0.071
[2025-05-09 18:51:35] [CRIT]     Home_1, RR: 0.901, mean_RRE: 2.126, mean_RTE: 0.070, median_RRE: 1.695, median_RTE: 0.055
[2025-05-09 18:51:35] [CRIT]     Home_2, RR: 0.847, mean_RRE: 2.678, mean_RTE: 0.069, median_RRE: 1.928, median_RTE: 0.049
[2025-05-09 18:51:35] [CRIT]     Hotel_1, RR: 0.976, mean_RRE: 1.801, mean_RTE: 0.061, median_RRE: 1.602, median_RTE: 0.049
[2025-05-09 18:51:35] [CRIT]     Hotel_2, RR: 0.884, mean_RRE: 2.201, mean_RTE: 0.069, median_RRE: 1.813, median_RTE: 0.058
[2025-05-09 18:51:35] [CRIT]     Hotel_3, RR: 0.786, mean_RRE: 2.457, mean_RTE: 0.067, median_RRE: 1.920, median_RTE: 0.047
[2025-05-09 18:51:35] [CRIT]     Study, RR: 0.966, mean_RRE: 2.410, mean_RTE: 0.087, median_RRE: 1.843, median_RTE: 0.068
[2025-05-09 18:51:35] [CRIT]     MIT_Lab, RR: 0.886, mean_RRE: 2.753, mean_RTE: 0.087, median_RRE: 1.906, median_RTE: 0.065

SOAR
anchor_num=6 k=130 r=3.0 !!!! top-8 blocks

[2025-05-10 12:43:40] [CRIT]   Coarse Matching, PIR: 0.366, PMR>0: 0.996, PMR>=0.1: 0.920, PMR>=0.3: 0.589, PMR>=0.5: 0.267
[2025-05-10 12:43:40] [CRIT]     Kitchen, PIR: 0.293, PMR>0: 0.998, PMR>=0.1: 0.850, PMR>=0.3: 0.467, PMR>=0.5: 0.128
[2025-05-10 12:43:40] [CRIT]     Home_1, PIR: 0.357, PMR>0: 1.000, PMR>=0.1: 0.948, PMR>=0.3: 0.588, PMR>=0.5: 0.211
[2025-05-10 12:43:40] [CRIT]     Home_2, PIR: 0.405, PMR>0: 0.978, PMR>=0.1: 0.935, PMR>=0.3: 0.622, PMR>=0.5: 0.378
[2025-05-10 12:43:40] [CRIT]     Hotel_1, PIR: 0.417, PMR>0: 1.000, PMR>=0.1: 0.972, PMR>=0.3: 0.748, PMR>=0.5: 0.321
[2025-05-10 12:43:40] [CRIT]     Hotel_2, PIR: 0.384, PMR>0: 0.994, PMR>=0.1: 0.911, PMR>=0.3: 0.595, PMR>=0.5: 0.304
[2025-05-10 12:43:40] [CRIT]     Hotel_3, PIR: 0.358, PMR>0: 1.000, PMR>=0.1: 0.898, PMR>=0.3: 0.469, PMR>=0.5: 0.347
[2025-05-10 12:43:40] [CRIT]     Study, PIR: 0.343, PMR>0: 1.000, PMR>=0.1: 0.954, PMR>=0.3: 0.554, PMR>=0.5: 0.208
[2025-05-10 12:43:40] [CRIT]     MIT_Lab, PIR: 0.369, PMR>0: 1.000, PMR>=0.1: 0.889, PMR>=0.3: 0.667, PMR>=0.5: 0.236
[2025-05-10 12:43:40] [CRIT]   Fine Matching, FMR: 0.964, IR: 0.373, OV: 0.815, std: 0.027
[2025-05-10 12:43:40] [CRIT]     Kitchen, FMR: 0.901, IR: 0.237
[2025-05-10 12:43:40] [CRIT]     Home_1, FMR: 0.976, IR: 0.381
[2025-05-10 12:43:40] [CRIT]     Home_2, FMR: 0.970, IR: 0.445
[2025-05-10 12:43:40] [CRIT]     Hotel_1, FMR: 0.995, IR: 0.430
[2025-05-10 12:43:40] [CRIT]     Hotel_2, FMR: 0.949, IR: 0.398
[2025-05-10 12:43:40] [CRIT]     Hotel_3, FMR: 0.959, IR: 0.337
[2025-05-10 12:43:40] [CRIT]     Study, FMR: 0.988, IR: 0.390
[2025-05-10 12:43:40] [CRIT]     MIT_Lab, FMR: 0.972, IR: 0.364
[2025-05-10 12:43:40] [CRIT]   Registration, RR: 0.892, mean_RRE: 2.402, mean_RTE: 0.071, median_RRE: 1.913, median_RTE: 0.054
[2025-05-10 12:43:40] [CRIT]     Kitchen, RR: 0.830, mean_RRE: 3.774, mean_RTE: 0.087, median_RRE: 2.999, median_RTE: 0.069
[2025-05-10 12:43:40] [CRIT]     Home_1, RR: 0.936, mean_RRE: 2.016, mean_RTE: 0.067, median_RRE: 1.608, median_RTE: 0.047
[2025-05-10 12:43:40] [CRIT]     Home_2, RR: 0.860, mean_RRE: 2.659, mean_RTE: 0.068, median_RRE: 1.981, median_RTE: 0.046
[2025-05-10 12:43:40] [CRIT]     Hotel_1, RR: 0.971, mean_RRE: 1.741, mean_RTE: 0.058, median_RRE: 1.546, median_RTE: 0.047
[2025-05-10 12:43:40] [CRIT]     Hotel_2, RR: 0.906, mean_RRE: 2.160, mean_RTE: 0.068, median_RRE: 1.739, median_RTE: 0.054
[2025-05-10 12:43:40] [CRIT]     Hotel_3, RR: 0.786, mean_RRE: 2.147, mean_RTE: 0.057, median_RRE: 1.840, median_RTE: 0.046
[2025-05-10 12:43:40] [CRIT]     Study, RR: 0.958, mean_RRE: 2.045, mean_RTE: 0.077, median_RRE: 1.802, median_RTE: 0.061
[2025-05-10 12:43:40] [CRIT]     MIT_Lab, RR: 0.886, mean_RRE: 2.672, mean_RTE: 0.085, median_RRE: 1.792, median_RTE: 0.064
"""

"""
latest SOAR results on 3DLoMatch
[2025-05-24 15:25:50] [CRIT]   Coarse Matching, PIR: 0.811, PMR>0: 0.980, PMR>=0.1: 0.970, PMR>=0.3: 0.938, PMR>=0.5: 0.894
[2025-05-24 15:25:50] [CRIT]     Kitchen, PIR: 0.673, PMR>0: 0.962, PMR>=0.1: 0.926, PMR>=0.3: 0.861, PMR>=0.5: 0.773
[2025-05-24 15:25:50] [CRIT]     Home_1, PIR: 0.849, PMR>0: 0.990, PMR>=0.1: 0.976, PMR>=0.3: 0.965, PMR>=0.5: 0.945
[2025-05-24 15:25:50] [CRIT]     Home_2, PIR: 0.856, PMR>0: 0.970, PMR>=0.1: 0.965, PMR>=0.3: 0.961, PMR>=0.5: 0.930
[2025-05-24 15:25:50] [CRIT]     Hotel_1, PIR: 0.898, PMR>0: 1.000, PMR>=0.1: 0.991, PMR>=0.3: 0.986, PMR>=0.5: 0.968
[2025-05-24 15:25:50] [CRIT]     Hotel_2, PIR: 0.830, PMR>0: 0.975, PMR>=0.1: 0.968, PMR>=0.3: 0.937, PMR>=0.5: 0.911
[2025-05-24 15:25:50] [CRIT]     Hotel_3, PIR: 0.770, PMR>0: 0.959, PMR>=0.1: 0.959, PMR>=0.3: 0.898, PMR>=0.5: 0.837
[2025-05-24 15:25:50] [CRIT]     Study, PIR: 0.833, PMR>0: 0.996, PMR>=0.1: 0.988, PMR>=0.3: 0.954, PMR>=0.5: 0.929
[2025-05-24 15:25:50] [CRIT]     MIT_Lab, PIR: 0.777, PMR>0: 0.986, PMR>=0.1: 0.986, PMR>=0.3: 0.944, PMR>=0.5: 0.861
[2025-05-24 15:25:50] [CRIT]   Fine Matching, FMR: 0.968, IR: 0.664, OV: 0.911, std: 0.021
[2025-05-24 15:25:50] [CRIT]     Kitchen, FMR: 0.918, IR: 0.445
[2025-05-24 15:25:50] [CRIT]     Home_1, FMR: 0.979, IR: 0.719
[2025-05-24 15:25:50] [CRIT]     Home_2, FMR: 0.970, IR: 0.742
[2025-05-24 15:25:50] [CRIT]     Hotel_1, FMR: 0.991, IR: 0.756
[2025-05-24 15:25:50] [CRIT]     Hotel_2, FMR: 0.968, IR: 0.692
[2025-05-24 15:25:50] [CRIT]     Hotel_3, FMR: 0.959, IR: 0.617
[2025-05-24 15:25:50] [CRIT]     Study, FMR: 0.983, IR: 0.704
[2025-05-24 15:25:50] [CRIT]     MIT_Lab, FMR: 0.972, IR: 0.636
[2025-05-24 15:25:50] [CRIT]   Registration, RR: 0.901, mean_RRE: 2.496, mean_RTE: 0.073, median_RRE: 1.978, median_RTE: 0.055
[2025-05-24 15:25:50] [CRIT]     Kitchen, RR: 0.815, mean_RRE: 3.693, mean_RTE: 0.085, median_RRE: 3.032, median_RTE: 0.069
[2025-05-24 15:25:50] [CRIT]     Home_1, RR: 0.936, mean_RRE: 2.046, mean_RTE: 0.068, median_RRE: 1.736, median_RTE: 0.051
[2025-05-24 15:25:50] [CRIT]     Home_2, RR: 0.860, mean_RRE: 2.820, mean_RTE: 0.071, median_RRE: 2.003, median_RTE: 0.048
[2025-05-24 15:25:50] [CRIT]     Hotel_1, RR: 0.967, mean_RRE: 1.730, mean_RTE: 0.057, median_RRE: 1.526, median_RTE: 0.046
[2025-05-24 15:25:50] [CRIT]     Hotel_2, RR: 0.920, mean_RRE: 2.230, mean_RTE: 0.069, median_RRE: 1.787, median_RTE: 0.055
[2025-05-24 15:25:50] [CRIT]     Hotel_3, RR: 0.833, mean_RRE: 2.664, mean_RTE: 0.068, median_RRE: 2.179, median_RTE: 0.044
[2025-05-24 15:25:50] [CRIT]     Study, RR: 0.962, mean_RRE: 2.080, mean_RTE: 0.075, median_RRE: 1.773, median_RTE: 0.061
[2025-05-24 15:25:50] [CRIT]     MIT_Lab, RR: 0.914, mean_RRE: 2.709, mean_RTE: 0.089, median_RRE: 1.790, median_RTE: 0.062
"""