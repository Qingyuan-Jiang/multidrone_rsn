# Reconstruction Error from chamfer loss.

import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.utils.pcd_utils import *
from src.utils.env_utils import pose2mat, w2canonical, canonical2w
from src.args import TestArgs


def place_gt_pointcloud(pcd, pose_can2w_7d):
    trans_actor = pose2mat(pose_can2w_7d)
    pcd_w_torch = canonical2w(torch.from_numpy(pcd2np(pcd).reshape(-1, 3)).float(),
                              torch.from_numpy(trans_actor).float())
    pcd_can = pcd_update_points(pcd, pcd_w_torch.numpy())
    return pcd_can


def average_reconstruction_error(args, fname='circle60', niters=5, nframes=50, ngts=50, post_name='tspn'):
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rot_mat_i = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    # rot_mat_gt = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True)
    rot_mat_gt = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)

    dist_chamfer_f_list, dist_chamfer_b_list, dist_chamfer_list = [], [], []
    for i in range(niters):

        pcd_filename = save_path + '/pcd/pcd_' + '{:03d}_'.format(i) + post_name + '.pcd'
        if not os.path.exists(pcd_filename):
            break

        # Read reconstruction results.
        pcd_i = o3d.io.read_point_cloud(pcd_filename)
        pcd_i_np = rot_mat_i.apply(pcd2np(pcd_i))
        pcd_i_can = pcd_update_points(pcd_i, pcd_i_np)

        # Place ground truth back to world frame.
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt_sum_np = []
        pose_actor = np.load(save_path + '/apose_{:03d}.npy'.format(i))

        for k in range(len(args.obj_name)):

            name = args.obj_name[k]
            gt_filename_stl = save_path + '/gts/' + name + '.stl'
            if not os.path.exists(gt_filename_stl):
                break

            # Read reconstruction ground truth.
            mesh_gt = o3d.io.read_triangle_mesh(gt_filename_stl)
            pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=10000)
            pcd_gt = pcd_update_points(pcd_gt, rot_mat_gt.apply(pcd2np(pcd_gt) / 100))

            pose_i = pose_actor[k]
            pcd_w = place_gt_pointcloud(pcd_gt, pose_i)

            # frame_w = draw_frame()
            # o3d.visualization.draw_geometries([frame_w, pcd_w, pcd_i_can])
            # o3d.visualization.draw_geometries([frame_w, pcd_i_can])

            pcd_gt_sum_np.append(pcd_w.points)

        pcd_gt_sum_np = np.vstack(pcd_gt_sum_np)
        pcd_gt.points = o3d.utility.Vector3dVector(pcd_gt_sum_np)
        o3d.visualization.draw_geometries([pcd_gt, pcd_i_can])

        dist_forward = pcd_i_can.compute_point_cloud_distance(pcd_gt)
        dist_backward = pcd_gt.compute_point_cloud_distance(pcd_i_can)
        dist_chamfer = np.mean(np.concatenate((np.asarray(dist_forward), np.asarray(dist_backward))))
        dist_chamfer_f_list.append(np.mean(dist_forward) * 1000)
        dist_chamfer_b_list.append(np.mean(dist_backward) * 1000)
        dist_chamfer_list.append(dist_chamfer * 1000)

        print("Iteration: ", i, "Average: ", np.average(dist_chamfer_f_list), "Var: ", np.var((dist_chamfer_f_list)))
        print("Iteration: ", i, "Average: ", np.average(dist_chamfer_b_list), "Var: ", np.var(dist_chamfer_b_list))
        print("Iteration: ", i, "Average: ", np.average(dist_chamfer_list), "Var: ", np.var(dist_chamfer_list))
        # dist_chamfer_list.append(dist_chamfer)

    # print("All average: ", np.average(dist_chamfer_list))
    # print("All iteration Var: ", np.var(dist_chamfer_list))
    print("Average: ", np.average(dist_chamfer_f_list), "Var: ", np.var((dist_chamfer_f_list)))
    print("Average: ", np.average(dist_chamfer_b_list), "Var: ", np.var(dist_chamfer_b_list))
    print("Average: ", np.average(dist_chamfer_list), "Var: ", np.var(dist_chamfer_list))
    return dist_chamfer_list


if __name__ == '__main__':
    opts = TestArgs()
    args = opts.get_args()

    args.env_name = 'TrackSim3d-tspn'
    args.len_workspace = 20
    args.safe_rad = 6
    args.step_size_t = 0.5
    args.seed = 0
    args.create_cmap = False
    args.render = True
    args.app_thr = 0.12
    args.n_drones = 3
    args.time_sleep = 0.02
    args.obj_name = ['person_actor_%i' % (i + 1) for i in range(4)]

    average_reconstruction_error(args, args.fname, niters=1, nframes=200, post_name=args.post_name)
