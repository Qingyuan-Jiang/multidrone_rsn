# Reconstruction Error from chamfer loss.

import os
import sys

import torch
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.utils.pcd_utils import *


def exp_reconstruction_error_dynamic(fname='circle60', niters=5, nframes=50, ngts=50, post_name='tspn', save_frame=False):
    save_path = ROOT_DIR + "/archive/airsim/" + fname
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rot_mat_i = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
    # rot_mat_gt = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True)
    rot_mat_gt = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)

    dist_chamfer_f_list, dist_chamfer_b_list, dist_chamfer_list = [], [], []
    for i in range(niters):
        idx_frame_list = []
        for j in range(nframes):
            pcd_filename = save_path + '/pcd/pcd_' + '{:03d}_{:03d}_'.format(i, j) + post_name + '.pcd'
            if not os.path.exists(pcd_filename):
                break

            # Read reconstruction results.
            pcd_i = o3d.io.read_point_cloud(pcd_filename)
            pcd_i_np = rot_mat_i.apply(pcd2np(pcd_i))
            pcd_i_np = pcd_i_np - pcd_i_np.min(0)
            pcd_i_can = pcd_update_points(pcd_i, pcd_i_np)

            # Change back to canonical views.
            num_pcd = len(pcd_i.points)

            # Find the best GT frame.
            dist_chamfer_j = []
            for k in range(ngts):
                gt_filename_stl = save_path + '/gts/gt_{:03d}.stl'.format(k)
                if not os.path.exists(gt_filename_stl):
                    break

                # Read reconstruction ground truth.
                mesh_gt = o3d.io.read_triangle_mesh(gt_filename_stl)
                pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=num_pcd)

                # Align ground truth.
                pcd_gt_np = pcd2np(pcd_gt)
                pcd_gt_np_can = pcd_gt_np - np.mean(pcd_gt_np, axis=0)
                pcd_gt_np_align = rot_mat_gt.apply(pcd_gt_np_can)
                pcd_gt_np_align = pcd_gt_np_align - pcd_gt_np_align.min(0)
                pcd_gt_can = pcd_update_points(pcd_gt, pcd_gt_np_align)

                # frame_w = draw_frame()
                # o3d.visualization.draw_geometries([frame_w, pcd_gt_can, pcd_i_can])
                # o3d.visualization.draw_geometries([frame_w, pcd_i_can])

                dist_forward = pcd_i_can.compute_point_cloud_distance(pcd_gt_can)
                dist_backward = pcd_gt_can.compute_point_cloud_distance(pcd_i_can)
                dist_chamfer = np.mean(np.vstack((np.asarray(dist_forward), np.asarray(dist_backward))))
                dist_chamfer_j.append(dist_chamfer)

            # print(np.min(dist_chamfer_j))
            idx_frame = np.argmin(dist_chamfer_j)
            idx_frame_list.append(idx_frame)

            print("Iteration ", i, "frame ", j, "animation frame", idx_frame, "chamfer loss", np.min(dist_chamfer_j))
            # dist_chamfer_list.append(np.min(dist_chamfer_j))

            # # Visualization ----------------------------------------------------
            # Visualize alignment.
            gt_filename_stl_align = save_path + '/gts/gt_{:03d}.stl'.format(idx_frame)
            # Read reconstruction ground truth.
            mesh_gt = o3d.io.read_triangle_mesh(gt_filename_stl_align)
            pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=num_pcd)
            # Align ground truth.
            pcd_gt_np = pcd2np(pcd_gt)
            pcd_gt_np_can = pcd_gt_np - np.mean(pcd_gt_np, axis=0)
            pcd_gt_np_align = rot_mat_gt.apply(pcd_gt_np_can)
            pcd_gt_np_align = pcd_gt_np_align - pcd_gt_np_align.min(0)
            pcd_gt_can = pcd_update_points(pcd_gt, pcd_gt_np_align)
            #
            # frame_w = draw_frame()
            # o3d.visualization.draw_geometries([frame_w, pcd_gt_can, pcd_i_can])
            # # Visualization ----------------------------------------------------

            dist_forward = pcd_i_can.compute_point_cloud_distance(pcd_gt_can)
            dist_backward = pcd_gt_can.compute_point_cloud_distance(pcd_i_can)
            dist_chamfer = np.mean(np.concatenate((np.asarray(dist_forward), np.asarray(dist_backward))))
            # print(dist_chamfer)
            dist_chamfer_f_list.append(np.mean(dist_forward) * 1000)
            dist_chamfer_b_list.append(np.mean(dist_backward) * 1000)
            dist_chamfer_list.append(dist_chamfer * 1000)

        print("Iteration: ", i, "Average: ", np.average(dist_chamfer_f_list), "Var: ", np.var((dist_chamfer_f_list)))
        print("Iteration: ", i, "Average: ", np.average(dist_chamfer_b_list), "Var: ", np.var(dist_chamfer_b_list))
        print("Iteration: ", i, "Average: ", np.average(dist_chamfer_list), "Var: ", np.var(dist_chamfer_list))
        np.save(save_path + '/pcd/frame_' + '{:03d}_'.format(i) + post_name + '.npy', np.asarray(idx_frame_list))
        np.save(save_path + '/pcd/cd_f_' + '{:03d}_'.format(i) + post_name + '.npy', np.asarray(dist_chamfer_f_list))
        np.save(save_path + '/pcd/cd_b_' + '{:03d}_'.format(i) + post_name + '.npy', np.asarray(dist_chamfer_b_list))
        np.save(save_path + '/pcd/cd_m_' + '{:03d}_'.format(i) + post_name + '.npy', np.asarray(dist_chamfer_list))

    print("Average: ", np.average(dist_chamfer_f_list), "Var: ", np.var((dist_chamfer_f_list)))
    print("Average: ", np.average(dist_chamfer_b_list), "Var: ", np.var(dist_chamfer_b_list))
    print("Average: ", np.average(dist_chamfer_list), "Var: ", np.var(dist_chamfer_list))
    return dist_chamfer_list


if __name__ == '__main__':
    from src.args import TestArgs
    opts = TestArgs()
    args = opts.get_args()

    exp_reconstruction_error_dynamic(args.fname, niters=5, nframes=300, ngts=120, post_name=args.post_name)
