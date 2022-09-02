import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.utils.env_utils import canonical2w, setup_dir
from src.utils.plt_utils import draw_cuboid, draw_axis, set_axes_full
from src.envs.vp_drone import VPDroneSimCV


class ViewPlanningMultiActor:

    def __init__(self, args, client, mode='cv'):
        print("#################### CREATE TRACKING ENV ####################")
        self.args = args
        self.c = client
        self.obj_name = args.obj_name
        self.n_drones = args.n_drones
        self.scale = args.scale
        self.full_ws = args.len_workspace * self.scale
        self.safe_rad = args.safe_rad * self.scale
        self.step_size_t = args.step_size_t
        self.ppa_thr = self.args.ppa_thr / self.scale
        self.mode = mode

        self.pos_patches_can = torch.tensor([[0.25, 0, 1],
                                             [0., -0.5, 1],
                                             [0., 0.5, 1],
                                             [0., 0, 2],
                                             [-0.25, 0, 1]]) * self.scale
        self.ori_patches_can = torch.tensor([[1, 0, 0],
                                             [0, -1, 0],
                                             [0, 1, 0],
                                             [0, 0, 1],
                                             [-1, 0, 0]]).float()

        # if mode == 'cv':
        #     self._sim_update_states_actor(self.obj_name)
        #     self.pos_drones_sim = [torch.tensor([0, 0 + i * 2, -20]) for i in range(self.n_drones)]
        #     self.pos_drones_sim = torch.vstack(self.pos_drones_sim)
        # else:
        #     self._sim_update_states_actor(self.obj_name)
        #     self.pos_drones_sim = []
        #     for k in range(self.n_drones):
        #         pose_di_sim = self.c.getMultirotorState(vehicle_name='drone_' + str(k + 1))
        #         pos_di_sim = pose_di_sim.kinematics_estimated.position.to_numpy_array()
        #         self.pos_drones_sim.append(pos_di_sim)
        #     self.pos_drones_sim = torch.from_numpy(np.vstack(self.pos_drones_sim))

        # self._sim_update_states_drone(self.pos_drones_sim)
        self.vp_robot = VPDroneSimCV(client, self.obj_name, self.n_drones)

        if args.render:
            self.fig = plt.figure()
            self.ax = plt.subplot(111, projection='3d')

    def setup_drone(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        if self.mode == 'cv':
            self.vp_robot.take_off()

        elif self.mode == 'dynamic':
            for i_drone in range(self.n_drones):
                # Drone 1 and 2 Takeoff
                name = 'drone_%i' % (i_drone + 1)
                f1 = self.c.takeoffAsync(vehicle_name=name)
                f1 = self.c.moveToPositionAsync(0, 0, -10 - 1 * (i_drone + 1), 5, vehicle_name=name)
                f1.join()
        else:
            raise Exception("Wrong mode type.")

    def render(self, plt_ppas=False, plt_pause=True, plt_safe_reg=False):
        """ Plot the actor as a cuboid in matplotlib and drone position. """

        # draw coordinate system of camera
        self.ax.cla()
        draw_axis(self.ax, self.scale)

        for i_actor in range(self.num_actors):
            draw_cuboid(self.ax, self.Trans_can2w_list[i_actor], scale=self.scale)

        # pos_drones = spherical_to_cartesian(self.pos_drones)
        for i_drone in range(self.n_drones):
            pos_drone = self.pos_drones[i_drone, :].reshape(-1, 3).numpy()
            self.ax.scatter(xs=pos_drone[:, 0], ys=pos_drone[:, 1], zs=pos_drone[:, 2], marker='o', c='b')

        set_axes_full(self.ax, self.full_ws / 2)
        self.ax.set_xlabel('x axis')
        self.ax.set_ylabel('y axis')
        self.ax.set_zlabel('z axis')
        plt.grid()
        if plt_pause:
            plt.pause(0.1)
        return self.ax

    def update_states_actor(self, name_list):
        print("Update actor states --------------------")
        self.num_actors = len(name_list)
        self.pose_actors = []
        self.pos_actors = torch.zeros((self.num_actors, 3))
        self.ori_actors_quat = torch.zeros((self.num_actors, 4))
        self.Trans_can2w_list = torch.zeros((len(name_list), 4, 4))
        self.pos_patches = torch.zeros((self.num_actors, 5, 3))
        self.ori_patches = torch.zeros((self.num_actors, 5, 3))

        for i in range(len(name_list)):
            name = name_list[i]
            # -------------------- Obtain pose in world frame --------------------
            pose_actor_sim = self.c.simGetObjectPose(name)
            pos_actor_sim = pose_actor_sim.position.to_numpy_array()
            ori_actor_sim_quat = pose_actor_sim.orientation.to_numpy_array()

            r = R.from_quat(ori_actor_sim_quat)
            ori_actor_sim_matrix = r.as_matrix()

            rot_const = R.from_euler('x', 0, degrees=True).as_matrix()
            Trans = torch.hstack((torch.from_numpy(rot_const @ ori_actor_sim_matrix),
                                  torch.from_numpy(rot_const @ pos_actor_sim.reshape(3, -1))))
            Trans_can2w = torch.vstack((Trans, torch.tensor([0, 0, 0, 1])))

            self.pos_actors[i] = torch.from_numpy(pos_actor_sim)
            self.ori_actors_quat[i] = torch.from_numpy(ori_actor_sim_quat)
            self.Trans_can2w_list[i] = Trans_can2w

            self.pos_patches[i] = canonical2w(self.pos_patches_can, Trans_can2w)
            self.ori_patches[i] = (torch.from_numpy(rot_const @ ori_actor_sim_matrix).float() @ self.ori_patches_can.T).T

            print("Actor: ", name, "\tpos.\t", pos_actor_sim)
            print("Actor: ", name, "\tori.\t", ori_actor_sim_quat)

        self.n_patches = self.num_actors * 5
        self.pos_patches_flat = self.pos_patches.reshape(-1, 3)
        self.ori_patches_flat = self.ori_patches.reshape(-1, 3)

    def update_states_drone(self):
        if self.mode == 'cv':
            self.pos_drones, self.ori_drones = self.vp_robot.feedback_pose_abs()
        elif self.mode == 'dynamic':
            pos_drones_sim, ori_drones_sim = [], []
            for k in range(self.n_drones):
                pose_di_sim = self.c.getMultirotorState(vehicle_name='drone_' + str(k + 1))
                pos_di_sim = pose_di_sim.kinematics_estimated.position.to_numpy_array()
                ori_di_sim = pose_di_sim.kinematics_estimated.orientation.to_numpy_array()
                pos_drones_sim.append(pos_di_sim)
                ori_drones_sim.append(ori_di_sim)
            self.pos_drones_sim = torch.from_numpy(np.vstack(pos_drones_sim))
            self.ori_drones_sim = torch.from_numpy(np.vstack(ori_drones_sim))
            self.pos_drones = self.pos_drones_sim
            self.ori_drones = self.ori_drones_sim

    def plan_path_multiactors(self, pos_patches, ori_patches, pos_centers, pos_drones):
        """
        Plan path for multiple actors.
        :param pos_patches:  (n_actors * 5, 3)
        :param ori_patches:
        :param pos_centers:
        :param pos_drones:
        :return:
        path_length: number of waypoints
        path_full: position of drones actors
        vis_center_full: position of visual centers
        """
        print("Start Planning path for %i patches --------------------" % len(pos_patches))

        # Plot human positions accordingly.
        self.render(plt_ppas=False, plt_pause=True, plt_safe_reg=False)

        # Start planning for each drone.
        for k in range(self.n_drones):
            pos_di = pos_drones.reshape(-1, 3)[k].reshape(-1, 3)

            # TODO: plan path for each drone.

            # self.ax.plot(path_w_di[:, 0], path_w_di[:, 1], path_w_di[:, 2], 'b-')

        # Return path length (m, 1), path_full (k, m, 3), and orientation center (k, m, 3).
        return path_length, path_full, vis_center_full

    def execute_path_multiactors(self, path_length, path_full, vis_center_full, save_path, i):
        """
        Execute path for multiple actors.
        :param path_length: path length for each drone. (m, )
        :param path_full:
        :param vis_center_full:
        :param save_path:
        :param i:
        :return:
        """

        for j in range(np.max(path_length)):

            # Synchronize cameras by pausing the simulation.
            self.c.simPause(True)

            for k in range(self.n_drones):
                # Obtain the view point to go for k-th drone
                pos_2go_c_w_i = path_full[k, j].reshape(-1, 3).float().flatten().numpy()
                pos_vis_w_i = vis_center_full[k, j].reshape(-1, 3).float().flatten().numpy()
                print("================================================================================")
                print("Iterate %i, path step %i, drone idx %i" % (i, j, k))
                print("Planning viewing point ", pos_2go_c_w_i)
                print("Viewing center", pos_vis_w_i)

                # Automatically calculate orientation of the camera.
                camera_pose = self.vp_robot.get_ori_drone(pos_2go_c_w_i, pos_vis_w_i)
                self.vp_robot.move_to(camera_pose, self.args.time_sleep, k)

                # obtain images from camera.
                print("Save point cloud with path index %i, img index %i, drone index %i" % (i, j, k))
                _, cpose = self.vp_robot.capture_rgbd('/{}_' + '{:03d}_{:03d}_{:03d}'.format(i, j, k), k)

                # Save camera pose.
                camera_pose_save = cpose.flatten()
                np.save(save_path + '{}_'.format('/pose/pose') + '{:03d}_{:03d}_{:03d}.npy'.format(i, j, k),
                        camera_pose_save)

                # Visualize Messages sent in canonical views.
                self.ax.scatter(xs=cpose[0], ys=cpose[1], zs=cpose[2], marker='o', c='b')

            self.c.simPause(False)
            time.sleep(self.args.time_sleep)


    def view_planning_multiactor(self, num_iters=1, fname='sim_drones'):
        print("#################### START TRACKING ####################")

        # -------------------- Initial movement --------------------
        self.c.simPause(False)
        self.setup_drone()

        save_path = ROOT_DIR + "/archive/airsim/" + fname
        setup_dir(save_path)
        self.vp_robot.set_fname(save_path)

        # -------------------- Start main loop --------------------
        for i in range(num_iters):
            print("Start tracking multiple actors iteration %i --------------------" % i)

            # Pause from the beginning to obtain static mesh.
            self.c.simPause(True)

            # Update status for actors.
            self.update_states_actor(self.obj_name)
            np.save(save_path + '/T_{:03d}.npy'.format(i), self.Trans_can2w_list)
            np.save(save_path + '/apose_{:03d}.npy'.format(i),
                    np.hstack((self.pos_actors, self.ori_actors_quat)))

            self.update_states_drone()

            # Plan the path
            path_length, path_full, vis_center_full = self.plan_path_multiactors(pos_patches, ori_patches, centers, self.pos_drones)
            print("Iteration", i, "path length", path_length)

            # Now we are ready to execute the trajectory. -----------
            # Move the vehicle to each view points. -----------------
            self.execute_path_multiactors(path_length, path_full, vis_center_full, save_path, i)

            plt.savefig(save_path + "/path_{}_{:03d}.png".format('tspn', i))
            np.save(save_path + '/path_tspn_{:03d}.npy'.format(i), path_full)
            print("Finish tracking multiple actors iteration %i --------------------" % i)
