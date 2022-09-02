import os
import sys

import airsim as sim

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from src.args import VPArgs
from src.envs.multiactor_sim_env import ViewPlanningMultiActor
from src.reconstruction.exp_reconstruction_dynamic import exp_dynamic_reconstruction_multiple_drones
from src.visual.vis_reconstruction_dynamic import vis_dynamic_reconstruction_multiple_drones


if __name__ == '__main__':

    opts = VPArgs()
    args = opts.get_args()

    args.obj_name = ['person_actor_%i' % (i+1) for i in range(args.n_actors)]

    c = sim.MultirotorClient()
    c.confirmConnection()

    env = ViewPlanningMultiActor(args, client=c, mode='cv')
    print("Finish environment initialization. ")

    args.fname = 'exp_multiactor'   # Folder name.
    env.view_planning_multiactor(num_iters=1, fname=args.fname)
