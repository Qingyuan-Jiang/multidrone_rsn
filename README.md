# multidrone_rsn
 
## Installation
Install in the virtual environments, and the API for the AirSim with `pip`.

```pip install airsim```

## Codes structure
Environments are placed in the `envs` folder. The `envs` folder contains the following files:
- `vp_drone.py' : wrapper of APIs for the drone
- `multiactor_sim_env.py': the environment for the simulation

To use the environment, you may want to call scripts from `simulation` folder.
- 'sim_cv_multiactor.py': call the environment in CV model.

Experiments are placed in the `experiments` folder. Reconstructed actor is compared with ground truth in this part.
- 'exp_chamfer_dynamic.py': compare reconstruction results by chamfer distance.
- 'exp_chamfer_multiactor.py': multiactors are placed based on their pose.

