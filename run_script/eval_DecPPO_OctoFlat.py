import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import numpy as np
from numba import njit
from tqdm import tqdm
import time
import types

from functools import partial
import sys
sys.path.append("../../../../") # include elastica-python directory
sys.path.append("../../")       # include ActuationModel directory

from set_environment import Environment

from custom_ppo import CustomPPO

from callback_func import OthersCallBack

import gym

from stable_baselines3.common.vec_env import SubprocVecEnv #DummyVecEnv
#from stable_baselines3 import PPO # ,A2C,DDPG,SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

if __name__ == "__main__":

    """ Create simulation environment
    Total number of simulataneous data-collection is n_body * n_envs
    """
    final_time = 20.0
    n_elems = 9
    n_body = 1
    n_arm = 5
    fps = 5

    mode = "decentralized"

    # Set policy
    if mode == "centralized":
        from custom_ppo import CustomPPO as module
        policy = "MultiInputPolicy"
    elif mode == "decentralized": # Fully Decentralized
        from custom_decppo import CustomDecPPO as module
        policy = "MultiInputPolicy"
    elif mode == "DTCE":
        raise NotImplementedError
    else:
        raise NotImplementedError

    env_kwargs = {
            'final_time': final_time,
            'time_step': 4e-5, #8e-6,
            'recording_fps': fps,
            'n_elems': n_elems,
            'n_body': n_body,
            'policy_mode': mode,
        }
    env = Environment(**env_kwargs, config_generate_video=True)
    state = env.reset()


    """ Read arm params """
    #step_skip = env.step_skip
    #others_parameters_dict = defaultdict(list)
    #others_callback = OthersCallBack(step_skip, others_parameters_dict)

    # Load
    print("----- Loading -----")
    model_path = "model/PPO_decentralized/run_11/rl_model_572800_steps.zip"
    model = module.load(model_path)

    total_steps = int(final_time * fps)#750 # 751
    for k_sim in tqdm(range(total_steps)):
        if mode == "decentralized":
            # Reshape spaces
            obs = {}
            for key, space in env.observation_space.spaces.items():
                if key == 'shared':
                    val = np.repeat(state[key], n_arm, axis=0)
                    obs[key] = np.reshape(val, [n_arm]+list(space.shape))
                else:
                    val = state[key]
                    obs[key] = np.reshape(val, [n_arm]+list(space.shape))
        elif mode == "centralized": # TODO
            # Reshape spaces
            obs = {}
            for key, space in env.observation_space.spaces.items():
                if key == 'shared':
                    val = np.repeat(state[key], n_arm, axis=0)
                    obs[key] = np.reshape(val, [n_arm]+list(space.shape))
                else:
                    val = state[key]
                    obs[key] = np.reshape(val, space.shape)
            #obs = np.reshape(state, [n_body]+list(env.observation_space.shape))
        action_kappa = model.predict(obs)[0]
        # Action reshape
        action_kappa = np.reshape(action_kappa, [n_body,n_arm,-1])
        action_kappa = np.clip(action_kappa, env.action_space.low, env.action_space.high)
        state, reward, done, info = env.step(action_kappa)
        print(f'action stat: mean={action_kappa.mean()}, std={action_kappa.std()}, max={action_kappa.max()}, min={action_kappa.min()}, absmin={np.abs(action_kappa).min()}')
        #print(info['time'])
        if done:
            break

    """ Save the data of the simulation """
    path = f'PPO_{mode}_1.mp4' # Name
    env.save_data(path, fps)

