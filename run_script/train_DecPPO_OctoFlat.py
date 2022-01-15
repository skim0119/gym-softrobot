import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import numpy as np
import time
import types

from functools import partial
import sys
sys.path.append("..")

import gym
import gym_softrobot

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

if __name__ == "__main__":
    """ Create simulation environment
    Total number of simulataneous data-collection is n_envs
    """
    runid = 4  # TAG: Repeated run will append another id

    final_time = 10.0
    fps = 4
    n_elems = 9
    n_arm = 8
    n_envs = 12
    mode = "decentralized"

    # Set policy
    if mode == "centralized":
        from stable_baselines3 import PPO as module #A2C,DDPG,SAC
        policy = "MultiInputPolicy"
    elif mode == "decentralized": # Fully Decentralized
        from marl.dec_ppo import DecPPO as module
        policy = "MultiInputPolicy"
    elif mode == "DTCE":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Number of steps to run for each environment per update 
    # The total rollout buffer size will be n_steps * n_envs (* n_arm if decentralized)
    n_steps = 40
    loop_freq = n_steps * n_envs

    #env = Environment(final_time, time_step = 8e-6,recording_fps=30,n_elems=n_elems)
    env_kwargs = {
            'final_time': final_time,
            'time_step': 4e-5, #8e-6,
            'recording_fps': fps,
            'n_elems': n_elems,
            'n_arm': n_arm,
            'policy_mode': mode,
        }
    env = make_vec_env('OctoFlat-v0', n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
    state = env.reset()

    """ Save Configuration """
    algo = f'PPO_{mode}' # Name
    tensorboard_log = f"./logs/{algo}/"
    tb_log_name = f"{algo}_run_{runid}"
    model_save_path = f"model/{algo}/run_{runid}"

    """ Start the simulation """
    print("Running simulation ...")
    LOAD = False
    LOAD_PATH = "PPO_e2e_5arm/run_7/rl_model_896000_steps.zip"
    model_kwargs = dict(
                env=env, 
                use_sde=True,
                vf_coef=0.01,
                ent_coef=0.0,
                verbose=2, 
                tensorboard_log=tensorboard_log,
                n_steps=n_steps,
                learning_rate=4e-4,
                n_agent=n_arm,
                progress_bar=True,
            )
    policy_kwargs = {
            }
    if not LOAD:
        model = module(
                policy=policy, 
                **model_kwargs,
            )
    else:
        model = module.load(
                LOAD_PATH,
                print_system_info=True,
                **model_kwargs,
            )

    """ Trains """
    print("----- Training -----")
    checkpoint_callback = CheckpointCallback(save_freq=40, save_path=model_save_path,
            name_prefix='rl_model', verbose=2)
    model.learn(
            total_timesteps=10000000,
            tb_log_name=tb_log_name,
            #reset_num_timesteps=False,
            callback=checkpoint_callback
        )
    model.save(f"{model_save_path}_final")
