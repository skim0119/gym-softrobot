import os, sys
import numpy as np
import gym

# import pybulletgym  # register PyBullet enviroments with open ai gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":

    env = SoftArmTracking()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/tensorboard/",
    )
    model.learn(
        total_timesteps=1000000,
    )
    # model.save('POLICY', )

    # model = PPO.load('POLICY', env = env)
    # obs = env.reset()
    # for _ in range(500):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
