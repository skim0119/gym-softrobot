import sys
sys.path.append("..")
import gym
import gym_softrobot

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

# Parallel environments
env = make_vec_env("ContinuumSnake-v0", n_envs=4)

runid = 1
algo = f'PPO'
model_save_path = f"model/{algo}/run_{runid}"
tensorboard_log = f"./logs/{algo}/"
tb_log_name = f"{algo}_run_{runid}"

model = PPO("MlpPolicy", env, n_steps=128, verbose=2, use_sde=True, tensorboard_log=tensorboard_log)

checkpoint_callback = CheckpointCallback(save_freq=20, save_path=model_save_path,
		            name_prefix='rl_model', verbose=2)
model.learn(total_timesteps=1000000, tb_log_name=tb_log_name, callback=checkpoint_callback)
model.save("ppo_snake")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_snake")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
