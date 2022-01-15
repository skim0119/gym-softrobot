import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from tqdm import tqdm

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer


class DecPPO(PPO):
    def __init__(self, n_agent=5, progress_bar=True, **kwargs):
        self.n_agent = n_agent
        self.progress_bar = progress_bar
        super(DecPPO, self).__init__(**kwargs)

    def _setup_model(self) -> None:
        super(DecPPO, self)._setup_model()
        """ Customize model to support multiple arm """
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs*self.n_agent,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        env_num_envs = env.num_envs * self.n_agent

        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env_num_envs)

        callback.on_rollout_start()

        # customize: last_episode_starts
        self._last_episode_starts = np.ones(env_num_envs, dtype=bool)

        if self.progress_bar:
            pbar = tqdm(total=n_rollout_steps)
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env_num_envs)

            # Reshape spaces
            obs = {}
            for key, space in self.observation_space.spaces.items():
                if key == 'shared':
                    val = np.repeat(self._last_obs[key], self.n_agent, axis=0)
                else:
                    val = self._last_obs[key]
                obs[key] = np.reshape(val, [env_num_envs]+list(space.shape))

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Reshape actions
            reshaped_actions = np.reshape(
                    clipped_actions,
                    [env.num_envs,self.n_agent]+list(self.action_space.shape))

            try:
                new_obs, reward, done, infos = env.step(reshaped_actions)
            except EOFError:
                print(reshaped_actions)
                raise EOFError("Caught")

            # Reshape reward and dones
            rewards = np.repeat(reward, self.n_agent)
            dones = np.repeat(done, self.n_agent)

            # Reshape infos

            self.num_timesteps += env.num_envs#env_num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1
            if self.progress_bar:
                pbar.update(1)

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            '''
            for idx, done in enumerate(dones):
                info_idx = idx // self.n_agent
                if (
                    done
                    and infos[info_idx].get("terminal_observation") is not None
                    and infos[info_idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[info_idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            '''
            rollout_buffer.add(obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Reshape spaces
        obs = {}
        for key, space in self.observation_space.spaces.items():
            if key == 'shared':
                val = np.repeat(self._last_obs[key], self.n_agent, axis=0)
            else:
                val = self._last_obs[key]
            obs[key] = np.reshape(val, [env_num_envs]+list(space.shape))

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        if self.progress_bar:
            pbar.close()

        return True
