__all__ = ["ConvertToPyMarlEnv"]

import typing
from typing import List, Iterable, Union, Optional
import numpy as np

import gym


class ConvertToPyMarlEnv:
    """
    Template from: oxwhirl/pymar/src/envs/multiagentenv.py

    The wrapper to convert gym_softrobot environment (that is MA-compatible)
    to 'multiagentenv' that is compatible to PyMARL.
    The purpose is to run benchmark study with standard CTDE algorithms, such
    as QMIX, COMA, etc.

        env = ConvertToPyMarlEnv(ma_env=gym.make("OctoCrawl-v0"))
    """

    def __init__(self, ma_env: gym.Env, *args, **kwargs):
        """

        Parameters
        ----------
        ma_env : gym.Env
            Multi-agent compatible environment
            In gym-softrobot, environment with 'multiagent' tag in meta data
            must be true.
        """
        super().__init__(*args, **kwargs)

        # Register environment
        assert (
            "PyMARL" in ma_env.metadata["multiagent"]
        ), f"The given environment is not multi-agent compatible"
        self.env = ma_env

    @property
    def n_agents(self):
        return self.env.n_agent

    def step(self, actions):
        """Returns reward, terminated, info"""
        self.observation, reward, done, info = self.env.step(actions.ravel())
        return reward, done, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        obs_size = self.get_obs_size()
        return self.observation[agent_id * obs_size : (agent_id + 1) * obs_size]

    def get_obs_size(self):
        """Returns the shape of the observation

        The observation in regular gym_softrobot is given as (n * state_space)
        """
        obs_size = self.env.observation_space.shape[0] // self.n_agents
        return obs_size

    def get_state(self):
        """return global state.

        Notes
        -----
        Ideally, this function should not be used during decentralized execution.
        """
        return self.env.get_shared_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.shared_space

    def get_avail_actions(self) -> List:
        """Returns the available actions of all agents in a list."""
        return [
            self.get_aviala_agent_actions(agent_id) for agent_id in range(self.n_agents)
        ]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        raise NotImplementedError

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def save_replay(self):
        # TODO: Low priority
        raise NotImplementedError

    # Implemented in original environment
    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Returns initial observations and states"""
        self.observation = self.env.reset(**kwargs)
        return self.observation, self.get_state()
