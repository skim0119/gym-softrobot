__all__ = ["ConvertToPyMarlEnv"]

import typing
import numpy as np

import gym

class ConvertToPyMarlEnv:
    """ 
    Template from: oxwhirl/pymar/src/envs/multiagentenv.py

    The wrapper to convert gym_softrobot environment (that is MA-compatible)
    to 'multiagentenv' that is compatible to PyMARL.
    The purpose is to run benchmark study with standard CTDE algorithms, such
    as QMIX, COMA, etc.

    Notes
    _____
    Since 
    """

    def __init__(self, ma_env:gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.COMPATIBLE_PYMARL

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    # Implemented in child
    #def close(self):
    #    raise NotImplementedError

    #def seed(self):
    #    raise NotImplementedError

    #def render(self):
    #    raise NotImplementedError

    #def reset(self):
    #    """ Returns initial observations and states"""
    #    raise NotImplementedError
