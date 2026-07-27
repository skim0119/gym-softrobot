from gymnasium import spaces
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class BasicMultiAgent(MultiAgentEnv):
    metadata = {
        "render_modes": ["rgb_array"],
    }

    def __init__(self, env, agent_ids, agent_state_index_list, shared_state_index_list):
        super().__init__()
        self.env = env
        self.agent_ids = agent_ids
        self.agent_state_index_list = agent_state_index_list
        self.shared_state_index_list = shared_state_index_list
        # self.agent_ids = ["joint0", "joint1"]
        # self.agent_state_index_list = [[0, 2], [1, 3]]
        # self.shared_state_index_list = [4, 5, 6, 7, 8, 9]
        self.num_agent = len(self.agent_ids)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.dones = set()
        self.resetted = False

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed, options=options)
        self.resetted = True
        self.dones = set()
        agent_obs = self.state_fn(obs)
        return agent_obs, {agent_id: {} for agent_id in self.agent_ids}

    def state_fn(self, obs):
        return {
            self.agent_ids[i]: np.concatenate([obs[self.agent_state_index_list[i]], obs[self.shared_state_index_list]],
                                              axis=0) for i in range(self.num_agent)}

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = self.env.step(
            list(action_dict.values())
        )
        agent_obs = self.state_fn(obs)
        agent_rew = {self.agent_ids[i]: rew for i in range(self.num_agent)}
        agent_terminated = {
            self.agent_ids[i]: terminated for i in range(self.num_agent)
        }
        agent_terminated["__all__"] = terminated
        agent_truncated = {
            self.agent_ids[i]: truncated for i in range(self.num_agent)
        }
        agent_truncated["__all__"] = truncated
        agent_info = {agent_id: info.copy() for agent_id in self.agent_ids}
        return (
            agent_obs,
            agent_rew,
            agent_terminated,
            agent_truncated,
            agent_info,
        )

    def render(self):
        # Just generate a random image here for demonstration purposes.
        # Also see `gym/envs/classic_control/cartpole.py` for
        # an example on how to use a Viewer object.
        return np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)
