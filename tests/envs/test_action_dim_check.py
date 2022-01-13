import pickle

import pytest

from gym import envs
import gym_softrobot

ENVIRONMENT_IDS = ("OctoFlat-v0",)

@pytest.mark.parametrize("environment_id", env_list)
def test_serialize_deserialize(environment_id):
    env = envs.make(environment_id)
    env.reset()

    with pytest.raises(AssertionError, match="invalid"):
        env.step([0.1])

    with pytest.raises(AssertionError, match="invalid"):
        env.step(0.1)
