import pickle

import pytest

from gym import envs
import gym_softrobot

from tests.envs.spec_list import spec_list 

@pytest.mark.parametrize("spec", spec_list)
def test_serialize_deserialize(spec):
    env = spec.make()
    env.reset()

    with pytest.raises(AssertionError, match="invalid"):
        env.step([0.1])

    with pytest.raises(AssertionError, match="invalid"):
        env.step(0.1)
