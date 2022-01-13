import pytest
import numpy as np

from gym import envs
from gym.spaces import Box
from gym.utils.env_checker import check_env

from tests.envs.spec_list import spec_list


"""
@pytest.mark.parametrize("spec", spec_list)
def test_env_render_result_is_immutable(spec):
    env = spec.make()
    env.reset()
    output = env.render(mode="ansi")
    assert isinstance(output, str)
    env.close()
"""
