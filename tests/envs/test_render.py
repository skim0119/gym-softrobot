import pytest
import numpy as np

from gym import envs
from gym.spaces import Box
from gym.utils.env_checker import check_env

from tests.envs.spec_list import spec_list

from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.config import RendererType

#RENDERER_CONFIG = RendererType.POVRAY
RENDERER_CONFIG = RendererType.MATPLOTLIB

# FIXME: Testing rendering is disabled for now. It needs to install POVRAY on CI
#@pytest.mark.parametrize("spec", spec_list)
#def test_env_render_result_np_array_for_rgb_mode(spec):
#    env = spec.make()
#    env.reset()
#    output = env.render(mode='rgb_array')
#    assert isinstance(output, np.ndarray)
#    assert output.shape[2] == 3
#    env.close()
