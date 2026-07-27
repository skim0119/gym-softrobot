import pytest
import numpy as np

from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

from tests.envs.spec_list import spec_list

from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.config import RendererType

RENDERER_CONFIG = RendererType.MATPLOTLIB


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.parametrize("spec", spec_list)
def test_env(spec):
    env = spec.make()

    # Test if env adheres to the Gymnasium API.
    check_env(env.unwrapped, skip_render_check=True)

    ob_space = env.observation_space
    act_space = env.action_space
    ob, info = env.reset()
    assert isinstance(info, dict)
    assert ob_space.contains(ob), f"Reset observation: {ob!r} not in space"
    if isinstance(ob_space, Box):
        # Only checking dtypes for Box spaces to avoid iterating through tuple entries
        assert (
            ob.dtype == ob_space.dtype
        ), f"Reset observation dtype: {ob.dtype}, expected: {ob_space.dtype}"

    a = act_space.sample()
    observation, reward, terminated, truncated, _info = env.step(a)
    assert ob_space.contains(
        observation
    ), f"Step observation: {observation!r} not in space"
    assert np.isscalar(reward), f"{reward} is not a scalar for {env}"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    if isinstance(ob_space, Box):
        assert (
            observation.dtype == ob_space.dtype
        ), f"Step observation dtype: {ob.dtype}, expected: {ob_space.dtype}"

    # FIXME: Test rendering need to install povray on CI. It is disabled for now.
    #for mode in env.metadata.get("render.modes", []):
    #    env.render(mode=mode)

    # Make sure we can render the environment after close.
    #for mode in env.metadata.get("render.modes", []):
    #    env.render(mode=mode)

    env.close()

@pytest.mark.parametrize("spec", spec_list)
def test_reset_info(spec):

    env = spec.make()

    ob_space = env.observation_space
    obs, info = env.reset()
    assert ob_space.contains(obs)
    assert isinstance(info, dict)
    env.close()

# FIXME: Current version of SB3 1.4.0 uses older gym version, which is causing some issue with 
# newer gym version 0.23.1. This block is commented out until those conflicts are solved.
#@pytest.mark.parametrize("spec", spec_list)
#def test_sb3_check_env(spec):
#    from stable_baselines3.common.env_checker import check_env
#    # Run SB3's check_env function for compatiblity check
#    with pytest.warns(None) as warnings:
#        env = spec.make()
#    # It will check your custom environment and output additional warnings if needed
#    check_env(env)
