import math
from types import SimpleNamespace

import numpy as np
import pytest

from gym_softrobot.envs.snake.continuum_snake import ContinuumSnakeEnv
from gym_softrobot.envs.snake.locomotion_task import SnakeLocomotionTask, SnakeWorld

# ContinuumSnakeEnv._build uses these values. The task must truncate on the
# same policy step that env.time first reaches final_time.
_PERIOD = 2.0
_FINAL_TIME = (11.0 + 0.01) * _PERIOD
_INTEGRATOR_DT = 8e-6
_STEP_SKIP = int(1.0 / (5 * _INTEGRATOR_DT))


def test_snake_task_spec_matches_the_gym_env() -> None:
    env = ContinuumSnakeEnv()
    spec = SnakeLocomotionTask.spec
    assert spec.action_space == env.action_space
    assert spec.observation_space == env.observation_space
    assert spec.action_space.dtype == np.float32
    assert spec.observation_space.dtype == np.float32
    assert spec.rod_nodes == (51,)
    assert spec.control_dt == _STEP_SKIP * _INTEGRATOR_DT
    assert spec.horizon == math.ceil(_FINAL_TIME / spec.control_dt)
    assert spec.episode_length_s == spec.horizon * spec.control_dt


def test_reward_is_forward_speed_over_this_step() -> None:
    rod = SimpleNamespace(
        compute_position_center_of_mass=lambda: np.array([0.0, 0.0, 0.25])
    )
    world = SnakeWorld(
        simulator=None,
        timestepper=None,
        dt=1.0,
        rods=(rod,),
        com_z=0.05,
    )
    speed = SnakeLocomotionTask().reward(world)
    assert speed == pytest.approx((0.25 - 0.05) / SnakeLocomotionTask.spec.control_dt)
    assert world.com_z == pytest.approx(0.25)
