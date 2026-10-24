import math

import numpy as np

from gym_softrobot.envs.snake.continuum_snake import ContinuumSnakeEnv
from gym_softrobot.envs.snake.locomotion_task import SnakeLocomotionTask

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
