"""Continuum-snake locomotion as an ``ElasticaTask``.

``ContinuumSnake-v0`` is unchanged. This task drives the same rod, muscle
torque, and anisotropic plane through ``RodBatch``. The build does not read
the generator, because ``ContinuumSnakeEnv.reset`` does not either.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gym_softrobot.envs.snake.continuum_snake import (
    ContinuumSnakeEnv,
    compute_projected_velocity,
)
from gym_softrobot.sim import ElasticaTask, ElasticaWorld, TaskSpec

# Same clock as ContinuumSnakeEnv._build. One policy step is step_skip
# integrator steps, and the episode ends on the first step at or past final_time.
_PERIOD = 2.0
_FINAL_TIME = (11.0 + 0.01) * _PERIOD
_INTEGRATOR_DT = 8e-6
_STEP_SKIP = int(1.0 / (5 * _INTEGRATOR_DT))
_CONTROL_DT = _STEP_SKIP * _INTEGRATOR_DT
_HORIZON = math.ceil(_FINAL_TIME / _CONTROL_DT)


class _BoundClock:
    """Adapt ``ContinuumSnakeEnv``'s bound stepper to ``ElasticaBatch``.

    The env stepper already closes over the simulator and the integrator dt.
    """

    def __init__(self, advance) -> None:
        self._advance = advance

    def step(self, simulator: object, time: float, dt: float) -> float:
        del simulator, dt
        return self._advance(time=time)


def snake_task_spec() -> TaskSpec:
    probe = ContinuumSnakeEnv()
    return TaskSpec(
        action_space=probe.action_space,
        observation_space=probe.observation_space,
        control_dt=_CONTROL_DT,
        horizon=_HORIZON,
        rod_nodes=(probe._n_elem + 1,),
    )


@dataclass(eq=False)
class SnakeWorld(ElasticaWorld):
    env: ContinuumSnakeEnv | None = None


class SnakeLocomotionTask(ElasticaTask[SnakeWorld]):
    spec: TaskSpec = snake_task_spec()

    def build(self, rng: np.random.Generator) -> SnakeWorld:
        del rng
        env = ContinuumSnakeEnv()
        env.reset()
        return SnakeWorld(
            simulator=env.snake_sim,
            timestepper=_BoundClock(env.stepper),
            dt=_INTEGRATOR_DT,
            rods=(env.shearable_rod,),
            env=env,
        )

    def apply_action(self, world: SnakeWorld, action: NDArray[np.float32]) -> None:
        if world.env is None:
            raise RuntimeError("snake world has no environment")
        world.env.set_action(np.asarray(action, dtype=np.float32))

    def observe(self, world: SnakeWorld) -> NDArray[np.float32]:
        if world.env is None:
            raise RuntimeError("snake world has no environment")
        return world.env.get_state()

    def reward(self, world: SnakeWorld) -> float:
        if world.env is None:
            raise RuntimeError("snake world has no environment")
        forward = compute_projected_velocity(world.env.data, world.env.period)[2]
        return float(forward)
