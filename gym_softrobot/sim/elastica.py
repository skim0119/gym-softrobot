from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import numpy as np
from elastica import BaseSystemCollection
from elastica.rod import RodBase
from numpy.typing import NDArray

from gym_softrobot.sim._process_batch import ProcessBatch
from gym_softrobot.sim.batch import (
    BatchFrame,
    BatchStep,
    ProcessPool,
    RodBatch,
    TaskSpec,
    require_actions,
    require_generators,
)


@dataclass(eq=False)
class ElasticaWorld:
    """One finalized PyElastica simulation. Task subclasses add their own fields."""

    simulator: BaseSystemCollection
    timestepper: object
    dt: float
    rods: tuple[RodBase, ...]
    time: float = 0.0


W = TypeVar("W", bound=ElasticaWorld)


class ElasticaTask(ABC, Generic[W]):
    spec: ClassVar[TaskSpec]

    @abstractmethod
    def build(self, rng: np.random.Generator) -> W:
        """Build and finalize a fresh world. All randomness comes from ``rng``."""

    @abstractmethod
    def apply_action(self, world: W, action: NDArray[np.float32]) -> None: ...

    @abstractmethod
    def observe(self, world: W) -> NDArray[np.float32]: ...

    @abstractmethod
    def reward(self, world: W) -> float: ...

    def failed(self, world: W) -> bool:
        return False

    def make_batch(
        self, num_envs: int, processes: ProcessPool | None = None
    ) -> RodBatch:
        if processes is None:
            return ElasticaBatch(self, num_envs)
        return ProcessBatch(self, num_envs, processes)


def _substep_count(control_dt: float, dt: float) -> int:
    if dt <= 0:
        raise ValueError(f"world dt must be positive, got {dt}")
    ratio = control_dt / dt
    nearest = int(round(ratio))
    if nearest < 1 or abs(ratio - nearest) > 1e-9 * abs(ratio):
        raise ValueError(
            f"control_dt / dt = {ratio} is not an integer within 1e-9 relative"
        )
    return nearest


def _diverged(world: ElasticaWorld) -> bool:
    return any(
        not all(
            np.isfinite(np.asarray(state)).all()
            for state in (
                rod.position_collection,
                rod.velocity_collection,
                rod.director_collection,
            )
        )
        for rod in world.rods
    )


class ElasticaBatch(Generic[W]):
    def __init__(self, task: ElasticaTask[W], num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._task = task
        self._num_envs = num_envs
        self._worlds: list[W | None] = [None] * num_envs

    @property
    def spec(self) -> TaskSpec:
        return self._task.spec

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(
        self, env_ids: NDArray[np.intp], rngs: Sequence[np.random.Generator]
    ) -> NDArray[np.float32]:
        ids = np.asarray(env_ids, dtype=np.intp).reshape(-1)
        require_generators(ids, rngs)
        obs_dim = int(self.spec.observation_space.shape[0])
        obs = np.empty((ids.size, obs_dim), dtype=np.float32)
        for row, (index, rng) in enumerate(zip(ids, rngs, strict=True)):
            world = self._task.build(rng)
            self._check_world(world)
            self._worlds[int(index)] = world
            observed = np.asarray(self._task.observe(world), dtype=np.float32)
            if observed.shape != (obs_dim,):
                raise ValueError(
                    f"observe returned shape {observed.shape}, expected {(obs_dim,)}"
                )
            obs[row] = observed
        return obs

    def step(self, actions: NDArray[np.float32]) -> BatchStep:
        self._require_ready()
        actions = require_actions(actions, self._num_envs, self.spec)
        obs_dim = int(self.spec.observation_space.shape[0])
        obs = np.empty((self._num_envs, obs_dim), dtype=np.float32)
        reward = np.empty(self._num_envs, dtype=np.float32)
        failed = np.empty(self._num_envs, dtype=np.bool_)
        for index, world in enumerate(self._worlds):
            if world is None:
                raise RuntimeError(f"world {index} was never reset")
            self._task.apply_action(world, actions[index])
            substeps = _substep_count(self.spec.control_dt, world.dt)
            for _ in range(substeps):
                world.time = world.timestepper.step(
                    world.simulator, world.time, world.dt
                )
            numerical_failure = _diverged(world)
            failed[index] = numerical_failure or bool(self._task.failed(world))
            if numerical_failure:
                obs[index] = 0.0
                reward[index] = 0.0
                continue
            observed = np.asarray(self._task.observe(world), dtype=np.float32)
            if observed.shape != (obs_dim,):
                raise ValueError(
                    f"observe returned shape {observed.shape}, expected {(obs_dim,)}"
                )
            reward_value = np.float32(self._task.reward(world))
            if not np.isfinite(observed).all() or not np.isfinite(reward_value):
                failed[index] = True
                obs[index] = np.nan_to_num(
                    observed, nan=0.0, posinf=0.0, neginf=0.0
                )
                reward[index] = 0.0
            else:
                obs[index] = observed
                reward[index] = reward_value
        return BatchStep(obs=obs, reward=reward, failed=failed)

    def frame(self) -> BatchFrame:
        self._require_ready()
        centerlines = []
        radii = []
        for rod_index in range(len(self.spec.rod_nodes)):
            centerlines.append(
                np.stack(
                    [
                        np.asarray(
                            world.rods[rod_index].position_collection, dtype=np.float64
                        )
                        for world in self._worlds
                        if world is not None
                    ]
                )
            )
            radii.append(
                np.stack(
                    [
                        np.asarray(world.rods[rod_index].radius, dtype=np.float64)
                        for world in self._worlds
                        if world is not None
                    ]
                )
            )
        return BatchFrame(centerlines=tuple(centerlines), radii=tuple(radii))

    def close(self) -> None:
        self._worlds = [None] * self._num_envs

    def _require_ready(self) -> None:
        missing = [index for index, world in enumerate(self._worlds) if world is None]
        if missing:
            raise RuntimeError(f"worlds were never reset: {missing}")

    def _check_world(self, world: ElasticaWorld) -> None:
        expected = self.spec.rod_nodes
        if len(world.rods) != len(expected):
            raise ValueError(f"expected {len(expected)} rods, got {len(world.rods)}")
        for index, (rod, n_nodes) in enumerate(zip(world.rods, expected, strict=True)):
            position = np.asarray(rod.position_collection)
            if position.shape != (3, n_nodes):
                raise ValueError(
                    f"rod {index} position shape {position.shape}, "
                    f"expected {(3, n_nodes)}"
                )
        _substep_count(self.spec.control_dt, world.dt)
