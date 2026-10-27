from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


def _require_vector_box(space: spaces.Space, name: str) -> None:
    if (
        not isinstance(space, spaces.Box)
        or space.dtype != np.float32
        or len(space.shape) != 1
    ):
        raise ValueError(f"{name} must be a 1-d float32 Box, got {space!r}")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Spaces, control period, horizon, and rod node counts."""

    action_space: spaces.Box
    observation_space: spaces.Box
    control_dt: float
    horizon: int
    rod_nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_vector_box(self.action_space, "action_space")
        _require_vector_box(self.observation_space, "observation_space")
        if (
            isinstance(self.control_dt, bool)
            or not isinstance(self.control_dt, (int, float))
            or self.control_dt <= 0
        ):
            raise ValueError(f"control_dt must be positive, got {self.control_dt!r}")
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon < 1
        ):
            raise ValueError(f"horizon must be an integer >= 1, got {self.horizon!r}")
        for count in self.rod_nodes:
            if isinstance(count, bool) or not isinstance(count, int) or count < 2:
                raise ValueError(
                    f"each rod needs at least 2 nodes, got {self.rod_nodes!r}"
                )

    @property
    def episode_length_s(self) -> float:
        return float(self.horizon * self.control_dt)


@dataclass(frozen=True, slots=True)
class BatchStep:
    obs: NDArray[np.float32]
    reward: NDArray[np.float32]
    failed: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BatchFrame:
    """Per-rod centerlines `(N, 3, K)` and element radii `(N, K - 1)`."""

    centerlines: tuple[NDArray[np.float64], ...]
    radii: tuple[NDArray[np.float64], ...]

    def __post_init__(self) -> None:
        if len(self.centerlines) != len(self.radii):
            raise ValueError(
                "centerlines and radii must have one array per rod, "
                f"got {len(self.centerlines)} and {len(self.radii)}"
            )
        for index, (centerline, radius) in enumerate(
            zip(self.centerlines, self.radii, strict=True)
        ):
            shape = np.shape(centerline)
            if len(shape) != 3 or shape[1] != 3:
                raise ValueError(
                    f"rod {index} centerline must have shape (N, 3, K), got {shape}"
                )
            expected = (shape[0], shape[2] - 1)
            if np.shape(radius) != expected:
                raise ValueError(
                    f"rod {index} radii must have shape {expected}, got {np.shape(radius)}"
                )


@dataclass(frozen=True, slots=True)
class ProcessPool:
    """Step worlds in worker processes. ``None`` workers means the CPU count."""

    workers: int | None = None

    def __post_init__(self) -> None:
        if self.workers is not None and (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise ValueError(
                f"workers must be None or an integer >= 1, got {self.workers!r}"
            )


def require_actions(
    actions: NDArray[np.float32], num_envs: int, spec: TaskSpec
) -> NDArray[np.float32]:
    actions = np.asarray(actions, dtype=np.float32)
    expected = (num_envs, spec.action_space.shape[0])
    if actions.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {actions.shape}")
    return actions


def require_generators(
    ids: NDArray[np.intp], rngs: Sequence[np.random.Generator]
) -> None:
    if len(rngs) != ids.size:
        raise ValueError(f"expected {ids.size} generators, got {len(rngs)}")


class RodBatch(Protocol):
    """N independent worlds advanced in lockstep."""

    @property
    def spec(self) -> TaskSpec: ...

    @property
    def num_envs(self) -> int: ...

    def reset(
        self, env_ids: NDArray[np.intp], rngs: Sequence[np.random.Generator]
    ) -> NDArray[np.float32]: ...

    def step(self, actions: NDArray[np.float32]) -> BatchStep: ...

    def frame(self) -> BatchFrame: ...

    def close(self) -> None: ...


class SoftRobotTask(Protocol):
    spec: TaskSpec

    def make_batch(
        self, num_envs: int, processes: ProcessPool | None = None
    ) -> RodBatch: ...
