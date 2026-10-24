from typing import Any

import numpy as np
from gymnasium import Env
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from gym_softrobot.sim.batch import BatchFrame, SoftRobotTask


class RodTaskEnv(Env[NDArray[np.float32], NDArray[np.float32]]):
    """Single-world Gymnasium env over ``SoftRobotTask.make_batch(1)``."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, task: SoftRobotTask, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self._batch = task.make_batch(1)
        self.action_space = task.spec.action_space
        self.observation_space = task.spec.observation_space
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._batch.reset(np.array([0], dtype=np.intp), [self.np_random])
        self._step_count = 0
        return np.array(obs[0], dtype=np.float32, copy=True), {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        err_msg = f"{action!r} ({type(action)}) invalid: expected {self.action_space}"
        assert self.action_space.contains(action), err_msg

        batched = np.asarray(action, dtype=np.float32).reshape(1, -1)
        result = self._batch.step(batched)
        self._step_count += 1
        observation = np.array(result.obs[0], dtype=np.float32, copy=True)
        reward = float(result.reward[0])
        terminated = bool(result.failed[0])
        truncated = self._step_count >= self._batch.spec.horizon
        return observation, reward, terminated, truncated, {}

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        return _draw_top_down(self._batch.frame())

    def close(self) -> None:
        self._batch.close()


def _draw_top_down(frame: BatchFrame) -> NDArray[np.uint8]:
    figure = Figure(figsize=(4, 4), dpi=80)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_aspect("equal")
    for centerlines, radii in zip(frame.centerlines, frame.radii, strict=True):
        for env_index in range(centerlines.shape[0]):
            center = centerlines[env_index]
            marker = max(2.0, 80.0 * float(np.mean(radii[env_index])))
            axis.plot(center[0], center[2], marker="o", markersize=marker, linewidth=1)
    axis.axis("off")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[:, :, :3].copy()
