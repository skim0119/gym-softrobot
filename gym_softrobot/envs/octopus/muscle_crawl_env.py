"""Feedback-controlled Octopus-muscle crawler.

The muscle-group simulator is deliberately exposed as a regular Gymnasium
environment here.  Each action controls one interval of muscle actuation;
the policy parameters used by the old bandit interface remain in
the internal crawling simulation modules for the later whole-rollout wrapper.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_softrobot.envs.octopus.crawling_simulation.config import OctopusMuscleConfig
from gym_softrobot.envs.octopus.crawling_simulation.muscle_simulation import (
    PhaseOctopusMuscleSimulation,
)


class OctoMuscleCrawlEnv(gym.Env[np.ndarray, np.ndarray]):
    """Eight-arm Octopus crawling with explicit TM/LM/OM muscle feedback.

    The action is one normalized command per muscle group and arm, in the
    order ``tm, lm0..lm3, base_suction, middle_suction, om_positive,
    om_negative``.  One :meth:`step` advances the physical system by one
    ``control_dt`` interval, so the environment is suitable for RL policies.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        config: OctopusMuscleConfig | None = None,
        horizon: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.config = config or OctopusMuscleConfig()
        self.horizon = horizon or max(
            1, math.ceil(self.config.episode_duration_s / self.config.control_dt)
        )
        self.n_arms = self.config.n_arms
        self.n_channels = self.config.n_channels
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_arms, self.n_channels),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        self.simulation: PhaseOctopusMuscleSimulation | None = None
        self._step_count = 0
        self._previous_forward = 0.0

    def _require_simulation(self) -> PhaseOctopusMuscleSimulation:
        if self.simulation is None:
            raise RuntimeError("reset() must be called before stepping the environment")
        return self.simulation

    def _observation(self) -> np.ndarray:
        return self._require_simulation().observation().astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        simulation = self._require_simulation()
        command = np.asarray(action, dtype=np.float64).reshape(
            self.n_arms, self.n_channels
        )
        command = np.clip((command + 1.0) * 0.5, 0.0, 1.0)
        policy = simulation.policies[0]
        policy.target_muscle_groups[:, :] = command[:, [0, 1, 2, 3, 4, 7, 8]]
        policy.base_suction_active[:] = command[:, 5]
        policy.middle_suction_active[:] = command[:, 6]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.simulation = PhaseOctopusMuscleSimulation(self.config)
        self._step_count = 0
        self._previous_forward = 0.0
        return self._observation(), {"time": self.simulation.time}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        simulation = self._require_simulation()
        self._apply_action(action)
        previous_position = simulation.sphere_position().copy()
        simulation.advance(self.config.control_dt, apply_phase=False)
        self._step_count += 1

        current_position = simulation.sphere_position()
        forward = float(previous_position[2] - current_position[2])
        lateral = float(abs(current_position[0]))
        reward = forward / self.config.base_length
        reward -= self.config.lateral_penalty * lateral / self.config.base_length
        terminated = not simulation.is_finite()
        truncated = self._step_count >= self.horizon and not terminated
        if terminated:
            reward = float(self.config.failure_penalty * self.config.reward_scale)
        info = {
            "time": simulation.time,
            "forward_progress": forward,
            "lateral_deviation": lateral,
            "strain_energy": simulation.strain_energy(),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return an RGB snapshot of the current crawler state."""
        if self.render_mode is None:
            return None

        simulation = self._require_simulation()
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.patches import Circle

        figure, axis = plt.subplots(figsize=(8, 5), dpi=100)
        positions = [np.asarray(rod.position_collection) for rod in simulation.rods]
        points = np.concatenate([rod[[0, 2], :].T for rod in positions], axis=0)
        sphere_position = simulation.sphere_position()
        points = np.concatenate([points, sphere_position[[0, 2]][None, :]], axis=0)
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        span = np.maximum(upper - lower, 0.2)
        center = 0.5 * (lower + upper)
        half_span = 0.5 * np.max(span) + 0.12
        axis.set_xlim(center[0] - half_span, center[0] + half_span)
        axis.set_ylim(center[1] - half_span, center[1] + half_span)
        for rod in positions:
            axis.plot(rod[0], rod[2], color="tab:blue", linewidth=2)
        axis.add_patch(
            Circle(
                (sphere_position[0], sphere_position[2]),
                radius=self.config.base_sphere_radius,
                color="tab:orange",
            )
        )
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        axis.set_title(f"OctoMuscleCrawl-v0  t={simulation.time:.2f} s")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
        return frame

    def close(self) -> None:
        self.simulation = None
