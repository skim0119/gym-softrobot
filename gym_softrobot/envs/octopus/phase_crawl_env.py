"""Feedback-controlled Octopus crawler with abstract phase actuation."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_softrobot.envs.octopus.phase_physics.config import OctopusV1Config
from gym_softrobot.envs.octopus.phase_physics.simulation import PhaseOctopusSimulation


class OctoPhaseCrawlEnv(gym.Env[np.ndarray, np.ndarray]):
    """Eight-arm crawling with stiffness, extension, suction, and bend control.

    This is the Gymnasium counterpart of the abstract ``Octopus-v1`` phase
    policy.  Actions control one interval at a time and therefore form a
    feedback environment rather than a single open-loop bandit pull.
    """

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        *,
        config: OctopusV1Config | None = None,
        horizon: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.config = config or OctopusV1Config()
        self.horizon = horizon or max(
            1, math.ceil(self.config.episode_duration_s / self.config.control_dt)
        )
        self.n_arms = self.config.n_arms
        self.n_channels = 5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_arms, self.n_channels), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.simulation: PhaseOctopusSimulation | None = None
        self._step_count = 0

    def _require_simulation(self) -> PhaseOctopusSimulation:
        if self.simulation is None:
            raise RuntimeError("reset() must be called before stepping the environment")
        return self.simulation

    def _observation(self) -> np.ndarray:
        return self._require_simulation().observation().astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        simulation = self._require_simulation()
        command = np.asarray(action, dtype=np.float64).reshape(self.n_arms, 5)
        policy = simulation.policies[0]
        policy.target_stiffness[:] = (command[:, 0] + 1.0) * 0.5
        policy.target_extension[:] = command[:, 1]
        policy.base_suction_active[:] = (command[:, 2] + 1.0) * 0.5
        policy.middle_suction_active[:] = (command[:, 3] + 1.0) * 0.5
        policy.target_bend[:] = command[:, 4] * (np.pi / 2.0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.simulation = PhaseOctopusSimulation(self.config)
        self._step_count = 0
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

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.simulation = None


# Backward-compatible import name used during the initial migration.
OctopusPhaseCrawlEnv = OctoPhaseCrawlEnv
