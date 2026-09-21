"""Moving-target tracking task for the Spirob tendon arm."""

from __future__ import annotations

from typing import Any

import numpy as np

from .reward import tendon_arm_tracking_reward
from .tendon_arm_env import TendonArmReachEnv


class TendonArmTrackingEnv(TendonArmReachEnv):
    """Track a phase-randomized figure-eight target with target-velocity feedback."""

    include_target_velocity = True

    def __init__(
        self,
        *,
        figure_eight_period: float = 8.0,
        figure_eight_x_amplitude: float = 0.06,
        figure_eight_z_amplitude: float = 0.04,
        figure_eight_center_y: float = -0.26,
        **kwargs: Any,
    ) -> None:
        parameters = (
            figure_eight_period,
            figure_eight_x_amplitude,
            figure_eight_z_amplitude,
            figure_eight_center_y,
        )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("figure-eight parameters must be finite")
        if figure_eight_period <= 0.0:
            raise ValueError("figure_eight_period must be positive")
        if figure_eight_x_amplitude <= 0.0 or figure_eight_z_amplitude <= 0.0:
            raise ValueError("figure-eight amplitudes must be positive")

        self.figure_eight_period = float(figure_eight_period)
        self.figure_eight_x_amplitude = float(figure_eight_x_amplitude)
        self.figure_eight_z_amplitude = float(figure_eight_z_amplitude)
        self.figure_eight_center_y = float(figure_eight_center_y)
        self._phase = 0.0
        super().__init__(**kwargs)

    def _episode_duration(self) -> float:
        return 2.0 * self.figure_eight_period

    def _target_state_at(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        omega = 2.0 * np.pi / self.figure_eight_period
        phase = omega * time + self._phase
        position = np.array(
            (
                self.figure_eight_x_amplitude * np.sin(phase),
                self.figure_eight_center_y,
                self.figure_eight_z_amplitude * np.sin(2.0 * phase),
            ),
            dtype=np.float64,
        )
        velocity = np.array(
            (
                self.figure_eight_x_amplitude * omega * np.cos(phase),
                0.0,
                2.0 * self.figure_eight_z_amplitude * omega * np.cos(2.0 * phase),
            ),
            dtype=np.float64,
        )
        return position, velocity

    def _reset_target(self, options: dict[str, Any]) -> None:
        if "target" in options:
            raise ValueError("TendonArmReach-v1 uses a moving target; target is fixed")
        phase = options.get("phase")
        if phase is None:
            phase = self.np_random.uniform(0.0, 2.0 * np.pi)
        self._phase = float(phase)
        if not np.isfinite(self._phase):
            raise ValueError("options['phase'] must be finite")
        self._target, self._target_velocity = self._target_state_at(0.0)

    def _advance_target(self) -> None:
        target_time = (
            (self._step_count + 1)
            * self.time_step
            * self.simulation_steps_per_action
        )
        self._target, self._target_velocity = self._target_state_at(target_time)

    def _compute_reward(
        self, distance: float, tip_speed: float, failure: bool
    ) -> tuple[float, dict[str, float]]:
        tip_velocity = self.rod.velocity_collection[:, -1]
        if not np.all(np.isfinite(tip_velocity)):
            relative_speed = 0.0
        else:
            relative_speed = float(np.linalg.norm(tip_velocity - self._target_velocity))
        return tendon_arm_tracking_reward(
            distance=distance,
            relative_speed=relative_speed,
            failure=failure,
        )
