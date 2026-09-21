"""Gymnasium port of the rl-cr-robot tendon-driven reaching environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from elastica.boundary_conditions import OneEndFixedBC
from elastica.dissipation import AnalyticalLinearDamper
from elastica.external_forces import GravityForces
from elastica.modules import (
    BaseSystemCollection,
    Constraints,
    Damping,
    Forcing,
)
from elastica.timestepper.symplectic_steppers import PositionVerlet
from gymnasium import Env, spaces

from .rendering import add_tapered_rod
from .reward import tendon_arm_reward
from .spirob_geometry import (
    SPIROB_BASE_LENGTH,
    SPIROB_BASE_RADIUS,
    create_spirob_rod,
)
from .tendon_forces import TendonActuation


_N_ELEMENTS = 25
_TIME_STEP = 3.0e-5
_SIMULATION_STEPS_PER_ACTION = 1111  # approximately 30 policy controls per second
_EPISODE_TIME = 6.0


class TendonArmSimulator(BaseSystemCollection, Constraints, Forcing, Damping):
    pass


class TendonArmReachEnv(Env[np.ndarray, np.ndarray]):
    """Reach a random 3-D target with a twelve-tendon Spirob arm.

    Six tendons route along the full arm and six along its proximal half.
    Commands in ``[0, 1]`` are mapped directly to tensions in
    ``[0, max_tension]``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    include_target_velocity = False

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_tension: float = 55.2,
        damping_constant: float = 1.0,
        stack_frame: int = 5,
        target: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if not np.isfinite(max_tension) or max_tension <= 0.0:
            raise ValueError("max_tension must be finite and positive")
        if not np.isfinite(damping_constant) or damping_constant < 0.0:
            raise ValueError("damping_constant must be finite and nonnegative")
        if stack_frame < 2:
            raise ValueError("stack_frame must be at least 2")

        self.render_mode = render_mode
        self.n_elements = _N_ELEMENTS
        self.time_step = _TIME_STEP
        self.simulation_steps_per_action = _SIMULATION_STEPS_PER_ACTION
        self.episode_time = self._episode_duration()
        self.max_episode_steps = max(
            1, int(self.episode_time / self.time_step / self.simulation_steps_per_action)
        )
        self.max_tension = max_tension
        self.damping_constant = damping_constant
        self.stack_frame = stack_frame
        self.base_length = SPIROB_BASE_LENGTH
        self.base_radius = SPIROB_BASE_RADIUS
        self._fixed_target = (
            None if target is None else np.asarray(target, dtype=np.float64)
        )
        if self._fixed_target is not None and self._fixed_target.shape != (3,):
            raise ValueError("target must contain exactly three coordinates")

        self.action_space = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
        observation_size = 15 * stack_frame + 50 + (
            3 if self.include_target_velocity else 0
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
        )
        self._node_indices = np.linspace(
            1, self.n_elements - 1, num=10, dtype=np.int64
        )
        self._tensions = np.zeros(12)
        self._action_history = np.zeros((stack_frame, 12))
        self._tip_history = np.zeros((stack_frame, 3))
        self._target = np.zeros(3)
        self._target_velocity = np.zeros(3)
        self._step_count = 0
        self._time = np.float64(0.0)
        self._build_simulator()

    def _episode_duration(self) -> float:
        return _EPISODE_TIME

    def _build_simulator(self) -> None:
        self.simulator = TendonArmSimulator()
        self.rod = create_spirob_rod(
            n_elements=self.n_elements,
            start=np.zeros(3),
            direction=np.array((0.0, -1.0, 0.0)),
            normal=np.array((0.0, 0.0, 1.0)),
            density=1000.0,
            youngs_modulus=3.0e6,
            base_length=self.base_length,
            base_radius=self.base_radius,
        )
        self.simulator.append(self.rod)
        self.simulator.constrain(self.rod).using(
            OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        long_nodes = self._contact_nodes(0.95)
        short_nodes = self._contact_nodes(0.50)
        self.simulator.add_forcing_to(self.rod).using(
            TendonActuation,
            contact_nodes=long_nodes,
            tensions=self._tensions[:6],
            rod_radii=self.rod.radius,
        )
        self.simulator.add_forcing_to(self.rod).using(
            TendonActuation,
            contact_nodes=short_nodes,
            tensions=self._tensions[6:],
            rod_radii=self.rod.radius,
        )
        self.simulator.add_forcing_to(self.rod).using(
            GravityForces, acc_gravity=np.zeros(3)
        )
        self.simulator.dampen(self.rod).using(
            AnalyticalLinearDamper,
            damping_constant=self.damping_constant,
            time_step=self.time_step,
        )
        self.simulator.finalize()
        self._stepper = PositionVerlet()

        self._initial_state = {
            "position_collection": self.rod.position_collection.copy(),
            "director_collection": self.rod.director_collection.copy(),
            "lengths": self.rod.lengths.copy(),
            "tangents": self.rod.tangents.copy(),
        }

    def _contact_nodes(self, final_fraction: float) -> np.ndarray:
        final_node = min(
            self.n_elements - 2, max(2, round(final_fraction * self.n_elements))
        )
        return np.rint(np.linspace(2, final_node, num=6)).astype(np.int64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self._reset_target(options)

        self._tensions[:] = 0.0
        self._restore_rod()
        self._action_history.fill(0.0)
        self._tip_history.fill(0.0)
        tip = self.rod.position_collection[:, -1].copy()
        self._tip_history[-1] = tip
        distance = float(np.linalg.norm(self._target - tip))
        self._step_count = 0
        self._time = np.float64(0.0)
        observation = self._observation()
        return observation, {
            "target_position": self._target.copy(),
            "target_velocity": self._target_velocity.copy(),
            "distance_to_target": distance,
        }

    def _reset_target(self, options: dict[str, Any]) -> None:
        option_target = options.get("target")
        if option_target is not None:
            self._target = np.asarray(option_target, dtype=np.float64)
            if self._target.shape != (3,):
                raise ValueError("options['target'] must contain three coordinates")
        elif self._fixed_target is not None:
            self._target = self._fixed_target.copy()
        else:
            self._target = self._sample_target()

        self._target_velocity.fill(0.0)

    def _advance_target(self) -> None:
        """Advance task-specific target dynamics by one control step."""

    def _sample_target(self) -> np.ndarray:
        radius = 0.1 * np.sqrt(self.np_random.uniform())
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        return np.array(
            (
                radius * np.cos(angle),
                self.np_random.uniform(-0.26, -0.20),
                radius * np.sin(angle),
            )
        )

    def _restore_rod(self) -> None:
        for name, value in self._initial_state.items():
            getattr(self.rod, name)[:] = value
        for name in (
            "velocity_collection",
            "acceleration_collection",
            "omega_collection",
            "alpha_collection",
            "internal_forces",
            "internal_torques",
            "external_forces",
            "external_torques",
        ):
            getattr(self.rod, name).fill(0.0)
        self.rod.dilatation.fill(1.0)
        self.rod.voronoi_dilatation.fill(1.0)
        self.rod.dilatation_rate.fill(0.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")

        scaled_action = action.astype(np.float64) * self.max_tension
        self._action_history[:-1] = self._action_history[1:]
        self._action_history[-1] = scaled_action
        if self._step_count == 0:
            self._action_history[-2] = scaled_action
        # Mutate in place: each forcing object holds a view of this array.
        self._tensions[:] = scaled_action

        for _ in range(self.simulation_steps_per_action):
            self._time = self._stepper.step(self.simulator, self._time, self.time_step)

        self._advance_target()
        tip = self.rod.position_collection[:, -1].copy()
        invalid = not np.all(np.isfinite(tip))
        if invalid:
            tip = np.nan_to_num(tip, nan=-self.base_length)
        distance = float(np.linalg.norm(self._target - tip))
        tip_speed = float(np.linalg.norm(self.rod.velocity_collection[:, -1]))
        if not np.isfinite(tip_speed):
            tip_speed = 0.0
        terminated = invalid or distance > self.base_length

        reward, components = self._compute_reward(distance, tip_speed, terminated)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        self._tip_history[:-1] = self._tip_history[1:]
        self._tip_history[-1] = tip
        observation = self._observation()
        info = {
            "target_position": self._target.copy(),
            "target_velocity": self._target_velocity.copy(),
            "distance_to_target": distance,
            "tip_position": tip.copy(),
            "tip_speed": tip_speed,
            "tendon_tensions": scaled_action.copy(),
            "reward_components": components,
        }
        if invalid:
            info["termination_reason"] = "non_finite_state"
        elif distance > self.base_length:
            info["termination_reason"] = "control_collapse"
        elif truncated:
            info["termination_reason"] = "time_limit"
        return observation, reward, bool(terminated), bool(truncated), info

    def _compute_reward(
        self, distance: float, tip_speed: float, failure: bool
    ) -> tuple[float, dict[str, float]]:
        return tendon_arm_reward(
            distance=distance,
            tip_speed=tip_speed,
            failure=failure,
        )

    def _observation(self) -> np.ndarray:
        tip = self.rod.position_collection[:, -1]
        tip_velocity = self.rod.velocity_collection[:, -1]
        node_velocities = self.rod.velocity_collection[:, self._node_indices]
        node_speeds = np.linalg.norm(node_velocities, axis=0)
        node_positions = self.rod.position_collection[:, self._node_indices].T
        observation = np.concatenate(
            (
                self._tip_history.ravel(),
                self._target,
                self._target_velocity if self.include_target_velocity else np.empty(0),
                self._target - tip,
                tip_velocity,
                np.array((np.linalg.norm(tip_velocity),)),
                node_speeds,
                node_positions.ravel(),
                self._action_history.ravel(),
            )
        )
        return observation.astype(np.float32)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(6, 6), dpi=100)
        axis = figure.add_subplot(111, projection="3d")
        positions = self.rod.position_collection
        add_tapered_rod(axis, positions, self.rod.radius)
        axis.scatter(*self._target, color="tab:red", marker="*", s=100)
        limit = self.base_length
        axis.set(
            xlim=(-limit / 2, limit / 2),
            ylim=(-limit - 0.05, limit / 2),
            zlim=(-limit / 2, limit / 2),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
        )
        axis.set_box_aspect((limit, limit + 0.20, limit))
        figure.canvas.draw()
        image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
        return image

    def close(self) -> None:
        return None
