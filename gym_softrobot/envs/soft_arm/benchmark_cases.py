"""Modern Gymnasium ports of Elastica-RL-control benchmark Cases 2--4."""

from collections import defaultdict
from typing import Any

import numpy as np
from elastica import Cylinder, OneEndFixedBC, RodCylinderContact
from gymnasium import spaces

from gym_softrobot.envs.soft_arm.soft_arm_tracking import SoftArmTrackingEnv
from gym_softrobot.utils.custom_elastica.muscle_torque import (
    MuscleTorquesWithVaryingBetaSplines,
)


def _quaternion(directors: np.ndarray) -> np.ndarray:
    """Return a normalized scalar-first quaternion from a director matrix."""
    matrix = directors.T
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        nxt = (axis + 1) % 3
        last = (axis + 2) % 3
        scale = 2.0 * np.sqrt(
            1.0 + matrix[axis, axis] - matrix[nxt, nxt] - matrix[last, last]
        )
        quat = np.zeros(4)
        quat[axis + 1] = 0.25 * scale
        quat[0] = (matrix[last, nxt] - matrix[nxt, last]) / scale
        quat[nxt + 1] = (matrix[nxt, axis] + matrix[axis, nxt]) / scale
        quat[last + 1] = (matrix[last, axis] + matrix[axis, last]) / scale
    return quat / np.linalg.norm(quat)


class ElasticaArmReachEnv(SoftArmTrackingEnv):
    """Case 2: reach a target and match its prescribed tip orientation."""

    def __init__(
        self,
        render_mode: str | None = None,
        number_of_control_points: int = 6,
        final_time: float = 5.0,
    ) -> None:
        self.target_orientation = np.array(
            [
                [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4)],
                [0.0, 1.0, 0.0],
                [-np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)],
            ],
            dtype=np.float64,
        )
        super().__init__(
            game_mode=1,
            render_mode=render_mode,
            number_of_control_points=number_of_control_points,
            final_time=final_time,
        )
        self.target_location = np.array([-400.0, 600.0, 200.0])
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(3 * self.number_of_control_points,),
            dtype=np.float64,
        )
        base_size = 2 * self.number_of_observation_segments + 6
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(base_size + 8,), dtype=np.float64
        )

    def _configure_task_systems(self) -> None:
        self.sphere.director_collection[..., 0] = self.target_orientation
        self.target_tip_orientation = _quaternion(self.target_orientation)
        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir: list[float] = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.alpha,
            direction="tangent",
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_twist_dir,
        )

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        rod_orientation = _quaternion(
            self.shearable_rod.director_collection[..., -1]
        )
        target_orientation = getattr(
            self, "target_tip_orientation", _quaternion(self.target_orientation)
        )
        return np.concatenate((state, rod_orientation, target_orientation))

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        count = self.number_of_control_points
        self.spline_points_func_array_twist_dir[:] = action[2 * count :]
        state, reward, terminated, truncated, info = super().step(action)
        rod_orientation = _quaternion(
            self.shearable_rod.director_collection[..., -1]
        )
        orientation_distance = 1.0 - float(
            np.dot(rod_orientation, self.target_tip_orientation) ** 2
        )
        reward -= 0.5 * orientation_distance**2
        info["orientation_distance"] = orientation_distance
        return state, reward, terminated, truncated, info


class _ObstacleArmEnv(SoftArmTrackingEnv):
    obstacle_samples = 5

    def __init__(
        self,
        *,
        planar: bool,
        number_of_control_points: int,
        number_of_obstacles: int,
        render_mode: str | None,
        final_time: float,
    ) -> None:
        self.planar = planar
        self.number_of_obstacles = number_of_obstacles
        self.obstacles: list[Cylinder] = []
        self.obstacle_state = np.empty((number_of_obstacles, self.obstacle_samples, 3))
        super().__init__(
            game_mode=1,
            render_mode=render_mode,
            number_of_control_points=number_of_control_points,
            final_time=final_time,
        )
        self.target_location = (
            np.array([-800.0, 500.0, 150.0])
            if planar
            else np.array([-800.0, 500.0, 350.0])
        )
        action_directions = 1 if planar else 2
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(action_directions * number_of_control_points,),
            dtype=np.float64,
        )
        base_size = 2 * self.number_of_observation_segments + 6
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(base_size + number_of_obstacles * self.obstacle_samples * 3,),
            dtype=np.float64,
        )

    def _obstacle_geometry(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        raise NotImplementedError

    def _configure_task_systems(self) -> None:
        for index, (start, direction, normal, length, radius) in enumerate(
            self._obstacle_geometry()
        ):
            obstacle = Cylinder(
                start=start,
                direction=direction,
                normal=normal,
                base_length=length,
                base_radius=radius,
                density=1.0e-3,
            )
            self.obstacles.append(obstacle)
            self.simulator.append(obstacle)
            self.simulator.constrain(obstacle).using(
                OneEndFixedBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )
            self.simulator.detect_contact_between(
                self.shearable_rod, obstacle
            ).using(
                RodCylinderContact, k=8.0e4, nu=4.0
            )
            fractions = np.linspace(0.0, 1.0, self.obstacle_samples)
            self.obstacle_state[index] = (
                start[None, :] + fractions[:, None] * length * direction[None, :]
            )

    def get_state(self) -> np.ndarray:
        return np.concatenate(
            (super().get_state(), self.obstacle_state.reshape(-1) / self.base_length)
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.planar:
            action = np.concatenate((action, np.zeros_like(action)))
        return super().step(action)


class ElasticaArmStructuredObstacleEnv(_ObstacleArmEnv):
    """Case 3: planar, underactuated reaching through eight fixed obstacles."""

    def __init__(
        self, render_mode: str | None = None, final_time: float = 5.0
    ) -> None:
        super().__init__(
            planar=True,
            number_of_control_points=2,
            number_of_obstacles=8,
            render_mode=render_mode,
            final_time=final_time,
        )

    def _obstacle_geometry(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        target = self.target_location
        starts = np.zeros((8, 3), dtype=np.float64)
        starts[0] = [target[0] + 200.0, target[1] - 500.0, target[2] - 30.0]
        starts[1] = [target[0] + 600.0, target[1] - 500.0, target[2] - 30.0]
        starts[2] = [target[0] + 400.0, target[1] - 500.0, target[2] + 220.0]
        starts[3] = starts[2]
        starts[3, 2] = starts[1, 2] - 120.0
        starts[4] = starts[3]
        starts[4, 2] -= 200.0
        starts[5] = starts[4]
        starts[5, 2] -= 200.0
        starts[6] = starts[2]
        starts[6, 2] += 200.0
        starts[7] = starts[6]
        starts[7, 2] += 200.0
        return [
            (
                start,
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                1000.0,
                100.0,
            )
            for start in starts
        ]


class ElasticaArmRandomObstacleEnv(_ObstacleArmEnv):
    """Case 4: 3D reaching through a seeded random nest of twelve obstacles."""

    def __init__(
        self, render_mode: str | None = None, final_time: float = 5.0
    ) -> None:
        super().__init__(
            planar=False,
            number_of_control_points=2,
            number_of_obstacles=12,
            render_mode=render_mode,
            final_time=final_time,
        )

    def _obstacle_geometry(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        geometry = []
        for _ in range(self.number_of_obstacles):
            alpha = self.np_random.uniform(np.pi / 3.0, 2.0 * np.pi / 3.0)
            beta = self.np_random.uniform(np.pi / 3.0, 2.0 * np.pi / 3.0)
            direction = np.array(
                [
                    np.cos(alpha) * np.cos(beta),
                    -np.sin(alpha),
                    np.cos(alpha) * np.sin(beta),
                ]
            )
            direction /= np.linalg.norm(direction)
            normal = np.array(
                [
                    np.sin(alpha) * np.cos(beta),
                    np.cos(alpha),
                    np.sin(alpha) * np.sin(beta),
                ]
            )
            normal /= np.linalg.norm(normal)
            midpoint = np.array(
                [
                    self.np_random.uniform(-550.0, -400.0),
                    self.target_location[1],
                    self.np_random.uniform(0.0, 1000.0),
                ]
            )
            start = midpoint - 500.0 * direction
            geometry.append((start, direction, normal, 1000.0, 30.0))
        return geometry
