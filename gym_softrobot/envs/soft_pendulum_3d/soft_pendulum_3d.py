"""Three-dimensional soft-pendulum environment."""

from collections import defaultdict
from typing import Optional

import numpy as np
from elastica import PositionVerlet
from elastica._calculus import _isnan_check
from gymnasium import Env, spaces

from gym_softrobot.envs.soft_pendulum import SoftPendulumEnv
from gym_softrobot.envs.soft_pendulum.soft_pendulum import BaseSimulator
from gym_softrobot.envs.soft_pendulum_3d.build import (
    MovingBaseController,
    build_soft_pendulum_3d,
)
from gym_softrobot.utils.custom_elastica.callback_func import RodCallBack


class SoftPendulum3DEnv(SoftPendulumEnv):
    """Stabilize a vertical soft pendulum by moving its base in two axes."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        final_time: float = 5.0,
        time_step: float = 1.0e-4,
        recording_fps: int = 25,
        n_elems: int = 50,
        config_generate_video: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            final_time=final_time,
            time_step=time_step,
            recording_fps=recording_fps,
            n_elems=n_elems,
            config_generate_video=config_generate_video,
            render_mode=render_mode,
        )
        self.n_action = 2
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_action,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )
        self._prev_action = np.zeros(self.n_action, dtype=np.float32)
        self.base_step = 1e-3
        self.base_limit = 0.5

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        Env.reset(self, seed=seed)
        self.simulator = BaseSimulator()
        self.base_controller = MovingBaseController()
        self._prev_action.fill(0.0)

        self.shearable_rod = build_soft_pendulum_3d(
            self.simulator,
            self.n_elems,
            self.base_controller,
            self.np_random,
            self.time_step,
        )

        if self.config_generate_video:
            self.rod_parameters_dict = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                RodCallBack,
                step_skip=self.step_skip,
                callback_params=self.rod_parameters_dict,
            )

        self.StatefulStepper = PositionVerlet()
        self.simulator.finalize()
        self.do_step = self.StatefulStepper.step
        self.time = np.float64(0.0)
        self.counter = 0
        self._target = np.zeros(3)
        return self.get_state(), {}

    def _tilt_angle(self) -> float:
        tangent = np.mean(self.shearable_rod.tangents, axis=1)
        tangent /= np.linalg.norm(tangent)
        return float(np.arccos(np.clip(tangent[2], -1.0, 1.0)))

    def get_state(self) -> np.ndarray:
        base_position = self.shearable_rod.position_collection[:, 0]
        base_velocity = self.shearable_rod.velocity_collection[:, 0]
        return np.hstack(
            [base_position, base_velocity, self._prev_action, self._tilt_angle()]
        ).astype(np.float32)

    def set_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        displacement = self.base_step * action
        next_position = self.base_controller.position.copy()
        next_position[:2] = np.clip(
            next_position[:2] + displacement,
            -self.base_limit,
            self.base_limit,
        )
        actual_displacement = next_position - self.base_controller.position
        self.base_controller.position[:] = next_position
        self.base_controller.velocity[:] = actual_displacement / (
            self.step_skip * self.time_step
        )
        self._prev_action[:] = action

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        self.set_action(action)

        for _ in range(self.step_skip):
            self.time = self.do_step(self.simulator, self.time, self.time_step)

        invalid = _isnan_check(
            np.concatenate(
                [
                    self.shearable_rod.position_collection,
                    self.shearable_rod.velocity_collection,
                ]
            )
        )
        tilt = self._tilt_angle()
        base_distance = np.linalg.norm(self.base_controller.position[:2])
        reward = -float(
            tilt**2
            + 0.1 * base_distance**2
            + 1e-3 * np.dot(action, action)
        )

        terminated = bool(invalid)
        truncated = bool(self.time >= self.final_time)
        if invalid:
            reward = -50.0

        self.counter += 1
        return (
            self.get_state(),
            reward,
            terminated,
            truncated,
            {"time": self.time, "tilt": tilt},
        )
