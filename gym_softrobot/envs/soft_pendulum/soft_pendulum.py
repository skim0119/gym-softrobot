from typing import Optional

import gym
from gym import core
from gym import error, spaces, utils
from gym.utils import seeding

from collections import defaultdict
import time
import copy

import numpy as np
from scipy.interpolate import interp1d

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check

from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.config import RendererType
from gym_softrobot.envs.soft_pendulum.build import build_soft_pendulum
from gym_softrobot.utils.custom_elastica.callback_func import (
    RodCallBack,
    RigidCylinderCallBack,
)
from gym_softrobot.utils.render.post_processing import plot_video
from gym_softrobot.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class SoftPendulumEnv(core.Env):
    """
    Description:
    Source:
    Observation:
    Actions:
    Reward:
    Starting State:
    Episode Termination:
    Solved Requirements:
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        final_time=5.0,
        time_step=1.0e-4,
        recording_fps=25,
        n_elems=50,
        config_generate_video=False,
    ):

        # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))

        self.n_elems = n_elems
        self.n_seg = n_elems - 1

        # Spaces
        self.n_action = 1
        action_size = (self.n_action,)
        action_low = np.ones(action_size) * (-22)
        action_high = np.ones(action_size) * (22)
        self.action_space = spaces.Box(
            action_low, action_high, shape=action_size, dtype=np.float32
        )
        self._observation_size = (
            (4),
        )  # 2 for target
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self._observation_size, dtype=np.float32
        )

        self.metadata = {}
        self.reward_range = 100.0
        self._prev_action = np.zeros(
            list(self.action_space.shape), dtype=self.action_space.dtype
        )

        # Configurations
        self.config_generate_video = config_generate_video

        # Rendering-related
        self.viewer = None
        self.renderer = None

        # Determinism
        self.seed()

    def seed(self, seed=None):
        # Deprecated in new gym
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.simulator = BaseSimulator()

        self.point_force = np.zeros((self.n_action))

        self.shearable_rod = build_soft_pendulum(
            self.simulator,
            self.n_elems,
            self.point_force,
            self.np_random,
        )

        # CallBack
        if self.config_generate_video:
            self.rod_parameters_dict = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                RodCallBack,
                step_skip=self.step_skip,
                callback_params=self.rod_parameters_dict,
            )

        """ Finalize the simulator and create time stepper """
        self.StatefulStepper = PositionVerlet()
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        self.time = np.float64(0.0)
        self.counter = 0

        # Initial State
        state = self.get_state()

        if return_info:
            return state, {}
        else:
            return state

    def get_state(self):
        # Build state
        rod = self.shearable_rod
        pos_state1 = rod.position_collection[0, 0]
        vel_state1 = rod.velocity_collection[0, 0]
        tangents_mean = np.mean(self.shearable_rod.tangents, axis=1)
        theta = np.arctan(tangents_mean[0] / tangents_mean[1])  # Target angle is 0
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        previous_action = self._prev_action.copy()
        state = np.hstack([pos_state1, vel_state1, previous_action, theta]).astype(
            np.float32
        )
        return state

    def set_action(self, action) -> None:
        # FIXME: why do you set current action to previous action?
        self._prev_action[:] = action
        self.point_force[:] = action
        # action = np.concatenate([[0], action, [0]], axis=-1)
        # action = interp1d(
        #         np.linspace(0,1,self.n_action+2), # added zero on the boundary
        #         action,
        #         kind='cubic',
        #         axis=-1,
        #     )(np.linspace(0,1,self.n_seg))
        # self.shearable_rod.rest_kappa[0, :] = action # Planar curvature

    def step(self, action):

        self.set_action(action)

        """ Post-simulation """

        """ Run the simulation for one step """
        stime = time.perf_counter()
        for _ in range(self.step_skip):
            self.time = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time,
                self.time_step,
            )
        etime = time.perf_counter()
        # print(f'{self.counter=}, {etime-stime}sec, {self.time=}')

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        survive_reward = 0.0
        forward_reward = 0.0
        # FIXME: How to set control penalty
        control_penalty = 0.0  # 0.005 * np.square(rest_kappa.ravel()).mean()
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(
            np.concatenate(
                [
                    self.shearable_rod.position_collection,
                    self.shearable_rod.velocity_collection,
                ]
            )
        )

        if invalid_values_condition:
            print(f" Nan detected in, exiting simulation now. {self.time=}")
            done = True
            survive_reward = -50.0
        else:
            distance_to_origin = np.abs(self.shearable_rod.position_collection[0, 0])
            tangents_mean = np.mean(self.shearable_rod.tangents, axis=1)
            theta = np.arctan(tangents_mean[0] / tangents_mean[1])  # Target angle is 0
            distance_to_target_angle = ((theta + np.pi) % (2 * np.pi)) - np.pi
            forward_reward = distance_to_origin * 10 + distance_to_target_angle ** 2
            # cm_pos = self.shearable_rod.compute_position_center_of_mass()[:2]
            # dist_to_target = np.linalg.norm(cm_pos - self._target, ord=2)
            # forward_reward = (self.prev_dist_to_target - dist_to_target) * 10
            # self.prev_dist_to_target = dist_to_target
            """ Goal """
            # FIXME: How to set survival reward
            # if distance_to_origin < 0.01  and distance_to_target_angle < 0.01:
            #     survive_reward = 100.0
            #     done = True

        """ Time limit """
        timelimit = False
        if self.time > self.final_time:
            timelimit = True
            done = True

        reward = forward_reward - control_penalty + survive_reward
        # reward *= 10 # Reward scaling
        # print(f'{reward=:.3f}: {forward_reward=:.3f}, {control_panelty=:.3f}, {survive_reward=:.3f}')

        """ Return state:
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        # systems = [self.shearable_rod]
        states = self.get_state()

        # Info
        info = {
            "time": self.time,
            "rod": self.shearable_rod,
            "TimeLimit.truncated": timelimit,
        }

        self.counter += 1

        return states, reward, done, info

    def save_data(self, filename_video, fps):
        if self.config_generate_video:
            filename_video = f"save/{filename_video}"
            plot_video(self.rod_parameters_dict, filename_video, margin=0.2, fps=fps)

    def render(self, mode="human", close=False):
        maxwidth = 800
        aspect_ratio = 3 / 4

        if self.viewer is None:
            from gym_softrobot.utils.render import pyglet_rendering

            self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)

        if self.renderer is None:
            # Switch renderer depending on configuration
            if RENDERER_CONFIG == RendererType.POVRAY:
                from gym_softrobot.utils.render.povray_renderer import Session
            elif RENDERER_CONFIG == RendererType.MATPLOTLIB:
                from gym_softrobot.utils.render.matplotlib_renderer import Session
            else:
                raise NotImplementedError("Rendering module is not imported properly")
            assert issubclass(
                Session, BaseRenderer
            ), "Rendering module is not properly subclassed"
            assert issubclass(
                Session, BaseElasticaRendererSession
            ), "Rendering module is not properly subclassed"
            self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)
            self.renderer = Session(width=maxwidth, height=int(maxwidth * aspect_ratio))
            self.renderer.add_rod(
                self.shearable_rod
            )  # TODO: maybe need add_rod instead
            self.renderer.add_point(self._target.tolist() + [0], 0.02)

        # POVRAY
        if RENDERER_CONFIG == RendererType.POVRAY:
            state_image = self.renderer.render(
                maxwidth, int(maxwidth * aspect_ratio * 0.7)
            )
            state_image_side = self.renderer.render(
                maxwidth // 2,
                int(maxwidth * aspect_ratio * 0.3),
                camera_param=("location", [0.0, 0.0, -0.5], "look_at", [0.0, 0, 0]),
            )
            state_image_top = self.renderer.render(
                maxwidth // 2,
                int(maxwidth * aspect_ratio * 0.3),
                camera_param=("location", [0.0, 0.3, 0.0], "look_at", [0.0, 0, 0]),
            )

            state_image = np.vstack(
                [state_image, np.hstack([state_image_side, state_image_top])]
            )
        elif RENDERER_CONFIG == RendererType.MATPLOTLIB:
            state_image = self.renderer.render()
        else:
            raise NotImplementedError("Rendering module is not imported properly")

        self.viewer.imshow(state_image)

        return state_image

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
