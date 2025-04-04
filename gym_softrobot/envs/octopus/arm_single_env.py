from typing import Optional

from gymnasium import spaces, Env
from gymnasium.utils import seeding

from collections import defaultdict
import time
import copy

import numpy as np
from scipy.interpolate import interp1d

import elastica as el
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check

from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.config import RendererType
from gym_softrobot.envs.octopus.build import build_arm
from gym_softrobot.utils.custom_elastica.callback_func import (
    RodCallBack,
)
from gym_softrobot.utils.linalg import do_normalization
from gym_softrobot.utils.render.post_processing import plot_video
from gym_softrobot.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)


class BaseSimulator(
    el.BaseSystemCollection,
    el.Constraints,
    el.Connections,
    el.Forcing,
    el.Damping,
    el.CallBacks,
):
    pass


class ArmSingleEnv(Env):
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

    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        final_time=10.0,
        time_step=7.0e-5,
        recording_fps=20,
        n_elems=50,
        n_action=7,
        control_penalty_coeff=0.001,
        config_generate_video=False,
        policy_mode="centralized",
    ):
        # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))
        self.control_penalty_coeff = control_penalty_coeff

        self.n_elems = n_elems
        self.n_seg = n_elems - 1
        self.policy_mode = policy_mode

        # Spaces
        self.n_action = n_action  # number of interpolation point (3 curvatures)
        action_size = (self.n_action,)
        action_low = np.ones(action_size) * (-22)
        action_high = np.ones(action_size) * (22)
        self.action_space = spaces.Box(
            action_low, action_high, shape=action_size, dtype=np.float32
        )
        self._observation_size = (
            25,
        )  # ((self.n_seg + (self.n_elems+1) * 4 + self.n_action + 2),) # 2 for target
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self._observation_size, dtype=np.float32
        )

        self.metadata = {}
        self.reward_range = 10.0
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

        self.kappa_range = [-49.33508476187419, 49.33545827754751]
        self.sigma_range = [-0.041537734572300755, 0.1019431615063144]
        self.kappa_rate_range = [-21.063520620377012, 24.664591289161944]
        self.sigma_rate_range = [-0.08387293777925456, 0.06838264835333994]

    def seed(self, seed=None):
        # Deprecated in new gym
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def summary(
        self,
    ):
        print(
            f"""
        {self.final_time=}
        {self.time_step=}
        {self.total_steps=}
        {self.step_skip=}
        simulation time per action: {1.0/self.step_skip=}
        max number of action per episode: {self.total_steps / self.step_skip}

        {self.n_elems=}
        {self.action_space=}
        {self.observation_space=}
        {self.reward_range=}
        """
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.simulator = BaseSimulator()

        self.shearable_rod = build_arm(
            self.simulator,
            self.n_elems,
        )

        # CallBack
        if self.config_generate_video:
            self.rod_parameters_dict = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                RodCallBack,
                step_skip=self.step_skip,
                callback_params=rod_parameters_dict,
            )

        """ Finalize the simulator and create time stepper """
        self.StatefulStepper = el.PositionVerlet()
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        self.time = np.float64(0.0)
        self.counter = 0

        # Set Target
        self._target = np.array([1.0, 0.0])  # (2-0.5)*self.np_random.random(2) + 0.5
        # self._target /= np.linalg.norm(self._target) # I don't see why this is here
        # print(self._target)

        # Initial State
        rod = self.shearable_rod
        self.prev_kappa_state = copy.deepcopy(rod.kappa[0])
        self.prev_com_state = rod.compute_position_center_of_mass()[:2]
        state = self.get_state()

        # Preprocessing
        self.prev_dist_to_target = np.linalg.norm(
            self.shearable_rod.compute_position_center_of_mass()[:2] - self._target,
            ord=2,
        )
        # self.prev_cm_vel = self.shearable_rod.compute_velocity_center_of_mass()

        if return_info:
            return state, {}
        else:
            return state

    def get_state(self):
        # Build state
        rod = self.shearable_rod
        kappa_state = rod.kappa[0]
        kappa_rate_state = kappa_state - self.prev_kappa_state
        self.prev_kappa_state[...] = kappa_state

        mean_kappa_state = np.mean(kappa_state.reshape((7, 7)), axis=1)
        mean_kappa_rate_state = np.mean(kappa_rate_state.reshape((7, 7)), axis=1)

        com_state = rod.compute_position_center_of_mass()[:2]
        com_rate_state = com_state - self.prev_com_state
        self.prev_com_state[...] = com_state
        # pos_state1 = rod.position_collection[0] # x
        # pos_state2 = rod.position_collection[1] # y
        # vel_state1 = rod.velocity_collection[0] # x
        # vel_state2 = rod.velocity_collection[1] # y
        previous_action = self._prev_action.copy()
        target = self._target
        # state = np.hstack([
        #     mean_kappa_state, mean_kappa_rate_state,com_rate_state,# com_state, #pos_state1, pos_state2, vel_state1, vel_state2,
        #     previous_action, target]).astype(np.float32)
        normalized_kappa_state = self.normalize_state(
            mean_kappa_state, mean_kappa_rate_state
        )
        state = np.hstack(
            [
                normalized_kappa_state,
                com_rate_state,  # com_state, #pos_state1, pos_state2, vel_state1, vel_state2,
                previous_action,
                target,
            ]
        ).astype(np.float32)
        return state

    def normalize_state(self, kappa, kappa_rate):
        k = do_normalization(kappa, self.kappa_range)
        kr = do_normalization(kappa_rate, self.kappa_rate_range)
        return np.concatenate([k, kr])

    def set_action(self, action) -> None:
        self._prev_action[:] = action
        # action = np.concatenate([[0], action, [0]], axis=-1)
        action = interp1d(
            np.linspace(0, 1, self.n_action),  # added zero on the boundary
            action,
            kind="cubic",
            axis=-1,
        )(np.linspace(0, 1, self.n_seg))
        self.shearable_rod.rest_kappa[0, :] = action  # Planar curvature

    def step(self, action):
        rest_kappa = action  # alias

        """ Set intrinsic strains (set actions) """
        self.set_action(rest_kappa)

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
        terminated = False
        truncated = False
        self.survive_reward = 0.0
        self.forward_reward = 0.0
        self.control_panelty = (
            self.control_penalty_coeff * np.square(rest_kappa.ravel()).mean()
        )
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = (
            _isnan_check(
                np.concatenate(
                    [
                        self.shearable_rod.position_collection,
                        self.shearable_rod.velocity_collection,
                    ]
                )
            )
            or np.linalg.norm(self.shearable_rod.omega_collection) > 250
        )

        if invalid_values_condition:
            print(f" Nan detected in, exiting simulation now. {self.time=}")
            terminated = True
            truncated = True
            self.survive_reward = -1.0  # 50.0
        else:
            self.cm_pos = self.shearable_rod.compute_position_center_of_mass()[:2]
            dist_to_target = np.linalg.norm(self.cm_pos - self._target, ord=2)
            self.forward_velocity = self.prev_dist_to_target - dist_to_target
            self.forward_reward = (
                np.exp(-dist_to_target / 0.35) - 0.096
            )  # np.exp((self.cm_pos[0]-0.18)/0.35)-1#np.exp(self.forward_velocity)-1#(self.prev_dist_to_target - dist_to_target) * 10
            self.prev_dist_to_target = dist_to_target
            """ Goal """
            if dist_to_target < 0.1:
                self.survive_reward = 5.0
                terminated = True

        """ Time limit """
        timelimit = False
        if self.time > self.final_time:
            timelimit = True
            terminated = True

        reward = self.forward_reward - self.control_panelty + self.survive_reward
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

        return states, reward, terminated, truncated, info

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
            assert issubclass(Session, BaseRenderer), (
                "Rendering module is not properly subclassed"
            )
            assert issubclass(Session, BaseElasticaRendererSession), (
                "Rendering module is not properly subclassed"
            )
            self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)
            self.renderer = Session(width=maxwidth, height=int(maxwidth * aspect_ratio))
            self.renderer.add_rods(
                [self.shearable_rod]
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
