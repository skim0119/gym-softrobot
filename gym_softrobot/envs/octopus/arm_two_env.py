from typing import Optional, Any, Union, Dict

import gym
from gym import core
from gym import error, spaces, utils
from gym.utils import seeding

from collections import defaultdict, OrderedDict
import time
import copy

import numpy as np
from scipy.interpolate import interp1d

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check

from gym_softrobot.envs.octopus.build_muscle_octopus import build_two_arms

from gym_softrobot.config import RendererType
from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.utils.render.base_renderer import BaseRenderer, BaseElasticaRendererSession
from gym_softrobot.envs.octopus.controllable_constraint import ControllableFixConstraint


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class ArmTwoEnv(core.Env):
    """
    Description:
        Decentralized
    Source:
    Observation:
    Actions:
    Reward:
    Starting State:
    Episode Termination:
    Solved Requirements:
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 final_time=5.0,
                 time_step=5.0e-5,
                 recording_fps=25,
                 n_elems=20,
                 ):

        # Integrator type
        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))

        n_arm = 2
        self.n_arm = n_arm
        self.n_elems = n_elems
        self.n_seg = n_elems - 1
        self.n_sucker = 3
        self.sucker_location = [self.n_elems // (self.n_sucker * 2) * (2 * i + 1) for i in range(self.n_sucker)]
        self.control_location = [0] + self.sucker_location + [self.n_elems - 1]

        # Spaces
        n_action = 9
        self.n_action = n_action
        # TODO: for non-HER, use decentral training shapes
        self.action_space = spaces.Box(0.0, 1.0, shape=(n_arm * n_action,), dtype=np.float32)

        shared_space = 3
        self.grid_size = 1
        self._observation_size = (n_arm * (self.n_seg * 2 + n_action + n_arm + shared_space),)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32)

        self.metadata = {}
        self.reward_range = 100.0
        self._prev_action = np.zeros(list(self.action_space.shape),
                                     dtype=self.action_space.dtype)
        self._prev_kappa = np.zeros((n_arm, self.n_elems - 1), dtype=np.float32)

        # Configurations

        # Rendering-related
        self.viewer = None
        self.renderer = None

        # Determinism
        self.seed()

    def get_env_info(self):
        return dict(n_actions=self.n_action, n_agents=8)

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

        self.shearable_rods, self.rigid_rod, self.muscle_activations = build_two_arms(
            self.simulator,
            self.n_elems,
        )

        """ Controller Setup """
        constraint_ids = []
        for i in range(self.n_arm):
            each_arm_constraints = []
            for j in range(self.n_sucker):
                constraint_id = self.simulator.constrain(self.shearable_rods[i]).using(
                    ControllableFixConstraint,
                    index=self.sucker_location[j],
                ).id()
                each_arm_constraints.append(constraint_id)
            constraint_ids.append(each_arm_constraints)

        self.simulator.finalize()

        """ Sucker Controller Hook """
        self.sucker_controller = []
        for each_arm_constraints in constraint_ids:
            each_arm_sucker = []
            for constraint_id in each_arm_constraints:
                controllable_constraint = dict(self.simulator._constraints)[constraint_id]
                controller = controllable_constraint.get_controller
                controller.turn_on()
                each_arm_sucker.append(controller)
            self.sucker_controller.append(each_arm_sucker)

        """ Finalize the simulator and create time stepper """
        self.StatefulStepper = PositionVerlet()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # """ Return
        #     (1) total time steps for the simulation step iterations
        #     (2) systems for controller design
        # """
        # systems = [self.shearable_rod]
        self.time = np.float64(0.0)
        self.counter = 0
        # self.bias=self.shearable_rod.compute_position_center_of_mass()[0].copy()

        # Set Target
        # self._target = self.np_random.uniform(-self.grid_size, self.grid_size, size=2).astype(np.float32)
        self._target = np.array([5, 0], dtype=np.float32)

        # Initial State
        state = self.get_state()

        if return_info:
            return state, {}
        else:
            return state

    def get_state(self):
        # Build state
        kappa_state = np.vstack([rod.kappa[0] for rod in self.shearable_rods])
        pos_state1 = np.vstack([rod.position_collection[0] for rod in self.shearable_rods])  # x
        pos_state2 = np.vstack([rod.position_collection[1] for rod in self.shearable_rods])  # y
        vel_state1 = np.vstack([rod.velocity_collection[0] for rod in self.shearable_rods])  # x
        vel_state2 = np.vstack([rod.velocity_collection[1] for rod in self.shearable_rods])  # y
        previous_action = self._prev_action.reshape([self.n_arm, self.n_action])

        shared_state = np.concatenate([
            # self._target,  # 2
            # self.rigid_rod.position_collection[:, 0],  # 3
            self.rigid_rod.velocity_collection[:, 0],  # 3
            # self.rigid_rod.director_collection[:, :, 0].ravel(),  # 9: orientation
        ], dtype=np.float32)
        observation_state = np.hstack([
            kappa_state, self._prev_kappa,  # pos_state1, pos_state2, vel_state1, vel_state2,
            previous_action, np.eye(self.n_arm),
            np.repeat(shared_state[None, ...], self.n_arm, axis=0)
        ]).astype(np.float32)
        self._prev_kappa[...] = kappa_state
        return np.nan_to_num(observation_state.ravel())

    def set_action(self, action) -> None:
        # Action: (8, n_action)
        scale = 1.0  # min(time / 0.02, 1)

        action = np.reshape(action, [self.n_arm, self.n_action])

        # Continuous action
        for i in range(self.n_arm):
            # f2=interp1d(self.control_location, action[i,:3], kind='cubic')
            sucker_activation = action[i, :3]
            LM_activation = action[i, 3:6]
            TM_activation = action[i, 6:]

            for j in range(self.n_sucker):
                self.sucker_controller[i][j].reduction_ratio = sucker_activation[j]

            LM_activation = LM_activation - 0.5
            LM1_activation = np.max((LM_activation, [0.0] * 3), axis=0)
            LM2_activation = abs(np.min((LM_activation, [0.0] * 3), axis=0))
            apply_LM1 = interp1d(self.control_location, [0] + list(LM1_activation) + [0], kind='cubic')(
                range(self.n_elems - 1))
            apply_LM2 = interp1d(self.control_location, [0] + list(LM2_activation) + [0], kind='cubic')(
                range(self.n_elems - 1))
            apply_TM = interp1d(self.control_location, [0] + list(TM_activation) + [0], kind='cubic')(
                range(self.n_elems - 1))

            self.muscle_activations[i][0].set_activation(apply_LM1)
            self.muscle_activations[i][1].set_activation(apply_LM2)
            self.muscle_activations[i][2].set_activation(apply_TM)

        # update previous action
        self._prev_action = action

    def step(self, action):
        """ Set intrinsic strains (set actions) """
        self.set_action(action)

        """ Post-simulation """
        xposbefore = self.rigid_rod.position_collection[0:2, 0].copy()

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
            # Debug
            """
            invalid_values_condition = _isnan_check(np.concatenate(
                [rod.position_collection for rod in self.shearable_rods] + 
                [rod.velocity_collection for rod in self.shearable_rods]
                ))
            if invalid_values_condition == True:
                print(f" Nan detected in, exiting simulation now. {self.time=}")
                self.render()
                break
            else:
                self.render()
            """
        etime = time.perf_counter()
        states = self.get_state()

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        survive_reward = 0.0
        forward_reward = 0.0
        control_cost = 0.0  # 0.5 * np.square(action.ravel()).mean()
        bending_energy = 0.0  # sum([rod.compute_bending_energy() for rod in self.shearable_rods])
        shear_energy = 0.0  # sum([rod.compute_shear_energy() for rod in self.shearable_rods])
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(np.concatenate(
            [rod.position_collection for rod in self.shearable_rods] +
            [rod.velocity_collection for rod in self.shearable_rods]
        ))

        if invalid_values_condition == True:
            # print(f" Nan detected in, exiting simulation now. {self.time=}")
            done = True
            survive_reward = -5.0
        else:
            xposafter = self.rigid_rod.position_collection[0:2, 0]
            forward_reward = (np.linalg.norm(self._target - xposbefore) -
                              np.linalg.norm(self._target - xposafter)) * 1e2

            # forward_reward = self.compute_reward(
            #        self.rigid_rod.position_collection[:2,0],
            #        self._target,
            #        None)
            if np.linalg.norm(self._target - xposafter) < 0.2:
                survive_reward = 5
                done = True

        # print(self.rigid_rods.position_collection)
        # print(f'{self.counter=}, {etime-stime}sec, {self.time=}')
        if not done and self.time > self.final_time:
            forward_reward -= np.linalg.norm(self._target - xposafter)
            done = True

        reward = forward_reward - control_cost + survive_reward - bending_energy
        # reward *= 10 # Reward scaling
        # print(f'{reward=:.3f}, {forward_reward=:.3f}, {control_cost=:.3f}, {survive_reward=:.3f}, {bending_energy=:.3f}') #, {shear_energy=:.3f}')

        """ Return state:
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        # systems = [self.shearable_rod]

        # Info
        info = {'time': self.time, 'rods': self.shearable_rods, 'body': self.rigid_rod}
        if np.isnan(reward):
            reward = -5
            done = True
        reward = min(self.reward_range, reward)

        self.counter += 1

        return states, reward, done, info

    def compute_reward(
            self,
            achieved_goal: Union[int, np.ndarray],
            desired_goal: Union[int, np.ndarray],
            _info: Optional[Dict[str, Any]] = None,
    ) -> np.float32:
        eps = 0.01
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > eps).astype(np.float32)

    def render(self, mode='human', close=False) -> Optional[np.ndarray]:
        maxwidth = 800
        aspect_ratio = (3 / 4)

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
            assert issubclass(Session, BaseRenderer), \
                "Rendering module is not properly subclassed"
            assert issubclass(Session, BaseElasticaRendererSession), \
                "Rendering module is not properly subclassed"
            self.renderer = Session(width=maxwidth, height=int(maxwidth * aspect_ratio))
            self.renderer.add_rods(self.shearable_rods)
            self.renderer.add_rigid_body(self.rigid_rod)
            # self.renderer.add_point(self._target.tolist()+[0], 0.20)

        # POVRAY
        if RENDERER_CONFIG == RendererType.POVRAY:
            state_image = self.renderer.render(maxwidth, int(maxwidth * aspect_ratio * 0.7))
            state_image_side = self.renderer.render(
                maxwidth // 2,
                int(maxwidth * aspect_ratio * 0.3),
                camera_param=('location', [0.0, 0.0, -0.5], 'look_at', [0.0, 0, 0])
            )
            state_image_top = self.renderer.render(
                maxwidth // 2,
                int(maxwidth * aspect_ratio * 0.3),
                camera_param=('location', [0.0, 0.3, 0.0], 'look_at', [0.0, 0, 0])
            )

            state_image = np.vstack([
                state_image,
                np.hstack([state_image_side, state_image_top])
            ])
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
