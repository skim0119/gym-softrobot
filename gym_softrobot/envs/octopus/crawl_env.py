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

from gym_softrobot.envs.octopus.build_muscle_octopus import build_octopus_muscles

from gym_softrobot.config import RendererType
from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.utils.render.base_renderer import BaseRenderer, BaseElasticaRendererSession
from gym_softrobot.envs.octopus.controllable_constraint import ControllableFixConstraint


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class CrawlEnv(core.Env):
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

    metadata = {'render.modes': ['rgb_array'], 'multiagent': ['PyMARL']}

    def __init__(self,
            final_time=10.0,  # Final total time
            time_step=5.0e-5,
            recording_fps=25,
            n_elems=20,
            config_random_final_time=False,
        ):

        # Integrator type
        self.final_time = final_time
        self.time_step = time_step
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))

        n_arm = 8
        self.n_arm = n_arm 
        self.n_agent = n_arm  # for MARL
        self.n_elems = n_elems
        self.n_seg = n_elems-1

        # Spaces
        n_action = 3
        self.n_action = n_action
        self.action_space = spaces.Box(0.0, 1.0, shape=(n_arm* n_action,), dtype=np.float32)

        self.shared_space = 17
        self.grid_size = 1
        self._observation_size = (n_arm* (self.n_seg + (self.n_elems+1) * 4 + n_action + n_arm + self.shared_space),)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32)

        self.metadata= {}
        self.reward_range = 100.0
        self._prev_action = np.zeros(list(self.action_space.shape),
                dtype=self.action_space.dtype)

        # Configurations
        self.config_random_final_time = config_random_final_time

        # Rendering-related
        self.viewer = None
        self.renderer = None

        # Determinism
        self.seed()

    @property
    def agent_id(self):
        return ["LF1", "LF2", "LB2", "LB1", "RB1", "RB2", "RF2", "RF1"]

    def get_env_info(self):
        # Need for decentralized control
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
        if self.config_random_final_time:
            self.final_time = self.np_random.uniform(3.0, 10.0)
        self.simulator = BaseSimulator()

        self.shearable_rods, self.rigid_rod, self.tm_muscle_activations = build_octopus_muscles(
                self.simulator,
                self.n_elems,
            )

        """ Controller Setup """
        constraint_ids = []
        for i in range(self.n_arm):
            constraint_id = self.simulator.constrain(self.shearable_rods[i]).using(
                ControllableFixConstraint,
                index=0,
            ).id()
            constraint_ids.append(constraint_id)

        self.simulator.finalize()

        """ Sucker Controller Hook """
        self.sucker_controller = []
        for constraint_id in constraint_ids:
            controllable_constraint = dict(self.simulator._constraints)[constraint_id]
            controller = controllable_constraint.get_controller
            controller.turn_on()
            self.sucker_controller.append(controller)

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
        self.time= np.float64(0.0)
        self.counter=0
        # self.bias=self.shearable_rod.compute_position_center_of_mass()[0].copy()

        # Set Target
        #self._target = self.np_random.uniform(-self.grid_size, self.grid_size, size=2).astype(np.float32)
        self._target = np.array([5,0], dtype=np.float32)

        # Initial State
        state = self.get_state()

        if return_info:
            return state, {}
        else:
            return state

    def get_state(self):
        # Build state
        kappa_state = np.vstack([rod.kappa[0] for rod in self.shearable_rods])
        pos_state1 = np.vstack([rod.position_collection[0] for rod in self.shearable_rods]) # x
        pos_state2 = np.vstack([rod.position_collection[1] for rod in self.shearable_rods]) # y
        vel_state1 = np.vstack([rod.velocity_collection[0] for rod in self.shearable_rods]) # x
        vel_state2 = np.vstack([rod.velocity_collection[1] for rod in self.shearable_rods]) # y
        previous_action = self._prev_action.reshape([self.n_arm,self.n_action])
        shared_state = self.get_shared_state()
        observation_state = np.hstack([
            kappa_state, pos_state1, pos_state2, vel_state1, vel_state2,
            previous_action, np.eye(self.n_arm),
            np.repeat(shared_state[None,...], self.n_arm, axis=0)]).astype(np.float32)
        return np.nan_to_num(observation_state.ravel())

    def get_shared_state(self):
        shared_state = np.concatenate([
            self._target, # 2
            self.rigid_rod.position_collection[:,0], # 3
            self.rigid_rod.velocity_collection[:,0], # 3
            self.rigid_rod.director_collection[:,:,0].ravel(), # 9: orientation
            ], dtype=np.float32)
        return shared_state

    def set_action(self, action) -> None:
        # Action: (8, n_action)
        scale = 1.0  # min(time / 0.02, 1)

        action = np.reshape(action, [self.n_arm, self.n_action])

        # Continuous action
        for i in range(self.n_arm):
            location = action[i,0]
            activation = action[i,1]
            r_ratio = action[i,2]
            self.sucker_controller[i].index = int(np.clip(location * self.n_elems, 0, self.n_elems-1))
            self.tm_muscle_activations[i][2].apply_activation(activation) # 2 for TM
            self.sucker_controller[i].reduction_ratio = r_ratio

        # update previous action
        self._prev_action = action

    def step(self, action):
        """ Set intrinsic strains (set actions) """
        self.set_action(action)

        """ Post-simulation """
        xposbefore = self.rigid_rod.position_collection[0:2,0].copy() 

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
        states = self.get_state()

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        survive_reward = 0.0
        forward_reward = 0.0
        control_cost = 0.0 # 0.5 * np.square(action.ravel()).mean()
        bending_energy = 0.0 #sum([rod.compute_bending_energy() for rod in self.shearable_rods])
        shear_energy = 0.0 # sum([rod.compute_shear_energy() for rod in self.shearable_rods])
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(np.concatenate(
            [rod.position_collection for rod in self.shearable_rods] + 
            [rod.velocity_collection for rod in self.shearable_rods]
            ))

        if invalid_values_condition == True:
            #print(f" Nan detected in, exiting simulation now. {self.time=}")
            done = True
            survive_reward = -5.0
        else:
            xposafter = self.rigid_rod.position_collection[0:2,0]
            forward_reward = (np.linalg.norm(self._target - xposbefore) - 
                np.linalg.norm(self._target - xposafter)) * 1e2

            if np.linalg.norm(self._target - xposafter) < 0.2:
                survive_reward = 5
                done = True

        # print(self.rigid_rods.position_collection)
        #print(f'{self.counter=}, {etime-stime}sec, {self.time=}')
        if not done and self.time>self.final_time:
            #forward_reward -= np.linalg.norm(self._target - xposafter) * 1.0
            done=True

        reward = forward_reward - control_cost + survive_reward - bending_energy
        #reward *= 10 # Reward scaling
        #print(f'{reward=:.3f}, {forward_reward=:.3f}, {control_cost=:.3f}, {survive_reward=:.3f}, {bending_energy=:.3f}') #, {shear_energy=:.3f}')
            
        # Info
        info = {'time':self.time, 'rods':self.shearable_rods, 'body':self.rigid_rod}
        if np.isnan(reward):
            reward -= 5
            done = True
        reward = min(self.reward_range, reward)

        self.counter += 1

        return states, reward, done, info


    def render(self, mode='human', close=False) -> Optional[np.ndarray]:
        maxwidth = 800
        aspect_ratio = (3/4)

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
            self.renderer = Session(width=maxwidth, height=int(maxwidth*aspect_ratio))
            self.renderer.add_rods(self.shearable_rods)
            self.renderer.add_rigid_body(self.rigid_rod)
            #self.renderer.add_point(self._target.tolist()+[0], 0.20)

        # POVRAY
        if RENDERER_CONFIG == RendererType.POVRAY:
            state_image = self.renderer.render(maxwidth, int(maxwidth*aspect_ratio*0.7))
            state_image_side = self.renderer.render(
                    maxwidth//2,
                    int(maxwidth*aspect_ratio*0.3),
                    camera_param=('location',[0.0, 0.0, -0.5],'look_at',[0.0,0,0])
                )
            state_image_top = self.renderer.render(
                    maxwidth//2,
                    int(maxwidth*aspect_ratio*0.3),
                    camera_param=('location',[0.0, 0.3, 0.0],'look_at',[0.0,0,0])
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
