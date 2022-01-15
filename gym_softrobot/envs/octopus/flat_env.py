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

from gym_softrobot.envs.octopus.build import build_octopus
from gym_softrobot.utils.custom_elastica.callback_func import RodCallBack, RigidCylinderCallBack
from gym_softrobot.utils.render.post_processing import plot_video


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class FlatEnv(core.Env):
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

    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self,
            final_time=5.0,
            time_step=5.0e-5,
            recording_fps=5,
            n_elems=10,
            n_arm=5,
            n_action=3,
            config_generate_video=False,
            config_save_head_data=False,
            policy_mode='centralized'
        ):

        # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time/self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))

        self.n_arm = n_arm
        self.n_elems = n_elems
        self.n_seg = n_elems-1
        self.policy_mode = policy_mode

        # Spaces
        self.n_action = n_action * 3 # number of interpolation point (3 for 3 curvatures)
        shared_space = 17
        if policy_mode == 'centralized':
            action_size = (n_arm*self.n_action,)
            action_low = np.ones(self.n_action) * (-22); action_low[self.n_action//3:] = -5
            action_high = np.ones(self.n_action) * (22); action_high[self.n_action//3:] =  5
            action_low = np.repeat(action_low, n_arm)
            action_high = np.repeat(action_high, n_arm)
            self.action_space = spaces.Box(action_low, action_high, shape=action_size, dtype=np.float32)
            self._observation_size = (n_arm, (self.n_seg + (self.n_elems+1) * 4 + self.n_action))
            self.observation_space = spaces.Dict({
                "individual": spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32),
                "shared": spaces.Box(-np.inf, np.inf, shape=(shared_space,), dtype=np.float32)
            })
        elif policy_mode == 'decentralized':
            action_size = (self.n_action,)
            action_low = np.ones(action_size) * (-22); action_low[self.n_action//3:] = -5
            action_high = np.ones(action_size) * (22); action_high[self.n_action//3:] =  5
            self.action_space = spaces.Box(action_low, action_high, shape=action_size, dtype=np.float32)
            self._observation_size = ((self.n_seg + (self.n_elems+1) * 4 + self.n_action + n_arm),)
            self.observation_space = spaces.Dict({
                "individual": spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32),
                "shared": spaces.Box(-np.inf, np.inf, shape=(shared_space,), dtype=np.float32)
            })
        else:
            raise NotImplementedError
        self.metadata= {}
        self.reward_range=50.0
        if policy_mode == 'centralized':
            self._prev_action = np.zeros(list(self.action_space.shape),
                    dtype=self.action_space.dtype)
        elif policy_mode == 'decentralized':
            self._prev_action = np.zeros([n_arm]+list(self.action_space.shape),
                    dtype=self.action_space.dtype)
        else:
            raise NotImplementedError

        # Configurations
        self.config_generate_video = config_generate_video
        self.config_save_head_data = config_save_head_data

        # Rendering-related
        self.viewer = None
        self.renderer = None

        # Determinism
        self.seed()

    def seed(self, seed=None):
        # Deprecated in new gym
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def summary(self,):
        print(f"""
        {self.final_time=}
        {self.time_step=}
        {self.total_steps=}
        {self.step_skip=}
        simulation time per action: {1.0/self.step_skip=}
        max number of action per episode: {self.total_steps/self.step_skip}

        {self.n_elems=}
        {self.action_space=}
        {self.observation_space=}
        {self.reward_range=}
        """)

    def reset(self):
        self.simulator = BaseSimulator()

        self.shearable_rods, self.rigid_rod = build_octopus(
                self.simulator,
                self.n_arm,
                self.n_elems,
            )

        # CallBack
        if self.config_generate_video:
            self.rod_parameters_dict_list=[]
            for arm in self.shearable_rods:
                rod_parameters_dict = defaultdict(list)
                self.simulator.collect_diagnostics(arm).using(
                    RodCallBack,
                    step_skip=self.step_skip,
                    callback_params=rod_parameters_dict
                )
                self.rod_parameters_dict_list.append(rod_parameters_dict)
        if self.config_save_head_data:
            self.head_dict = defaultdict(list)
            self.simulator.collect_diagnostics(self.rigid_rod).using(
                RigidCylinderCallBack, step_skip=self.step_skip, callback_params=self.head_dict
            )

        """ Finalize the simulator and create time stepper """
        self.StatefulStepper = PositionVerlet()
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # """ Return
        #     (1) total time steps for the simulation step iterations
        #     (2) systems for controller design
        # """
        # systems = [self.shearable_rod]
        self.rest_sigma=np.zeros_like(self.shearable_rods[0].sigma)
        self.time= np.float64(0.0)
        self.counter=0
        # self.bias=self.shearable_rod.compute_position_center_of_mass()[0].copy()

        # Set Target
        self._target = (2-1)*self.np_random.random(2) + 1
        #self._target /= np.linalg.norm(self._target) # I don't see why this is here

        # Initial State
        state = self.get_state()

        return state

    def get_state(self):
        states = {}
        # Build state
        kappa_state = np.vstack([rod.kappa[0] for rod in self.shearable_rods])
        pos_state1 = np.vstack([rod.position_collection[0] for rod in self.shearable_rods]) # x
        pos_state2 = np.vstack([rod.position_collection[1] for rod in self.shearable_rods]) # y
        vel_state1 = np.vstack([rod.velocity_collection[0] for rod in self.shearable_rods]) # x
        vel_state2 = np.vstack([rod.velocity_collection[1] for rod in self.shearable_rods]) # y
        if self.policy_mode == 'decentralized':
            previous_action = self._prev_action[:, :]
            individual_state = np.hstack([
                kappa_state, pos_state1, pos_state2, vel_state1, vel_state2,
                previous_action, np.eye(self.n_arm)]).astype(np.float32)
        elif self.policy_mode == 'centralized':
            previous_action = self._prev_action.reshape([self.n_arm, self.n_action])
            individual_state = np.hstack([
                kappa_state, pos_state1, pos_state2, vel_state1, vel_state2,
                previous_action]).astype(np.float32)
        else:
            raise NotImplementedError
        shared_state = np.concatenate([
            self._target, # 2
            self.rigid_rod.position_collection[:,0], # 3
            self.rigid_rod.velocity_collection[:,0], # 3
            self.rigid_rod.director_collection[:,:,0].ravel(), # 9
            ], dtype=np.float32)
        states["individual"] = individual_state
        states["shared"] = shared_state
        return states

    def set_action(self, action):
        reshaped_kappa = action.reshape((self.n_arm, self.n_action))
        if self.policy_mode == "decentralized":
            self._prev_action[:] = reshaped_kappa
        elif self.policy_mode == "centralized":
            self._prev_action[:] = reshaped_kappa.reshape([self.n_arm * self.n_action])
        else:
            raise NotImplementedError
        reshaped_kappa = reshaped_kappa.reshape((self.n_arm, 3, self.n_action // 3))
        reshaped_kappa = np.concatenate([
                np.zeros((self.n_arm, 3, 1)),
                reshaped_kappa,
                np.zeros((self.n_arm, 3, 1)),
            ], axis=-1)
        reshaped_kappa = interp1d(
                np.linspace(0,1,self.n_action//3+2), # added zero on the boundary
                reshaped_kappa,
                kind='cubic',
                axis=-1,
            )(np.linspace(0,1,self.n_seg))
        for arm_i in range(self.n_arm):
            self.shearable_rods[arm_i].rest_kappa[:, :] = reshaped_kappa[arm_i]
            #self.shearable_rods[arm_i].rest_sigma[1, :] = self.rest_sigma[1,:] #rest_sigma.copy()
            #self.shearable_rods[arm_i].rest_sigma[2, :] = self.rest_sigma[2,:] #rest_sigma.copy()

    def step(self, action):
        rest_kappa = action # alias

        """ Set intrinsic strains (set actions) """
        self.set_action(rest_kappa)

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

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        survive_reward = 0.0
        forward_reward = 0.0
        control_cost = 0.0 # 0.5 * np.square(rest_kappa.ravel()).mean()
        bending_energy = 0.0 #sum([rod.compute_bending_energy() for rod in self.shearable_rods])
        shear_energy = 0.0 # sum([rod.compute_shear_energy() for rod in self.shearable_rods])
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(np.concatenate(
            [rod.position_collection for rod in self.shearable_rods] + 
            [rod.velocity_collection for rod in self.shearable_rods]
            ))

        if invalid_values_condition == True:
            print(f" Nan detected in, exiting simulation now. {self.time=}")
            done = True
            survive_reward = -50.0
        else:
            xposafter = self.rigid_rod.position_collection[0:2,0]
            forward_reward = (np.linalg.norm(self._target - xposafter) - 
                np.linalg.norm(self._target - xposbefore)) * 1e3

        # print(self.rigid_rods.position_collection)
        #print(f'{self.counter=}, {etime-stime}sec, {self.time=}')
        if self.time>self.final_time:
            done=True

        reward = forward_reward - control_cost + survive_reward - bending_energy
        #reward *= 10 # Reward scaling
        #print(f'{reward=:.3f}, {forward_reward=:.3f}, {control_cost=:.3f}, {survive_reward=:.3f}, {bending_energy=:.3f}') #, {shear_energy=:.3f}')
            

        """ Return state:
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        # systems = [self.shearable_rod]
        states = self.get_state()

        # Info
        info = {'time':self.time}

        self.counter += 1

        return states, reward, done, info

    def save_data(self, filename_video, fps):
        
        if self.config_save_head_data:
            print("Saving data to pickle files ...", end='\r')
            np.savez(algo+"_head",**self.head_dict)
            print("Saving data to pickle files ... Done!")

        if self.config_generate_video:
            filename_video = f"save/{filename_video}"
            plot_video(self.rod_parameters_dict_list, filename_video, margin=0.2, fps=fps)

    def render(self, mode='human', close=False):
        maxwidth = 800

        if self.viewer is None:
            from gym_softrobot.utils.render import pyglet_rendering
            from gym_softrobot.utils.render.povray_rendering import Session
            self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)
            self.renderer = Session(width=maxwidth, height=maxwidth*3//4)
            self.renderer.add_rods(self.shearable_rods)
            self.renderer.add_rigid_body(self.rigid_rod)
            self.renderer.add_point(self._target.tolist()+[0], 0.05)

        state_image = self.renderer.render()

        self.viewer.imshow(state_image)

        return state_image

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
