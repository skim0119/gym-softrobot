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
from gym_softrobot.utils.intersection import intersection
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
            time_step=7.0e-5,
            recording_fps=5,
            n_elems=10,
            n_arm=8,
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
        self.n_action = n_action  # number of interpolation point (3 curvatures)
        shared_space = 13
        if policy_mode == 'centralized':
            action_size = (n_arm*self.n_action,)
            action_low = np.ones(self.n_action) * (-22)
            action_high = np.ones(self.n_action) * (22)
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
            action_low = np.ones(action_size) * (-22)
            action_high = np.ones(action_size) * (22)
            self.action_space = spaces.Box(action_low, action_high, shape=action_size, dtype=np.float32)
            self._observation_size = ((self.n_seg + (self.n_elems+1) * 4 + self.n_action + n_arm),)
            self.observation_space = spaces.Dict({
                "individual": spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32),
                "shared": spaces.Box(-np.inf, np.inf, shape=(shared_space,), dtype=np.float32)
            })
        else:
            raise NotImplementedError
        self.metadata= {}
        self.reward_range=100.0
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
        self._target = (2-0.5)*self.np_random.random(2) + 0.5
        #self._target /= np.linalg.norm(self._target) # I don't see why this is here

        # Initial State
        state = self.get_state()

        return state

    def get_state(self):
        states = {}
        cx, cy, _ = self.rigid_rod.position_collection[:,0]
        # Build state
        kappa_state = np.vstack([rod.kappa[0] for rod in self.shearable_rods])
        pos_state1 = np.vstack([rod.position_collection[0]-cx for rod in self.shearable_rods]) # x
        pos_state2 = np.vstack([rod.position_collection[1]-cy for rod in self.shearable_rods]) # y
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
            self._target-self.rigid_rod.position_collection[:2,0], # 2
            #self.rigid_rod.position_collection[:,0], # 3
            self.rigid_rod.velocity_collection[:2,0], # 2
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
        reshaped_kappa = np.concatenate([
                np.zeros((self.n_arm, 1)),
                reshaped_kappa,
                np.zeros((self.n_arm, 1)),
            ], axis=-1)
        reshaped_kappa = interp1d(
                np.linspace(0,1,self.n_action+2), # added zero on the boundary
                reshaped_kappa,
                kind='cubic',
                axis=-1,
            )(np.linspace(0,1,self.n_seg))
        for arm_i in range(self.n_arm):
            self.shearable_rods[arm_i].rest_kappa[0, :] = reshaped_kappa[arm_i]
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
        control_panelty = 0.0 # 0.005 * np.square(rest_kappa.ravel()).mean()
        bending_energy = 0.0 #np.mean([rod.compute_bending_energy().mean() for rod in self.shearable_rods]) * 0.001
        shear_energy = 0.0 # sum([rod.compute_shear_energy() for rod in self.shearable_rods])
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(np.concatenate(
            [rod.position_collection for rod in self.shearable_rods] + 
            [rod.velocity_collection for rod in self.shearable_rods]
            ))
        arm_crossing = sum([len(intersection(self.shearable_rods[i-1].position_collection[:2,:],
            self.shearable_rods[i].position_collection[:2,:])[0])
            for i in range(self.n_arm-1)])

        xposafter = self.rigid_rod.position_collection[0:2,0]
        to_target = self._target - xposafter
        dist_to_target = np.linalg.norm(to_target)
        if invalid_values_condition == True:
            print(f" Nan detected in, exiting simulation now. {self.time=}")
            done = True
            survive_reward = -50.0
        else:
            survive_reward = -0.02 * arm_crossing
            forward_reward = (dist_to_target - np.linalg.norm(self._target - xposbefore)) / (self.step_skip * self.time_step)
            #forward_reward = np.dot(self.rigid_rod.velocity_collection[:2,0], to_target / dist_to_target)
            """ touched """
            if dist_to_target < 0.1:
                survive_reward = 100.0
                done = True

        #print(f'{self.counter=}, {etime-stime}sec, {self.time=}')
        timelimit = False
        if self.time>self.final_time:
            timelimit = True
            done=True

        reward = forward_reward - control_panelty + survive_reward - bending_energy
        #reward *= 10 # Reward scaling
        #print(f'  {reward=:.3f}: {forward_reward=:.8f}')
        #print(f'                 {control_panelty=:.3f}, {survive_reward=:.3f}')
        #print(f'                 {bending_energy=:.3f}, {shear_energy=:.3f}')
        #print(f'                 {dist_to_target-0.1=:.3f}')
        if done:
            reward -= (dist_to_target-0.1)
            

        """ Return state:
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        # systems = [self.shearable_rod]
        states = self.get_state()

        # Info
        info = {'time':self.time, 'rods':self.shearable_rods, 'body':self.rigid_rod,
                'TimeLimit.truncated': timelimit}

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
        aspect_ratio = (3/4)

        if self.viewer is None:
            from gym_softrobot.utils.render import pyglet_rendering
            from gym_softrobot.utils.render.povray_rendering import Session
            self.viewer = pyglet_rendering.SimpleImageViewer(maxwidth=maxwidth)
            self.renderer = Session(width=maxwidth, height=int(maxwidth*aspect_ratio))
            self.renderer.add_rods(self.shearable_rods)
            self.renderer.add_rigid_body(self.rigid_rod)
            self.renderer.add_point(self._target.tolist()+[0], 0.05)

        # Temporary rendering to add side-view
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

        self.viewer.imshow(state_image)

        return state_image

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
