import gym
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

from gym_softrobot.utils.actuation.forces.drag_force import DragForce
from gym_softrobot.utils.custom_elastica.callback_func import RodCallBack, RigidCylinderCallBack
from gym_softrobot.utils.custom_elastica.joint import FixedJoint2Rigid
from gym_softrobot.utils.render.post_processing import plot_video

def z_rotation(vector, theta):
    theta = theta / 180.0 * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta),0.0],[np.sin(theta), np.cos(theta),0],[0.0,0.0,1.0]])
    return np.dot(R,vector.T).T


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class FlatEnv(gym.Env):
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
            final_time,
            time_step=1.0e-5,
            recording_fps=40,
            n_elems=40,
            n_body=1,
            n_arm=5,
            config_generate_video=False,
            config_save_head_data=False,
            policy_mode=None
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
        self.n_body = n_body
        self.policy_mode = policy_mode

        # Spaces
        self.n_action = 3 # number of interpolation point
        shared_space = 17
        if policy_mode == 'centralized':
            self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.n_arm*self.n_action,), dtype=np.float32)
            self._observation_size = (self.n_arm, (self.n_seg + (self.n_elems+1) * 4 + self.n_action))
            self.observation_space = spaces.Dict({
                "individual": spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32),
                "shared": spaces.Box(-np.inf, np.inf, shape=(shared_space,), dtype=np.float32)
            })
        elif policy_mode == 'decentralized':
            self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.n_action,), dtype=np.float32)
            self._observation_size = ((self.n_seg + (self.n_elems+1) * 4 + self.n_action),)
            self.observation_space = spaces.Dict({
                "individual": spaces.Box(-np.inf, np.inf, shape=self._observation_size, dtype=np.float32),
                "shared": spaces.Box(-np.inf, np.inf, shape=(shared_space,), dtype=np.float32)
            })
        else:
            raise NotImplementedError
        self.metadata= {}
        self.reward_range=50.0
        self._prev_action = np.zeros([n_body, n_arm]+list(self.action_space.shape),
                dtype=self.action_space.dtype)

        self.config_generate_video = config_generate_video
        self.config_save_head_data = config_save_head_data

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

    def reset(self,):
        #print("resetting")
        self.simulator = BaseSimulator()

        """ Set up an arm """
        n_elem = self.n_elems            # number of discretized elements of the arm
        L0 = 0.35                # total length of the arm
        r0 = L0 * 0.011

        rigid_rod_length = r0 * 2
        rigid_rod_radius = 0.005 * 10

        # radius_base = r0     # radius of the arm at the base
        # radius_tip = r0     # radius of the arm at the tip
        # radius = np.linspace(radius_base, radius_tip, n_elem+1)
        # radius_mean = (radius[:-1]+radius[1:])/2
        # damp_coefficient = 0.02

        rotation_angle=360/self.n_arm
        self.angle_list=[rotation_angle*arm_i for arm_i in range(self.n_arm)]

        self.shearable_rods=[]  # arms
        self.rigid_rods=[]  # heads
        for body_i in range(self.n_body):
            for arm_i in range(self.n_arm):
                rod = CosseratRod.straight_rod(
                    n_elements=n_elem,
                    start=z_rotation(np.array([rigid_rod_radius, 0.0, 0.0]),self.angle_list[arm_i]),
                    direction=z_rotation(np.array([1.0, 0.0, 0.0]),self.angle_list[arm_i]),
                    normal=np.array([0.0, 0.0, 1.0]),
                    base_length=L0,
                    base_radius=r0,
                    density=1000.0,
                    nu=1e-3,
                    youngs_modulus=1e6,
                    poisson_ratio=0.5,
                    # nu_for_torques=damp_coefficient*((radius_mean/radius_base)**4),
                )
                self.shearable_rods.append(rod)
                self.simulator.append(rod)

            """ Add head """
            # setting up test params
            start = np.zeros((3,))
            direction = np.array([0.0, 0.0, 1.0])
            normal = np.array([0.0, 1.0, 0.0])
            binormal = np.cross(direction, normal)
            base_area = np.pi * rigid_rod_radius ** 2
            density = 1000

            rigid_rod = Cylinder(start, direction, normal, rigid_rod_length, rigid_rod_radius, density)
            self.simulator.append(rigid_rod)
            self.rigid_rods.append(rigid_rod)

            """ Set up boundary conditions """
            for arm_i in range(self.n_arm):
                _k = 1e2
                _kt = _k / 100
                a = self.n_arm * body_i + arm_i
                self.simulator.connect(
                    first_rod=rigid_rod, second_rod=self.shearable_rods[a], first_connect_idx=-1, second_connect_idx=0
                ).using(FixedJoint2Rigid, k=_k, nu=1e-3, kt=_kt,angle=self.angle_list[arm_i],radius=rigid_rod_radius)

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
            self.head_dicts = []
            for rr in self.rigid_rods:
                head_dict = defaultdict(list)
                self.simulator.collect_diagnostics(rr).using(
                    RigidCylinderCallBack, step_skip=self.step_skip, callback_params=head_dict
                )
                self.head_dicts.append(head_dict)
        """Add gravity forces"""
        gravitational_acc = -9.81
        for body_i in range(self.n_body):
            for arm_i in range(self.n_arm):
                a = self.n_arm * body_i + arm_i
                self.simulator.add_forcing_to(self.shearable_rods[a]).using(
                        GravityForces, acc_gravity=np.array([0.0, 0.0,gravitational_acc])
                    )
            self.simulator.add_forcing_to(self.rigid_rods[body_i]).using(
                        GravityForces, acc_gravity=np.array([0.0, 0.0,gravitational_acc])
                    )

        """ Add drag force """
        # dl = L0 / n_elem
        # fluid_factor = 1
        # r_bar = (radius_base + radius_tip) / 2
        # sea_water_dentsity = 1022
        # c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
        # c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor
        #
        # self.simulator.add_forcing_to(self.shearable_rod).using(
        #     DragForce,
        #     rho_environment=sea_water_dentsity,
        #     c_per=c_per,
        #     c_tan=c_tan,
        #     system=self.shearable_rod,
        #     step_skip=self.step_skip,
        #     callback_params=self.rod_parameters_dict
        # )

        """Add friction forces (always the last thing before finalize)"""
        normal = np.array([0.0, 0.0, 1.0])
        base_length = L0
        base_radius = r0
        period = 2.0

        origin_plane = np.array([0.0, 0.0, -base_radius])
        normal_plane = normal
        slip_velocity_tol = 1e-8
        froude = 0.1
        mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
        kinetic_mu_array = np.array(
            [mu, 1.5 * mu, 2.0 * mu]
        )  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        for body_i in range(self.n_body):
            for arm_i in range(self.n_arm):
                a = self.n_arm * body_i + arm_i
                self.simulator.add_forcing_to(self.shearable_rods[a]).using(
                    AnisotropicFrictionalPlane,
                    k=1.0,
                    nu=1e-6,
                    plane_origin=origin_plane,
                    plane_normal=normal_plane,
                    slip_velocity_tol=slip_velocity_tol,
                    static_mu_array=static_mu_array,
                    kinetic_mu_array=kinetic_mu_array,
                )
        mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
        kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
        static_mu_array = 2 * kinetic_mu_array
        for body_i in range(self.n_body):
            self.simulator.add_forcing_to(self.rigid_rods[body_i]).using(
                AnisotropicFrictionalPlaneRigidBody,
                k=1.0,
                nu=1e0,
                plane_origin=origin_plane,
                plane_normal=normal_plane,
                slip_velocity_tol=slip_velocity_tol,
                static_mu_array=static_mu_array,
                kinetic_mu_array=kinetic_mu_array,
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
        self._target = (2-1)*np.random.random(2) + 1
        self._target /= np.linalg.norm(self._target)

        # Initial State
        state = self.get_state()

        #
        self._pre_done = np.zeros(self.n_body, dtype=bool)

        return state

    def get_state(self, dones=None):
        states = defaultdict(list)
        # Build state
        for i in range(self.n_body):
            a = self.n_arm * i
            b = self.n_arm * (i+1)
            if dones is None or (not dones[i]):
                kappa_state = np.vstack([rod.kappa[0] for rod in self.shearable_rods[a:b]])
                pos_state1 = np.vstack([rod.position_collection[0] for rod in self.shearable_rods[a:b]]) # x
                pos_state2 = np.vstack([rod.position_collection[1] for rod in self.shearable_rods[a:b]]) # y
                vel_state1 = np.vstack([rod.velocity_collection[0] for rod in self.shearable_rods[a:b]]) # x
                vel_state2 = np.vstack([rod.velocity_collection[1] for rod in self.shearable_rods[a:b]]) # y
                previous_action = self._prev_action[i, :, :]
                individual_state = np.hstack([
                    kappa_state, pos_state1, pos_state2, vel_state1, vel_state2,
                    previous_action])
                shared_state = np.concatenate([
                    self._target, # 2
                    self.rigid_rods[i].position_collection[:,0], # 3
                    self.rigid_rods[i].velocity_collection[:,0], # 3
                    self.rigid_rods[i].director_collection[:,:,0].ravel(), # 9
                    ])
                states["individual"].append(individual_state)
                states["shared"].append(shared_state)
            else:
                for k, v in self.observation_space.spaces.items():
                    shape = v.shape
                    if self.policy_mode == "decentralized" and k != 'shared':
                        shape = tuple([self.n_arm]+list(shape))
                    states[k].append(np.zeros(shape, dtype=v.dtype))
        # Wrap numpy array
        for k in states:
            states[k] = np.array(states[k])
        return states

    def set_action(self, action):
        for body_i in range(self.n_body):
            reshaped_kappa  = action[body_i].reshape((self.n_arm, self.n_action))
            self._prev_action[body_i][:] = reshaped_kappa
            reshaped_kappa = np.concatenate([np.zeros((self.n_arm, 1)), reshaped_kappa], axis=1)
            reshaped_kappa = interp1d(
                    np.linspace(0,1,self.n_action+1),
                    reshaped_kappa,
                    kind='cubic'
                )(np.linspace(0,1,self.n_seg))
            for arm_i in range(self.n_arm):
                a = self.n_arm * body_i + arm_i
                self.shearable_rods[a].rest_kappa[0, :] = reshaped_kappa[arm_i]
                #self.shearable_rods[a].rest_sigma[1, :] = self.rest_sigma[1,:] #rest_sigma.copy()
                #self.shearable_rods[a].rest_sigma[2, :] = self.rest_sigma[2,:] #rest_sigma.copy()

    def step(self, action):
        rest_kappa = action # alias

        """ Set intrinsic strains (set actions) """
        self.set_action(rest_kappa)

        """ Post-simulation """
        xposbefore_list = [rb.position_collection[0:2,0].copy() for rb in self. rigid_rods]

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
        rewards = []
        dones = []
        for body_i in range(self.n_body):
            if self._pre_done[body_i]:
                rewards.append(0.0)
                dones.append(True)
                continue
            a = self.n_arm * body_i
            b = self.n_arm * (body_i+1)
            done = False
            survive_reward = 0.0
            forward_reward = 0.0
            control_cost = 0.0 # 0.5 * np.square(rest_kappa[body_i].ravel()).mean()
            bending_energy = sum([rod.compute_bending_energy() for rod in self.shearable_rods[a:b]])
            if np.isnan(bending_energy):
                bending_energy = 10
            #shear_energy = sum([rod.compute_shear_energy() for rod in self.shearable_rods[a:b]])
            # Position of the rod cannot be NaN, it is not valid, stop the simulation
            invalid_values_condition = _isnan_check(np.concatenate(
                [rod.position_collection for rod in self.shearable_rods[a:b]] + 
                [rod.velocity_collection for rod in self.shearable_rods[a:b]]
                ))

            if invalid_values_condition == True:
                print(f" Nan detected in {body_i}-body simulation, exiting simulation now. {self.time=}")
                done = True
                survive_reward = -10.0
            else:
                xposafter = self.rigid_rods[body_i].position_collection[0:2,0]
                forward_reward = (np.linalg.norm(self._target - xposafter) - 
                    np.linalg.norm(self._target - xposbefore_list[body_i])) * 1e3

            # print(self.rigid_rods[body_i].position_collection)
            #print(f'{self.counter=}, {etime-stime}sec, {self.time=}')
            if self.counter>=250 or self.time>self.final_time:
                done=True

            reward = forward_reward - control_cost + survive_reward - bending_energy
            reward = forward_reward
            #reward *= 10 # Reward scaling
            #print(f'{reward=:.3f}, {forward_reward=:.3f}, {control_cost=:.3f}, {survive_reward=:.3f}, {bending_energy=:.3f}') #, {shear_energy=:.3f}')
            
            rewards.append(reward)
            dones.append(done)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        """ Return state:
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        # systems = [self.shearable_rod]
        states = self.get_state(dones)

        # Info
        info = {'time':self.time,
                'done':dones,}
        self._pre_done[:] = dones

        self.counter += 1

        return states, rewards, np.all(dones), info

    def save_data(self, filename_video, fps):
        
        if self.config_save_head_data:
            print("Saving data to pickle files ...", end='\r')
            np.savez(algo+"_head",**self.head_dict)
            print("Saving data to pickle files ... Done!")

        if self.config_generate_video:
            filename_video = f"save/{filename_video}"
            plot_video(self.rod_parameters_dict_list, filename_video, margin=0.2, fps=fps)

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
