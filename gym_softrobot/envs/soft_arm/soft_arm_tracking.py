from collections import defaultdict
import time
import copy

import numpy as np
from scipy.interpolate import interp1d

from typing import Optional
import gym
from gym import core
from gym import error, spaces, utils
from gym.utils import seeding

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *
from elastica.external_forces import GravityForces

from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.config import RendererType
from gym_softrobot.envs.octopus.build import build_arm
from gym_softrobot.utils.custom_elastica.callback_func import (
    RodCallBack,
    RigidCylinderCallBack,
)
from gym_softrobot.utils.render.post_processing import plot_video
from gym_softrobot.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)

from gym_softrobot.utils.custom_elastica.muscle_torque import (
    MuscleTorquesWithVaryingBetaSplines,
)


# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


# TODO: generalize this as a Class for online trajectory generation.
# TODO: remove np.random: use fixed generator from seed value. (determinism)
def generate_trajectory(final_time, sim_dt, target_v_scale):
    end_time = (
        final_time * 1.1
    )  # Use of 1.1 just adds a little buffer on the end of the trajectory.
    numpoints = np.rint(1 / sim_dt * end_time).astype(int)
    target_trajectory = np.zeros((numpoints, 3))

    """ Generate a trajectory bounded in a box [-0.8, 0.8] x [-0.8, 0.8] x [0.0, 0.8] """
    t = np.array([end_time * float(i) / (numpoints - 1) for i in range(numpoints)])
    t += (
        np.random.rand(1)[0] * 3600
    )  # start the trajectory anywhere within a 1 hour period

    f1 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    f2 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    f3 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    direction = np.random.randint(0, 1) * 2 - 1
    target_trajectory[:, 0] = (
        direction
        * 0.8
        * np.sin(2 * np.pi * f1 * t)
        * np.sin(2 * np.pi * f2 * t)
        * np.sin(2 * np.pi * f3 * t)
        * 1000
    )

    f1 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    f2 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    f3 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    direction = np.random.randint(0, 1) * 2 - 1
    target_trajectory[:, 1] = (
        direction
        * 0.4
        * np.sin(2 * np.pi * f1 * t)
        * np.sin(2 * np.pi * f2 * t)
        * np.sin(2 * np.pi * f3 * t)
        * 1000
    )
    target_trajectory[:, 1] += 0.4

    f1 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    f2 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    f3 = np.random.uniform(2, 5) * 0.025 * target_v_scale
    direction = np.random.randint(0, 1) * 2 - 1
    target_trajectory[:, 2] = (
        direction
        * 0.8
        * np.sin(2 * np.pi * f1 * t)
        * np.sin(2 * np.pi * f2 * t)
        * np.sin(2 * np.pi * f3 * t)
        * 1000
    )

    return target_trajectory


class SoftArmTrackingEnv(core.Env):
    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(self, game_mode:int=1):
        self.n_elem = 40
        self.sim_dt = 2.0e-4
        self.RL_update_interval = 0.01  # This is 100 updates per second
        self.num_steps_per_update = np.rint(
            self.RL_update_interval / self.sim_dt
        ).astype(int)
        self.youngs_modulus = 2e6
        poisson_ratio = 0.45
        self.shear_modulus = 0.5 * self.youngs_modulus / (poisson_ratio + 1.0)

        self.torque_scale = 10

        self.max_episode_final_time = 5  # seconds

        self.base_length = 1000
        self.radius = 50
        self.COLLECT_DATA_FOR_POSTPROCESSING = False

        self.number_of_control_points = 4
        self.number_of_observation_segments = self.number_of_control_points

        self.rendering_fps = 30
        self.step_skip = np.rint(1.0 / (self.rendering_fps * self.sim_dt)).astype(int)

        self.max_rate_of_change_of_activation = np.infty
        self.target_v_scale = 0.1

        self.mode = game_mode
        self.target_location = np.array([500, 500.0, 500])

        self.StatefulStepper = PositionVerlet()

        # normal and/or binormal direction activation (3D)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.number_of_control_points,),
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.number_of_observation_segments * 2 + 6,),
            dtype=np.float64,
        )

        # Determinism
        self.seed()

        # Rendering-related
        self.viewer = None
        self.renderer = None

    def seed(self, seed=None):
        # Deprecated in new gym
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """
        avg_kappa_1 = np.zeros(self.number_of_observation_segments)
        # self.n_elem-1 is number of curvature values due to veroni regions
        avg_length = int((self.n_elem - 1) / (self.number_of_observation_segments))

        for i in range(0, self.number_of_observation_segments - 1):
            lower = np.rint(avg_length * (i)).astype(int)
            upper = np.rint(avg_length * (i + 1)).astype(int)
            avg_kappa_1[i] = self.shearable_rod.kappa[0, lower:upper].mean()
        lower = np.rint(avg_length * (self.number_of_observation_segments - 1)).astype(
            int
        )
        avg_kappa_1[-1] = self.shearable_rod.kappa[0, lower:].mean()

        avg_kappa_2 = np.zeros(self.number_of_observation_segments)
        for i in range(0, self.number_of_observation_segments - 1):
            lower = np.rint(avg_length * (i)).astype(int)
            upper = np.rint(avg_length * (i + 1)).astype(int)
            avg_kappa_2[i] = self.shearable_rod.kappa[1, lower:upper].mean()
        lower = np.rint(avg_length * (self.number_of_observation_segments - 1)).astype(
            int
        )
        avg_kappa_2[-1] = self.shearable_rod.kappa[1, lower:].mean()

        state = np.concatenate(
            (
                # rod curvature information
                avg_kappa_1
                * self.base_length
                / (
                    2 * np.pi
                ),  # normalize by the curvature of if the rod is a perfect ring.
                avg_kappa_2
                * self.base_length
                / (
                    2 * np.pi
                ),  # normalize by the curvature of if the rod is a perfect ring.
                # arm tip location
                self.shearable_rod.position_collection[..., -1] / self.base_length,
                # target location
                self.wsol[self.tick] / 1000,  # convert back to meters (~[-1,1])
            )
        )

        return state

    def step(self, action):
        # set binormal activations to 0 if solving 2D case
        self.spline_points_func_array_normal_dir[:] = action[
            : self.number_of_control_points
        ]
        self.spline_points_func_array_binormal_dir[:] = action[
            self.number_of_control_points :
        ]

        for _ in range(int(self.num_steps_per_update)):
            # print('step:', i)
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.sim_dt,
            )
            self.tick += 1
            self.sphere.position_collection[..., 0] = self.wsol[self.tick]
            self.sphere.velocity_collection[..., 0] = self.velocity_sphere[self.tick]

        """ Reward Engineering """
        tip_to_target = (
            self.sphere.position_collection[..., 0]
            - self.shearable_rod.position_collection[..., -1]
        ) / 1000
        reward_dist = -np.square(np.linalg.norm(tip_to_target))
        reward = 1.0 * reward_dist  # + reward_orientation

        # observe current state: current as sensed signal
        state = self.get_state()

        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(state)
        if invalid_values_condition == True:
            reward = -100
            state = np.nan_to_num(self.get_state())
            done = True
            print("Episode blew up. Maybe try a smaller dt?")

        if self.tick * self.sim_dt >= self.max_episode_final_time:
            done = True
            print("Episode has reached max time")

        self._target = self.wsol[self.tick]
        return state, reward, done, {"ctime": self.time_tracker}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.simulator = BaseSimulator()

        ###--------------ADD ARM TO SIMULATION--------------###
        start = np.zeros((3,))
        direction = np.array([0.0, 1.0, 0.0])  # rod direction: pointing upwards
        normal = np.array([0.0, 0.0, 1.0])
        binormal = np.cross(direction, normal)

        # Set the arm properties after defining rods

        damping = self.youngs_modulus * 1e-7 * 1

        radius_tip = self.radius  # radius of the arm at the tip
        radius_base = radius_tip  # radius of the arm at the base
        radius_along_rod = np.linspace(radius_tip, radius_tip, self.n_elem)

        # Arm is shearable Cosserat rod
        self.shearable_rod = CosseratRod.straight_rod(
            self.n_elem,
            start,
            direction,
            normal,
            self.base_length,
            base_radius=radius_along_rod,
            density=1000 * 1e-6,
            nu=damping,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )

        self.shearable_rod.dissipation_constant_for_torques *= (
            1e6  # accounts for the new g/mm/s units (compared to kg/m/s)
        )
        self.simulator.append(
            self.shearable_rod
        )  # Now rod is ready for simulation, append rod to simulation
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )
        # self.simulator.add_forcing_to(self.shearable_rod).using(GravityForces, acc_gravity=np.array([0.0, 9.81*1e3, 0.0]))

        # Call back function to collect arm data from simulation
        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            class ArmCallBack(CallBackBaseClass):
                """
                Call back function for Elastica rod
                """

                def __init__(
                    self,
                    step_skip: int,
                    callback_params: dict,
                ):
                    CallBackBaseClass.__init__(self)
                    self.every = step_skip
                    self.callback_params = callback_params

                def make_callback(self, system, time, current_step: int):
                    if current_step % self.every == 0:
                        self.callback_params["time"].append(time)
                        self.callback_params["step"].append(current_step)
                        self.callback_params["position"].append(
                            system.position_collection.copy()
                        )
                        self.callback_params["radius"].append(system.radius.copy())
                        self.callback_params["com"].append(
                            system.compute_position_center_of_mass()
                        )
                        self.callback_params["directors"].append(
                            system.director_collection.copy()
                        )
                        self.callback_params["kappa"].append(system.kappa.copy())
                        self.callback_params["omega_collection"].append(
                            system.omega_collection.copy()
                        )
                        self.callback_params["sigma"].append(system.sigma.copy())
                        self.callback_params["tangents"].append(system.tangents.copy())
                        self.callback_params["velocity_collection"].append(
                            system.velocity_collection.copy()
                        )
                        self.callback_params["acceleration_collection"].append(
                            system.acceleration_collection.copy()
                        )
                        return

            # Collect data using callback function for postprocessing
            self.post_processing_dict_rod = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

        """
        heuristic scaling of torque - not this works for a 25:1 slenderness ratio. Bending stiffness scales ~r^4 so if thicker or thinner, 
        the torque value might need to be adjusted. This scaling is more designed for changing the youngs modulus 
        """
        self.alpha = self.torque_scale * self.radius * self.youngs_modulus
        # Add muscle torques acting on the arm for actuation
        # MuscleTorquesWithVaryingBetaSplines uses the control points selected by RL to
        # generate torques along the arm.
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            # max_signal_rate_of_change=4*self.sim_dt,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = []
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=self.base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            # max_signal_rate_of_change=4*self.sim_dt,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        ###--------------GENERATE TARGET TRAJECTORY--------------###
        self.tick = 0


        if self.mode == 2:
            # TODO: change this to a object that will update thee target position as you go
            target_trajectory = generate_trajectory(
                self.max_episode_final_time, self.sim_dt, self.target_v_scale
            )
            self.wsol = target_trajectory
        elif self.mode == 1:
            end_time = self.max_episode_final_time * 1.1
            numpoints = np.rint(1 / self.sim_dt * end_time).astype(int)
            self.wsol = np.zeros((numpoints, 3))
            self.wsol[:] = self.target_location

        velocity_sphere = self.wsol * 0
        velocity_sphere[:-1] = (self.wsol[1:] - self.wsol[:-1]) / self.sim_dt
        velocity_sphere[-1] = velocity_sphere[-2]
        self.velocity_sphere = velocity_sphere

        ###--------------ADD TARGET TO SIMULATION--------------###
        self.sphere_radius = 0.025 * 1000
        target_position = self.wsol[0]
        # initialize sphere
        self.sphere = Sphere(
            center=target_position,  # initialize target position of the ball
            base_radius=self.sphere_radius,
            density=1000 * 1e-6,
        )
        self.simulator.append(self.sphere)
        self.sphere.velocity_collection[..., 0] = self.velocity_sphere[0]

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            # Call back function to collect target sphere data from simulation
            class RigidSphereCallBack(CallBackBaseClass):
                """
                Call back function for target sphere
                """

                def __init__(self, step_skip: int, callback_params: dict):
                    CallBackBaseClass.__init__(self)
                    self.every = step_skip
                    self.callback_params = callback_params

                def make_callback(self, system, time, current_step: int):
                    if current_step % self.every == 0:
                        self.callback_params["time"].append(time)
                        self.callback_params["step"].append(current_step)
                        self.callback_params["position"].append(
                            system.position_collection.copy()
                        )
                        self.callback_params["velocity"].append(
                            system.velocity_collection.copy()
                        )
                        self.callback_params["radius"].append(
                            copy.deepcopy(system.radius)
                        )
                        self.callback_params["com"].append(
                            system.compute_position_center_of_mass()
                        )
                        return

            self.post_processing_dict_sphere = defaultdict(list)
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

        ###--------------FINALIZE SIMULATION--------------###
        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )
        self.time_tracker = np.float64(0.0)

        state = self.get_state()
        self._target = self.wsol[self.tick]

        if return_info:
            return state, {}
        else:
            return state

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
            self.renderer = Session(width=maxwidth, height=int(maxwidth*aspect_ratio))
            self.renderer.add_rod(self.shearable_rod)  
            self.renderer.add_point(self._target.tolist(), self.sphere_radius)

        # POVRAY
        if RENDERER_CONFIG == RendererType.POVRAY:
            state_image = self.renderer.render(
                maxwidth, int(maxwidth * aspect_ratio * 0.7),
                camera_param=("location", [50.0, 10.0, -60.0], "look_at", [-3.5, 5.7, 10.0], "angle", 30)
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
