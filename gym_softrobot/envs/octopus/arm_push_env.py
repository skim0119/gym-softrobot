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


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


@dataclass
class PointerBC:
    flag: bool = None
    fixed_position


class ArmPushEnv(core.Env):
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
        final_time: float = 2.5,
        time_step: float = 7.0e-5,
        recording_fps: int = 5,
        n_action: int = 3,
        config_generate_video: bool = False,
    ):

        # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))


        # Spaces
        self.n_action = n_action  # number of interpolation point (3 curvatures)
        action_size = (self.n_action,)
        action_low = np.ones(action_size) * (-22)
        action_high = np.ones(action_size) * (22)
        self.action_space = spaces.Box(
            action_low, action_high, shape=action_size, dtype=np.float32
        )
        self._observation_size = (
            (self.n_seg + (self.n_elems + 1) * 4 + self.n_action + 2),
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

    def reset(self):
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
        self.StatefulStepper = PositionVerlet()
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        self.time = np.float64(0.0)
        self.counter = 0

        # Set Target
        self._target = (2 - 0.5) * self.np_random.random(2) + 0.5
        # self._target /= np.linalg.norm(self._target) # I don't see why this is here
        print(self._target)

        # Initial State
        state = self.get_state()

        # Preprocessing
        self.prev_dist_to_target = np.linalg.norm(
            self.shearable_rod.compute_position_center_of_mass()[:2] - self._target,
            ord=2,
        )
        # self.prev_cm_vel = self.shearable_rod.compute_velocity_center_of_mass()

        return state

	def reset(self,cylinder_params=None):
        self.simulator = BaseSimulator()

        """ Set up arm params """
        n_elem = 100
        L0 = 0.2
        radius_base = 0.012  # radius of the arm at the base
        radius_tip = 0.001  # radius of the arm at the tip
        radius = np.linspace(radius_base, radius_tip, n_elem + 1)
        radius_mean = (radius[:-1] + radius[1:]) / 2
        damp_coefficient = 0.05*2

        self.shearable_rod = CosseratRod.straight_rod(
            n_elements=n_elem,
            start=np.zeros((3,)),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, -1.0]),
            base_length=L0,
            base_radius=radius_mean.copy(),
            density=700,
            nu=damp_coefficient * ((radius_mean / radius_base) ** 2) * 1.0,
            youngs_modulus=1e4,
            poisson_ratio=0.5,
            nu_for_torques=damp_coefficient * ((radius_mean / radius_base) ** 4),
        )
        self.simulator.append(self.shearable_rod)

        """ Set up boundary condition """
        # self.simulator.constrain(self.shearable_rod).using(
        #     OneEndFixedRod,
        #     constrained_position_idx=(0,),
        #     constrained_director_idx=(0,)
        # )

        init_fixed_index = 0
        init_fixed_pos = self.shearable_rod.position_collection[..., init_fixed_index]
        init_fixed_dir = self.shearable_rod.director_collection[..., init_fixed_index]
        self.BC = PointerBC(True, init_fixed_pos, init_fixed_dir, init_fixed_index)
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod_with_Flag, pointer=self.BC
        )

        self.rod_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            RodCallBack,
            step_skip=self.step_skip,
            callback_params=self.rod_parameters_dict
        )

        """ Add muscle actuation """
        self.muscle_layers = [
            # LongitudinalMuscle(
            #     muscle_radius_ratio=np.stack(
            #         (np.zeros(radius_mean.shape),
            #          2 / 3 * np.ones(radius_mean.shape)),
            #         axis=0),
            #     max_force=1 * (radius_mean / radius_base) ** 2,
            # )
            LongitudinalMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape),
                     6 / 9 * np.ones(radius_mean.shape)),
                    axis=0),
                max_force=0.5 * (radius_mean / radius_base) ** 2,
            ),
            LongitudinalMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape),
                     -6 / 9 * np.ones(radius_mean.shape)),
                    axis=0),
                max_force=0.5 * (radius_mean / radius_base) ** 2,
            ),
            TransverseMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape),
                     4 / 9 * np.ones(radius_mean.shape)),
                    axis=0),
                max_force=1.0 * (radius_mean / radius_base) ** 2,
            )
        ]

        self.muscles_parameters = []
        for _ in self.muscle_layers:
            self.muscles_parameters.append(defaultdict(list))

        self.simulator.add_forcing_to(self.shearable_rod).using(
            ApplyMuscle,
            muscles=self.muscle_layers,
            step_skip=self.step_skip,
            callback_params_list=self.muscles_parameters,
        )

        # """ Set up a cylinder object """
        if cylinder_params!=None:
            self.cylinder = Cylinder(
                start=cylinder_params['start']* L0,# np.array([0.7,0.45, -0.2]) * L0,
                # start=np.array([0.8,0.65, -0.2]) * L0,
                direction=cylinder_params['direction'],#np.array([0, 0, 1]),
                normal=cylinder_params['normal'],#np.array([1, 0, 0]),
                base_length=cylinder_params['length']* L0,#0.4 * L0,
                base_radius=cylinder_params['radius']* L0,#0.2 * L0,
                # base_radius=0.4 * L0,
                density=200,
            )

            self.simulator.append(self.cylinder)
            self.simulator.constrain(self.cylinder).using(
                OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
            )
            self.simulator.connect(self.shearable_rod, self.cylinder).using(
                ExternalContact, k=0.001, nu=0.1
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

        self.simulator.finalize()

        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        systems = [self.shearable_rod]

        return self.total_steps, systems

    def get_state(self):
        # Build state
        rod = self.shearable_rod
        kappa_state = rod.kappa[0]
        pos_state1 = rod.position_collection[0]  # x
        pos_state2 = rod.position_collection[1]  # y
        vel_state1 = rod.velocity_collection[0]  # x
        vel_state2 = rod.velocity_collection[1]  # y
        previous_action = self._prev_action.copy()
        target = self._target
        state = np.hstack(
            [
                kappa_state,
                pos_state1,
                pos_state2,
                vel_state1,
                vel_state2,
                previous_action,
                target,
            ]
        ).astype(np.float32)
        return state

    def set_action(self, action) -> None:
        self._prev_action[:] = action
        action = np.concatenate([[0], action, [0]], axis=-1)
        action = interp1d(
            np.linspace(0, 1, self.n_action + 2),  # added zero on the boundary
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
        done = False
        survive_reward = 0.0
        forward_reward = 0.0
        control_panelty = 0.005 * np.square(rest_kappa.ravel()).mean()
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
            cm_pos = self.shearable_rod.compute_position_center_of_mass()[:2]
            dist_to_target = np.linalg.norm(cm_pos - self._target, ord=2)
            forward_reward = (self.prev_dist_to_target - dist_to_target) * 10
            self.prev_dist_to_target = dist_to_target
            """ Goal """
            if dist_to_target < 0.1:
                survive_reward = 100.0
                done = True

        """ Time limit """
        timelimit = False
        if self.time > self.final_time:
            timelimit = True
            done = True

        reward = forward_reward - control_panelty + survive_reward
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
        """Pass"""
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
            # TODO Maybe add 2D rendering instead
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

















    def step(self, time, muscle_activations):

        # set muscle activations
        # self.muscle_layers[0].set_activation(muscle_activations[0])
        scale = 1.0#min(time / 0.02, 1)
        for muscle_count in range(len(self.muscle_layers)):
            self.muscle_layers[muscle_count].set_activation(muscle_activations[muscle_count] * scale)

        # run the simulation for one step
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        if self.early_termination:
            done = self.check_early_termination()
        else:
            """ Done is a boolean to reset the environment before episode is completed """
            done = False
            # Position of the rod cannot be NaN, it is not valid, stop the simulation
            invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

            if invalid_values_condition == True:
                print(" Nan detected, exiting simulation now")
                done = True
            """ Done is a boolean to reset the environment before episode is completed """

        # systems = [self.shearable_rod]
        systems = [self.shearable_rod,self.cylinder]

        return time, systems, done

    def save_data(self, dir, eps):

        print("Saving data to pickle files...")

        import pickle

        with open(dir + "/simulation_data%03d.pickle" % eps, "wb") as f:
            data = dict(
                rods=[self.rod_parameters_dict],
                muscles=self.muscles_parameters
            )
            pickle.dump(data, f)

        with open(dir + "/simulation_systems%03d.pickle" % eps, "wb") as f:
            data = dict(
                rods=[self.shearable_rod],
                muscles=self.muscle_layers
            )
            pickle.dump(data, f)

    def set_algo_data(self, algo_data):
        self.desired_sigma = algo_data['sigma']
        self.desired_kappa = algo_data['kappa']

    def check_early_termination(self, cutoff_error=1e-7):
        desired_Hamiltonian = self.cal_desired_Hamiltonian()
        if desired_Hamiltonian < cutoff_error:
            return True
        return False

    def cal_desired_Hamiltonian(self):
        kinetic_energy = (
                compute_translational_energy(
                    self.shearable_rod.mass,
                    self.shearable_rod.velocity_collection) +
                compute_rotational_energy(
                    self.shearable_rod.mass_second_moment_of_inertia,
                    self.shearable_rod.omega_collection,
                    self.shearable_rod.dilatation
                )
        )
        desired_potential_energy = (
                compute_shear_energy(
                    self.shearable_rod.sigma,
                    self.desired_sigma,
                    self.shearable_rod.shear_matrix,
                    self.shearable_rod.dilatation * self.shearable_rod.rest_lengths
                ) +
                compute_bending_energy(
                    self.shearable_rod.kappa,
                    self.desired_kappa,
                    self.shearable_rod.bend_matrix,
                    self.shearable_rod.voronoi_dilatation * self.shearable_rod.rest_voronoi_lengths
                )
        )
        return kinetic_energy + desired_potential_energy

    def post_processing(self, algo_data, cutoff_error=1e-7):
        for k in range(len(self.rod_parameters_dict['time'])):

            # calculate the desired Hamiltonian for every time frame
            kinetic_energy = (
                    compute_translational_energy(
                        self.shearable_rod.mass,
                        self.rod_parameters_dict['velocity'][k]) +
                    compute_rotational_energy(
                        self.shearable_rod.mass_second_moment_of_inertia,
                        self.rod_parameters_dict['omega'][k],
                        self.rod_parameters_dict['dilatation'][k]
                    )
            )
            desired_potential_energy = (
                    compute_shear_energy(
                        self.rod_parameters_dict['sigma'][k],
                        algo_data['sigma'],
                        self.shearable_rod.shear_matrix,
                        self.rod_parameters_dict['dilatation'][k] * self.shearable_rod.rest_lengths
                    ) +
                    compute_bending_energy(
                        self.rod_parameters_dict['kappa'][k],
                        algo_data['kappa'],
                        self.shearable_rod.bend_matrix,
                        self.rod_parameters_dict['voronoi_dilatation'][k] * self.shearable_rod.rest_voronoi_lengths
                    )
            )
            desired_Hamiltonian = kinetic_energy + desired_potential_energy

            # calculate the control energy
            self.muscles_parameters[0]['control_energy'].append(
                0.5 * np.sum(self.muscles_parameters[0]['activation'][k] ** 2)
            )

            # check if the desired Hamiltonian is smaller then cutoff error
            flag = False
            if desired_Hamiltonian < cutoff_error:
                flag = True
            self.rod_parameters_dict['stable_flag'].append(flag)

        for k in range(len(self.rod_parameters_dict['time'])):
            if self.rod_parameters_dict['stable_flag'][-k - 1] is False:
                self.rod_parameters_dict['stable_flag'][:-k - 1] = (
                    [False for kk in range(len(self.rod_parameters_dict['stable_flag'][:-k - 1]))]
                )
                break

        energy_cost = 0
        reach_time = self.rod_parameters_dict['time'][-1]
        for k in range(len(self.rod_parameters_dict['time'])):
            if self.rod_parameters_dict['stable_flag'][k] is False:
                energy_cost += self.muscles_parameters[0]['control_energy'][k]
            else:
                reach_time = self.rod_parameters_dict['time'][k]

        return reach_time, energy_cost
