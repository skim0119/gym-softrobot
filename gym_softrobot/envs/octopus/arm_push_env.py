from typing import Optional

import gym
from gym import core, spaces
from gym.utils import seeding

from collections import defaultdict
import time

import numpy as np

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check

from gym_softrobot import RENDERER_CONFIG
from gym_softrobot.config import RendererType
from gym_softrobot.utils.custom_elastica.callback_func import (
    RodCallBack,
)
from gym_softrobot.utils.custom_elastica.constraint import ControllableFixConstraint
from gym_softrobot.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)

from gym_softrobot.utils.actuation.actuations.muscles.longitudinal_muscle import (
    LongitudinalMuscle,
)
from gym_softrobot.utils.actuation.actuations.muscles.transverse_muscle import (
    TransverseMuscle,
)
from gym_softrobot.utils.actuation.actuations.muscles.muscle import ApplyMuscle


class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


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
        recording_fps: int = 40,
        config_generate_video: bool = False,
        config_early_termination: bool = False,
    ):

        # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (recording_fps * time_step))

        # Spaces
        self.n_elem = 40
        n_action = self.n_elem - 1
        self.action_space = spaces.Box(0.0, 1.0, shape=(n_action,), dtype=np.float32)
        self._observation_size = (
            (self.n_elem + 1 + n_action),
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
        self.config_early_termination = config_early_termination

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
        self.shearable_rod = self._build()

        """ Create time stepper """
        self.StatefulStepper = PositionVerlet()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        self.time = np.float64(0.0)

        # Set Target
        self._target = (2 - 0.5) * self.np_random.random(2) + 0.5
        # self._target /= np.linalg.norm(self._target) # I don't see why this is here
        print(self._target)

        # Initial State
        state = self.get_state()

        if return_info:
            return state, {}
        else:
            return state

    def _build(self):
        """Set up arm params"""
        n_elem = self.n_elem
        L0 = 0.2
        radius_base = 0.012  # radius of the arm at the base
        radius_tip = 0.001  # radius of the arm at the tip
        radius = np.linspace(radius_base, radius_tip, n_elem + 1)
        radius_mean = (radius[:-1] + radius[1:]) / 2
        damp_coefficient = 0.05 * 2

        shearable_rod = CosseratRod.straight_rod(
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
        self.simulator.append(shearable_rod)

        controller_id = self.simulator.constrain(shearable_rod).using(
            ControllableFixConstraint,
            index=0,
        ).id()

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
                    (np.zeros(radius_mean.shape), 6 / 9 * np.ones(radius_mean.shape)),
                    axis=0,
                ),
                max_force=0.5 * (radius_mean / radius_base) ** 2,
            ),
            LongitudinalMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape), -6 / 9 * np.ones(radius_mean.shape)),
                    axis=0,
                ),
                max_force=0.5 * (radius_mean / radius_base) ** 2,
            ),
            TransverseMuscle(
                muscle_radius_ratio=np.stack(
                    (np.zeros(radius_mean.shape), 4 / 9 * np.ones(radius_mean.shape)),
                    axis=0,
                ),
                max_force=1.0 * (radius_mean / radius_base) ** 2,
            ),
        ]

        self.muscles_parameters = []
        for _ in self.muscle_layers:
            self.muscles_parameters.append(defaultdict(list))

        self.simulator.add_forcing_to(shearable_rod).using(
            ApplyMuscle,
            muscles=self.muscle_layers,
            step_skip=self.step_skip,
            callback_params_list=self.muscles_parameters,
        )

        # CallBack
        if self.config_generate_video:
            self.rod_parameters_dict = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                RodCallBack,
                step_skip=self.step_skip,
                callback_params=self.rod_parameters_dict,
            )

        self.simulator.finalize()

        controllable_constraint = dict(self.simulator._constraints)[controller_id]
        self.BC = controllable_constraint.get_controller

        return shearable_rod

    def get_state(self):
        # Build state
        rod = self.shearable_rod
        pos_state1 = rod.position_collection[0]  # x
        previous_action = self._prev_action.copy()
        state = np.hstack(
            [
                pos_state1,
                previous_action,
            ]
        ).astype(np.float32)
        return state

    def set_action(self, action) -> None:
        scale = 1.0  # min(time / 0.02, 1)
        for muscle_count, muscle_layer in enumerate(self.muscle_layers):
            muscle_layer.set_activation(action[muscle_count] * scale)

    def step(self, action):
        rest_kappa = action  # alias

        """ Set intrinsic strains (set actions) """
        self.set_action(action)

        """ Post-simulation """
        prev_cm_pos = self.shearable_rod.compute_position_center_of_mass()[:2]

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

        if self.config_early_termination:
            done = self.check_early_termination()
            survive_reward = -10.0
        elif invalid_values_condition:
            print(f" Nan detected in, exiting simulation now. {self.time=}")
            done = True
            survive_reward = -50.0
        else:
            cm_pos = self.shearable_rod.compute_position_center_of_mass()[:2]
            moved_distance = np.linalg.norm(cm_pos, ord=2) - np.linalg.norm(
                prev_cm_pos, ord=2
            )
            forward_reward = moved_distance * 10

        """ Time limit """
        timelimit = False
        if self.time > self.final_time:
            survive_reward = np.linalg.norm(cm_pos, ord=2) * 10
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

        return states, reward, done, info

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
            self.renderer.add_rod(self.shearable_rod)
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

    # def save_data(self, dir, eps):

    #     print("Saving data to pickle files...")

    #     import pickle

    #     with open(dir + "/simulation_data%03d.pickle" % eps, "wb") as f:
    #         data = dict(
    #             rods=[self.rod_parameters_dict], muscles=self.muscles_parameters
    #         )
    #         pickle.dump(data, f)

    #     with open(dir + "/simulation_systems%03d.pickle" % eps, "wb") as f:
    #         data = dict(rods=[self.shearable_rod], muscles=self.muscle_layers)
    #         pickle.dump(data, f)

    def check_early_termination(self, cutoff_error=1e-7):
        desired_Hamiltonian = self.cal_desired_Hamiltonian()
        if desired_Hamiltonian < cutoff_error:
            return True
        return False

    def cal_desired_Hamiltonian(self):
        kinetic_energy = (
            self.shearable_rod.compute_translational_energy()
            + self.shearable_rod.compute_rotational_energy()
        )
        desired_potential_energy = (
            self.shearable_rod.compute_shear_energy()
            + self.shearable_rod.compute_bending_energy()
        )
        return kinetic_energy + desired_potential_energy

    # def post_processing(self, algo_data, cutoff_error=1e-7):
    #     for k in range(len(self.rod_parameters_dict["time"])):

    #         # calculate the desired Hamiltonian for every time frame
    #         kinetic_energy = compute_translational_energy(
    #             self.shearable_rod.mass, self.rod_parameters_dict["velocity"][k]
    #         ) + compute_rotational_energy(
    #             self.shearable_rod.mass_second_moment_of_inertia,
    #             self.rod_parameters_dict["omega"][k],
    #             self.rod_parameters_dict["dilatation"][k],
    #         )
    #         desired_potential_energy = compute_shear_energy(
    #             self.rod_parameters_dict["sigma"][k],
    #             algo_data["sigma"],
    #             self.shearable_rod.shear_matrix,
    #             self.rod_parameters_dict["dilatation"][k]
    #             * self.shearable_rod.rest_lengths,
    #         ) + compute_bending_energy(
    #             self.rod_parameters_dict["kappa"][k],
    #             algo_data["kappa"],
    #             self.shearable_rod.bend_matrix,
    #             self.rod_parameters_dict["voronoi_dilatation"][k]
    #             * self.shearable_rod.rest_voronoi_lengths,
    #         )
    #         desired_Hamiltonian = kinetic_energy + desired_potential_energy

    #         # calculate the control energy
    #         self.muscles_parameters[0]["control_energy"].append(
    #             0.5 * np.sum(self.muscles_parameters[0]["activation"][k] ** 2)
    #         )

    #         # check if the desired Hamiltonian is smaller then cutoff error
    #         flag = False
    #         if desired_Hamiltonian < cutoff_error:
    #             flag = True
    #         self.rod_parameters_dict["stable_flag"].append(flag)

    #     for k in range(len(self.rod_parameters_dict["time"])):
    #         if self.rod_parameters_dict["stable_flag"][-k - 1] is False:
    #             self.rod_parameters_dict["stable_flag"][: -k - 1] = [
    #                 False
    #                 for kk in range(
    #                     len(self.rod_parameters_dict["stable_flag"][: -k - 1])
    #                 )
    #             ]
    #             break

    #     energy_cost = 0
    #     reach_time = self.rod_parameters_dict["time"][-1]
    #     for k in range(len(self.rod_parameters_dict["time"])):
    #         if self.rod_parameters_dict["stable_flag"][k] is False:
    #             energy_cost += self.muscles_parameters[0]["control_energy"][k]
    #         else:
    #             reach_time = self.rod_parameters_dict["time"][k]

    #     return reach_time, energy_cost
