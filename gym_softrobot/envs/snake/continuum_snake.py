__doc__ = """Snake friction case from X. Zhang et. al. Nat. Comm. 2021"""

""" This environment is constructed based on the PyElastica:ContinuumSnakeCase """

from collections import defaultdict
from functools import partial

from tqdm import tqdm

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from elastica import *
from elastica.utils import _bspline

import numpy as np

from gym_softrobot.utils.render.continuum_snake_postprocessing import (
    plot_snake_velocity,
    plot_video,
    plot_curvature,
)

def compute_projected_velocity(plot_params: dict, period):

    time_per_period = np.array(plot_params["time"]) / period
    avg_velocity = np.array(plot_params["avg_velocity"])
    center_of_mass = np.array(plot_params["center_of_mass"])

    # Compute rod velocity in rod direction. We need to compute that because,
    # after snake starts to move it chooses an arbitrary direction, which does not
    # have to be initial tangent direction of the rod. Thus we need to project the
    # snake velocity with respect to its new tangent and roll direction, after that
    # we will get the correct forward and lateral speed. After this projection
    # lateral velocity of the snake has to be oscillating between + and - values with
    # zero mean.

    # Number of steps in one period.
    period_step = int(1.0 / (time_per_period[-1] - time_per_period[-2]))
    number_of_period = int(time_per_period[-1])
    if number_of_period-2 <= 0:
        return (0,0,0,0)

    # Center of mass position averaged in one period
    center_of_mass_averaged_over_one_period = np.zeros((number_of_period - 2, 3))
    for i in range(1, number_of_period - 1):
        # position of center of mass averaged over one period
        center_of_mass_averaged_over_one_period[i - 1] = np.mean(
            center_of_mass[(i + 1) * period_step : (i + 2) * period_step]
            - center_of_mass[(i + 0) * period_step : (i + 1) * period_step],
            axis=0,
        )
    # Average the rod directions over multiple periods and get the direction of the rod.
    direction_of_rod = np.mean(center_of_mass_averaged_over_one_period, axis=0)
    direction_of_rod /= np.linalg.norm(direction_of_rod, ord=2)

    # Compute the projected rod velocity in the direction of the rod
    velocity_mag_in_direction_of_rod = np.einsum(
        "ji,i->j", avg_velocity, direction_of_rod
    )
    velocity_in_direction_of_rod = np.einsum(
        "j,i->ji", velocity_mag_in_direction_of_rod, direction_of_rod
    )

    # Get the lateral or roll velocity of the rod after subtracting its projected
    # velocity in the direction of rod
    velocity_in_rod_roll_dir = avg_velocity - velocity_in_direction_of_rod

    # Compute the average velocity over the simulation, this can be used for optimizing snake
    # for fastest forward velocity. We start after first period, because of the ramping up happens
    # in first period.
    average_velocity_over_simulation = np.mean(
        velocity_in_direction_of_rod[period_step * 2 :], axis=0
    )

    return (
        velocity_in_direction_of_rod,
        velocity_in_rod_roll_dir,
        average_velocity_over_simulation[2],
        average_velocity_over_simulation[0],
    )


class SnakeSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass


class ContinuumSnakeCallBack(CallBackBaseClass):
    """
    Call back function for continuum snake
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
            self.callback_params["avg_velocity"].append(
                system.compute_velocity_center_of_mass()
            )

            self.callback_params["center_of_mass"].append(
                system.compute_position_center_of_mass()
            )
            self.callback_params["curvature"].append(system.kappa.copy())
            return


class ContinuumSnakeEnv(gym.Env):
    metadata = {'render.modes': ['rgb', 'human']}

    def __init__(self):
        # Action space
        action_space_low = -np.ones(7) * 1e-2; action_space_low[-1] = 0.5
        action_space_high = np.ones(7) * 1e-2; action_space_high[-1] = 3.0
        self.action_space = spaces.Box(action_space_low, action_space_high, dtype=np.float32)

        # State space
        self._n_elem = 50
        observation_space = (((self._n_elem+1) * 3 * 2) + ((self._n_elem) * 9),)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=observation_space, dtype=np.float32)

        self.metadata = {}

        # Determinism
        self.seed()

    def seed(self, seed=None):
        # Deprecated in new gym
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.snake_sim, self.stepper, self.muscle_torque, self.data = self._build()
        self.time= np.float64(0.0)
        return self.get_state()

    def get_state(self):
        # Build state
        rod = self.shearable_rod
        pos_state = rod.position_collection.ravel()
        vel_state = rod.velocity_collection.ravel()
        dir_state = rod.director_collection.ravel()
        return np.concatenate([pos_state, vel_state, dir_state], dtype=np.float32)

    def set_action(self, action):
        wave_length = action[-1]
        b_coeff = action[:-1]
        self.muscle_torque().__init__(
            base_length=self.base_length,
            b_coeff=b_coeff,
            period=self.period,
            wave_number=2.0 * np.pi / (wave_length),
            phase_shift=0.0,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=self.period,
            direction=np.array([0.0, 1.0, 0.0]),
            with_spline=True,
        )

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid: expected {self.action_space}"
        assert self.action_space.contains(action), err_msg

        """ Set action """
        self.set_action(action)

        """ Run simulation """
        for _ in range(self.step_skip):
            self.time = self.stepper(time=self.time)

        # Compute the average forward velocity. These will be used for optimization.
        [_, _, avg_forward, avg_lateral] = compute_projected_velocity(self.data, self.period)

        done = False
        if self.time >= self.final_time:
            done = True

        info = {}

        return self.get_state(), avg_forward, done, info


    def render(self, mode='human', close=False):
        filename_plot = "continuum_snake_velocity.png"
        plot_snake_velocity(self.data, self.period, filename_plot, 1)
        plot_curvature(self.data, self.shearable_rod.rest_lengths, self.period, 1)

        if SAVE_VIDEO:
            filename_video = "continuum_snake.mp4"
            plot_video(
                self.data,
                video_name=filename_video,
                fps=60,
                xlim=(0, 4),
                ylim=(-1, 1),
            )

    def close(self):
        pass

    def _build(self, include_callback:bool=False):
        # Initialize the simulation class
        snake_sim = SnakeSimulator()

        # Simulation parameters
        period = 2
        self.period = period
        final_time = (11.0 + 0.01) * period
        self.final_time = final_time
        time_step = 8e-6
        total_steps = int(final_time / time_step)
        rendering_fps = 5
        step_skip = int(1.0 / (rendering_fps * time_step))
        self.step_skip = step_skip
        callback_step_skip = int(1.0 / (60 * time_step))

        # setting up test params
        n_elem = self._n_elem
        start = np.zeros((3,))
        direction = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        base_length = 0.35
        self.base_length = base_length
        base_radius = base_length * 0.011
        density = 1000
        nu = 1e-4
        E = 1e6
        poisson_ratio = 0.5
        shear_modulus = E / (poisson_ratio + 1.0)

        # Add rod
        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            E,
            shear_modulus=shear_modulus,
        )
        snake_sim.append(self.shearable_rod)

        # Add gravitational forces
        gravitational_acc = -9.80665
        snake_sim.add_forcing_to(self.shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
        )

        # Add muscle torques
        wave_length = 1
        muscle_torque = snake_sim.add_forcing_to(self.shearable_rod).using(
            MuscleTorques,
            base_length=base_length,
            b_coeff=np.zeros(4),
            period=period,
            wave_number=2.0 * np.pi / (wave_length),
            phase_shift=0.0,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=period,
            direction=normal,
            with_spline=True,
        )

        # Add friction forces
        origin_plane = np.array([0.0, -base_radius, 0.0])
        normal_plane = normal
        slip_velocity_tol = 1e-8
        froude = 0.1
        mu = base_length / (period * period * np.abs(gravitational_acc) * froude)
        kinetic_mu_array = np.array(
            [mu, 1.5 * mu, 2.0 * mu]
        )  # [forward, backward, sideways]
        static_mu_array = np.zeros(kinetic_mu_array.shape)
        snake_sim.add_forcing_to(self.shearable_rod).using(
            AnisotropicFrictionalPlane,
            k=1.0,
            nu=1e-6,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )

        # Callback
        data = defaultdict(list)
        snake_sim.collect_diagnostics(self.shearable_rod).using(
            ContinuumSnakeCallBack, step_skip=callback_step_skip, callback_params=data
        )

        # Integrator
        timestepper = PositionVerlet()
        snake_sim.finalize()
        #integrate(timestepper, snake_sim, final_time, total_steps)
        do_step, stages_and_updates = extend_stepper_interface(
            timestepper, snake_sim
        )
        stepper = partial(
                do_step,
                dt=time_step,
                TimeStepper=PositionVerlet(),
                SystemCollection=snake_sim,
                _steps_and_prefactors=stages_and_updates,
            )

        return snake_sim, stepper, muscle_torque, data
