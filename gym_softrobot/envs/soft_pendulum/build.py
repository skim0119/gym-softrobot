__doc__ = """
Module contains elastica interface to create soft pendulum.
"""

from typing import Optional, Tuple

import numpy as np

from elastica import *
from elastica.boundary_conditions import ConstraintBase
from elastica.timestepper import extend_stepper_interface


_PENDULUM_PROPERTIES = {  # default parameters
    # Arm properties
    "youngs_modulus": 1e6,
    "shear_modulus": 1e6/(1+0.5),
    #"poisson_ratio": 0.5,
    "density": 1000.0,
    "nu": 50,
}
_DEFAULT_SCALE_LENGTH = {
    "base_length": 1.0,
    "base_radius": 0.05,
}


def build_soft_pendulum(
    simulator, n_elem, point_force, np_random, override_params: Optional[dict] = None
):
    """Import default parameters (overridable)"""
    param = _PENDULUM_PROPERTIES.copy()  # Always copy parameter for safety
    if isinstance(override_params, dict):
        param.update(override_params)
    """ Import default parameters (non-overridable) """
    pendulum_scale_param = _DEFAULT_SCALE_LENGTH.copy()

    """ Set pendulum """

    start = np.zeros((3,))
    theta = np.deg2rad(
        90 + (np_random.random() - 0.5) * 10
    )  # Change this if you want to start with different initial condition.
    direction = np.array([1.0 * np.cos(theta), 1.0 * np.sin(theta), 0.0])
    normal = np.array([1.0 * np.sin(theta), -1.0 * np.cos(theta), 0.0])
    binormal = np.cross(direction, normal)

    shearable_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        **pendulum_scale_param,
        **param,
    )
    simulator.append(shearable_rod)

    # Add boundary conditions
    class PendulumBoundaryConditions(ConstraintBase):
        def __init__(self, fixed_position, fixed_directors, **kwargs):
            super().__init__(**kwargs)
            self.fixed_position = fixed_position
            self.fixed_directors = fixed_directors

        def constrain_values(self, rod, time):
            rod.position_collection[1:, 0] = self.fixed_position[1:]
            rod.director_collection[0, :, 0] = self.fixed_directors[0, :]
            rod.director_collection[2, :, 0] = self.fixed_directors[2, :]

        def constrain_rates(self, rod, time):
            rod.velocity_collection[1:, 0] = 0
            rod.omega_collection[0, 0] = 0
            rod.omega_collection[2, 0] = 0

    simulator.constrain(shearable_rod).using(
        PendulumBoundaryConditions,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    # Add gravity
    gravitational_acc = -9.80665
    simulator.add_forcing_to(shearable_rod).using(
        GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
    )

    # Apply external forces to the base of the rod.
    class PendulumPointForces(NoForces):
        def __init__(self, point_force):
            super(PendulumPointForces, self).__init__()

            self.point_force = point_force

        def apply_forces(self, system, time: np.float = 0.0):
            system.external_forces[0, 0] = self.point_force

    simulator.add_forcing_to(shearable_rod).using(
        PendulumPointForces, point_force=point_force
    )

    return shearable_rod
