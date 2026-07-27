"""Build the physical system used by the three-dimensional pendulum task."""

from dataclasses import dataclass, field

import numpy as np
from elastica import (
    AnalyticalLinearDamper,
    ConstraintBase,
    CosseratRod,
    GravityForces,
    LaplaceDissipationFilter,
)


@dataclass
class MovingBaseController:
    """State shared between the environment and the base constraint."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


class MovingBaseConstraint(ConstraintBase):
    """Fix rod orientation while allowing commanded motion in the x-y plane."""

    def __init__(self, fixed_position, fixed_director, controller, **kwargs):
        super().__init__(**kwargs)
        self.fixed_height = fixed_position[2]
        self.fixed_director = fixed_director
        self.controller = controller

    def constrain_values(self, system, time):
        system.position_collection[:, 0] = self.controller.position
        system.position_collection[2, 0] = self.fixed_height
        system.director_collection[:, :, 0] = self.fixed_director

    def constrain_rates(self, system, time):
        system.velocity_collection[:, 0] = self.controller.velocity
        system.velocity_collection[2, 0] = 0.0
        system.omega_collection[:, 0] = 0.0


def build_soft_pendulum_3d(
    simulator,
    n_elem: int,
    controller: MovingBaseController,
    np_random: np.random.Generator,
    time_step: float,
):
    """Create a vertical, gravity-loaded rod with a movable constrained base."""
    tilt = np.deg2rad(np_random.uniform(-1.0, 1.0))
    direction = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    normal = np.array([0.0, 1.0, 0.0])

    rod = CosseratRod.straight_rod(
        n_elements=n_elem,
        start=np.zeros(3),
        direction=direction,
        normal=normal,
        base_length=1.0,
        base_radius=0.1,
        density=4000.0,
        youngs_modulus=1e6,
    )
    simulator.append(rod)

    simulator.constrain(rod).using(
        MovingBaseConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
        controller=controller,
    )
    simulator.add_forcing_to(rod).using(
        GravityForces,
        acc_gravity=np.array([0.0, 0.0, -9.80665]),
    )
    simulator.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant=1.0,
        time_step=time_step,
    )
    simulator.dampen(rod).using(
        LaplaceDissipationFilter,
        filter_order=7,
    )
    return rod
