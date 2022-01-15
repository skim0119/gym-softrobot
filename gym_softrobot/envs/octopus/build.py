from typing import Optional, Tuple
import json

import numpy as np

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica.interaction import AnisotropicFrictionalPlaneRigidBody

from gym_softrobot.utils.custom_elastica.joint import FixedJoint2Rigid
from gym_softrobot.utils.actuation.forces.drag_force import DragForce

from scipy.spatial.transform import Rotation as Rot

_PARAM = {
        "youngs_modulus": 1e6,
        "arm_density": 1000.0,
        "body_arm_k": 1e6,
        "body_arm_kt": 5e-1,
        "head_radius": 0.05,
        "head_density": 1000.0,
        "friction_multiplier": 0.02,
        "friction_symmetry": False
    }


def build_octopus(
        simulator,
        n_arm:int=8,
        n_elem:int=11,
        override_params:Optional[dict]=None
    ):
    """ Import default parameters """
    param = _PARAM
    if isinstance(override_params, dict):
        param.update(override_params)

    """ Set up an arm """
    L0 = 0.35                # total length of the arm
    r0 = L0 * 0.03

    rigid_rod_length = r0 * 2
    rigid_rod_radius = param['head_radius']

    # radius_base = r0     # radius of the arm at the base
    # radius_tip = r0     # radius of the arm at the tip
    # radius = np.linspace(radius_base, radius_tip, n_elem+1)
    # radius_mean = (radius[:-1]+radius[1:])/2
    # damp_coefficient = 0.02

    rotation_angle=360/n_arm
    angle_list=[rotation_angle*arm_i for arm_i in range(n_arm)]

    shearable_rods=[]  # arms
    for arm_i in range(n_arm):
        arm_angle = angle_list[arm_i]
        rot = Rot.from_euler('z', arm_angle, degrees=True)
        arm_pos = rot.apply([rigid_rod_radius, 0.0, 0.0])
        arm_dir = rot.apply([1.0, 0.0, 0.0])
        rod = CosseratRod.straight_rod(
            n_elements=n_elem,
            start=arm_pos,
            direction=arm_dir,
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=L0,
            base_radius=r0,
            density=param['arm_density'],
            nu=1e-3,
            youngs_modulus=param['youngs_modulus'],
            poisson_ratio=0.5,
            # nu_for_torques=damp_coefficient*((radius_mean/radius_base)**4),
        )
        shearable_rods.append(rod)
        simulator.append(rod)

    """ Add head """
    start = np.zeros((3,))
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    binormal = np.cross(direction, normal)
    base_area = np.pi * rigid_rod_radius ** 2
    density = param['head_density']

    rigid_rod = Cylinder(start, direction, normal, rigid_rod_length, rigid_rod_radius, density)
    simulator.append(rigid_rod)

    """ Set up boundary conditions """
    for arm_i in range(n_arm):
        _k = param['body_arm_k']
        _kt = param['body_arm_kt']
        simulator.connect(
            first_rod=rigid_rod, second_rod=shearable_rods[arm_i], first_connect_idx=-1, second_connect_idx=0
        ).using(FixedJoint2Rigid, k=_k, nu=1e-3, kt=_kt,angle=angle_list[arm_i],radius=rigid_rod_radius)

    """Add gravity forces"""
    _g = -9.81
    gravitational_acc = np.array([0.0, 0.0, _g])
    for arm_i in range(n_arm):
        simulator.add_forcing_to(shearable_rods[arm_i]).using(
                GravityForces, acc_gravity=gravitational_acc
            )
    simulator.add_forcing_to(rigid_rod).using(
                GravityForces, acc_gravity=gravitational_acc
            )

    """ Add drag force """
    # dl = L0 / n_elem
    # fluid_factor = 1
    # r_bar = (radius_base + radius_tip) / 2
    # sea_water_dentsity = 1022
    # c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
    # c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor
    #
    # simulator.add_forcing_to(self.shearable_rod).using(
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
    mu = base_length / (period * period * np.abs(_g) * froude)
    if param['friction_symmetry']:
        kinetic_mu_array = np.array(
            [mu, mu, mu]
        ) * param['friction_multiplier'] # [forward, backward, sideways]
    else:
        kinetic_mu_array = np.array(
            [mu, 1.5 * mu, 2.0 * mu]
        ) * param['friction_multiplier'] # [forward, backward, sideways]
    static_mu_array = 2 * kinetic_mu_array
    for arm_i in range(n_arm):
        simulator.add_forcing_to(shearable_rods[arm_i]).using(
            AnisotropicFrictionalPlane,
            k=1.0,
            nu=1e-6,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
    mu = base_length / (period * period * np.abs(_g) * froude)
    kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
    static_mu_array = 2 * kinetic_mu_array
    simulator.add_forcing_to(rigid_rod).using(
        AnisotropicFrictionalPlaneRigidBody,
        k=1.0,
        nu=1e0,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )

    return shearable_rods, rigid_rod
