__doc__="""
Module contains elastica interface to create octopus model.
This model specifically create 8-arm octopus with each arm controlled by muscle model.
Suposedly, this model represend simplified control of arm coordination.
"""

from typing import Optional, Tuple
import json

from copy import deepcopy

import numpy as np

from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica.experimental.interaction import AnisotropicFrictionalPlaneRigidBody

from gym_softrobot.utils.custom_elastica.joint import FixedJoint2Rigid
from gym_softrobot.utils.custom_elastica.constraint import BodyBoundaryCondition
from gym_softrobot.utils.actuation.forces.drag_force import DragForce
from gym_softrobot.utils.actuation.actuations.muscles.longitudinal_muscle import (
    LongitudinalMuscle,
)
from gym_softrobot.utils.actuation.actuations.muscles.transverse_muscle import (
    TransverseMuscle,
)
from gym_softrobot.utils.actuation.actuations.muscles.muscle import ApplyMuscle

from gym_softrobot.envs.octopus.build import create_es_muscle_layers

from scipy.spatial.transform import Rotation as Rot

# default parameters
ARM_MATERIAL = {
    # Arm properties
    "density": 1000.0,
    "youngs_modulus": 1e4,
    "shear_modulus": 1e4 / (1.0+0.5), # 0.5 Poisson Ratio
    "nu": 0.20,
    "nu_scale": 1e-2,
}
DEFAULT_SCALE_LENGTH = {
    # Arm length scale
    "base_length": 0.25,
    "base_radius": 0.013,
    "tip_radius" : 0.0042, # https://www.liebertpub.com/doi/10.1089/soro.2019.0082
    "head_radius": 0.04,
}
HEAD_PROPERTIES = {
    # Head length scale
    "head_density": 50.0,
    # Head properties
    "body_arm_k": 1e6,
    "body_arm_kt": 1e2,
    "body_arm_nu": 1e-3,
}

def build_arm(n_elem:int, start:np.ndarray, direction:np.ndarray, normal:np.ndarray) -> CosseratRod:
    """ Set up properties """
    arm_material = deepcopy(ARM_MATERIAL)
    arm_material["n_elements"] = n_elem
    arm_material["start"] = start
    arm_material["direction"] = direction
    arm_material["normal"] = normal
    arm_material["base_length"] = DEFAULT_SCALE_LENGTH["base_length"]
    arm_material["base_radius"] = np.linspace(
            DEFAULT_SCALE_LENGTH["base_radius"], 
            DEFAULT_SCALE_LENGTH["tip_radius"],
            n_elem)
    arm_material["nu"] *= ((arm_material["base_radius"]/DEFAULT_SCALE_LENGTH["base_radius"])**2.0)*arm_material["nu_scale"]

    rod = CosseratRod.straight_rod(**arm_material)
    return rod

def build_octopus(simulator, n_elem:int=40):
    """
    build_octopus

    Parameters
    ----------
    simulator : PyElastica Simulator
    n_elem : int
    """

    """ Configuration"""
    n_arm = 8

    """ Set up an arm """
    L0 = DEFAULT_SCALE_LENGTH['base_length']
    r0 = DEFAULT_SCALE_LENGTH['base_radius']
    head_radius = DEFAULT_SCALE_LENGTH['head_radius']

    # Levi (2015) - Figure 3B 
    shearable_rods=[]  # arms
    angles_offset = 45.0 / 2
    angles = [angles_offset+45*arm_i for arm_i in range(n_arm)]
    for angle in angles:
        # Rotate director
        rot = Rot.from_euler('z', angle, degrees=True)
        arm_pos = rot.apply([head_radius, 0.0, 0.0])
        arm_dir = rot.apply([1.0, 0.0, 0.0])
        arm_normal = np.array([0.0, 0.0, 1.0]) 
        # Build Arm
        rod = build_arm(n_elem, arm_pos, arm_dir, arm_normal)
        shearable_rods.append(rod)
        simulator.append(rod)

    """ Add head """
    start = np.zeros((3,)); start[2] = -r0*2
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    density = HEAD_PROPERTIES['head_density']
    rigid_rod = Cylinder(start, direction, normal, r0 * 2, head_radius, density)
    simulator.append(rigid_rod)

    """ Constraint body """
    simulator.constrain(rigid_rod).using(
        # Upright rigid rod need restoration force/torque against the floor
        BodyBoundaryCondition, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    """ Set up boundary conditions """
    _k = HEAD_PROPERTIES['body_arm_k']
    _kt = HEAD_PROPERTIES['body_arm_kt']
    _nu = HEAD_PROPERTIES['body_arm_nu']
    for arm_i in range(n_arm):
        simulator.connect(
                    first_rod=rigid_rod,
                    second_rod=shearable_rods[arm_i],
                    first_connect_idx=-1,
                    second_connect_idx=0
                ).using(
                    FixedJoint2Rigid,
                    k=_k,
                    nu=_nu,
                    kt=_kt,
                    angle=angles[arm_i],
                    radius=head_radius
                )

    """ Add drag force """
    """
    dl = base_length_of_arm / n_elem # TODO: base length of arm
    fluid_factor = 1
    r_bar = (radius_base + radius_tip) / 2
    sea_water_dentsity = 1022
    c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
    c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor
    
    simulator.add_forcing_to(self.shearable_rod).using(
        DragForce,
        rho_environment=sea_water_dentsity,
        c_per=c_per,
        c_tan=c_tan,
        system=self.shearable_rod,
        step_skip=self.step_skip,
        callback_params=self.rod_parameters_dict
    )
    """

    """ Add muscle actuation """
    tm_activations = []
    for i in range(n_arm):
        muscle_layers = create_es_muscle_layers(
            shearable_rods[i].radius,
            DEFAULT_SCALE_LENGTH["base_radius"]
        )
        simulator.add_forcing_to(shearable_rods[i]).using(
            ApplyMuscle,
            muscles=muscle_layers,
            step_skip=10000, # Not relavent
            callback_params_list=[],
        )
        tm_activations.append(muscle_layers[2]) # 2 for TM

    return shearable_rods, rigid_rod, tm_activations


def build_octopus_muscles(simulator, n_elem: int = 40):
    """
    build_octopus_muscles

    Parameters
    ----------
    simulator : PyElastica Simulator
    n_elem : int
    """

    """ Configuration"""
    n_arm = 8

    """ Set up an arm """
    L0 = DEFAULT_SCALE_LENGTH['base_length']
    r0 = DEFAULT_SCALE_LENGTH['base_radius']
    head_radius = DEFAULT_SCALE_LENGTH['head_radius']

    # Levi (2015) - Figure 3B
    shearable_rods = []  # arms
    angles_offset = 45.0 / 2
    angles = [angles_offset + 45 * arm_i for arm_i in range(n_arm)]
    for angle in angles:
        # Rotate director
        rot = Rot.from_euler('z', angle, degrees=True)
        arm_pos = rot.apply([head_radius, 0.0, 0.0])
        arm_dir = rot.apply([1.0, 0.0, 0.0])
        arm_normal = np.array([0.0, 0.0, 1.0])
        # Build Arm
        rod = build_arm(n_elem, arm_pos, arm_dir, arm_normal)
        shearable_rods.append(rod)
        simulator.append(rod)

    """ Add head """
    start = np.zeros((3,));
    start[2] = -r0 * 2
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    density = HEAD_PROPERTIES['head_density']
    rigid_rod = Cylinder(start, direction, normal, r0 * 2, head_radius, density)
    simulator.append(rigid_rod)

    """ Constraint body """
    simulator.constrain(rigid_rod).using(
        # Upright rigid rod need restoration force/torque against the floor
        BodyBoundaryCondition, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    """ Set up boundary conditions """
    _k = HEAD_PROPERTIES['body_arm_k']
    _kt = HEAD_PROPERTIES['body_arm_kt']
    _nu = HEAD_PROPERTIES['body_arm_nu']
    for arm_i in range(n_arm):
        simulator.connect(
            first_rod=rigid_rod,
            second_rod=shearable_rods[arm_i],
            first_connect_idx=-1,
            second_connect_idx=0
        ).using(
            FixedJoint2Rigid,
            k=_k,
            nu=_nu,
            kt=_kt,
            angle=angles[arm_i],
            radius=head_radius
        )

    """ Add drag force """
    """
    dl = base_length_of_arm / n_elem # TODO: base length of arm
    fluid_factor = 1
    r_bar = (radius_base + radius_tip) / 2
    sea_water_dentsity = 1022
    c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
    c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor

    simulator.add_forcing_to(self.shearable_rod).using(
        DragForce,
        rho_environment=sea_water_dentsity,
        c_per=c_per,
        c_tan=c_tan,
        system=self.shearable_rod,
        step_skip=self.step_skip,
        callback_params=self.rod_parameters_dict
    )
    """

    """ Add muscle actuation """
    tm_activations = []
    for i in range(n_arm):
        muscle_layers = create_es_muscle_layers(
            shearable_rods[i].radius,
            DEFAULT_SCALE_LENGTH["base_radius"]
        )
        simulator.add_forcing_to(shearable_rods[i]).using(
            ApplyMuscle,
            muscles=muscle_layers,
            step_skip=10000,  # Not relavent
            callback_params_list=[],
        )
        tm_activations.append(muscle_layers)  # 2 for TM

    return shearable_rods, rigid_rod, tm_activations

def build_two_arms(simulator, n_elem: int = 40):
    """
    build_two_arms

    Parameters
    ----------
    simulator : PyElastica Simulator
    n_elem : int
    """

    """ Configuration"""
    n_arm = 2

    """ Set up an arm """
    L0 = DEFAULT_SCALE_LENGTH['base_length']
    r0 = DEFAULT_SCALE_LENGTH['base_radius']
    head_radius = DEFAULT_SCALE_LENGTH['head_radius']

    # Levi (2015) - Figure 3B
    shearable_rods = []  # arms
    angles_offset = 90.0
    angles = [angles_offset + 180.0 * arm_i for arm_i in range(n_arm)]
    for angle in angles:
        # Rotate director
        rot = Rot.from_euler('z', angle, degrees=True)
        arm_pos = rot.apply([head_radius, 0.0, 0.0])
        arm_dir = rot.apply([1.0, 0.0, 0.0])
        arm_normal = np.array([0.0, 0.0, 1.0])
        # Build Arm
        rod = build_arm(n_elem, arm_pos, arm_dir, arm_normal)
        shearable_rods.append(rod)
        simulator.append(rod)

    """ Add head """
    start = np.zeros((3,));
    start[2] = -r0 * 2
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    density = HEAD_PROPERTIES['head_density']
    rigid_rod = Cylinder(start, direction, normal, r0 * 2, head_radius, density)
    simulator.append(rigid_rod)

    """ Constraint body """
    simulator.constrain(rigid_rod).using(
        # Upright rigid rod need restoration force/torque against the floor
        BodyBoundaryCondition, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    """ Set up boundary conditions """
    _k = HEAD_PROPERTIES['body_arm_k']
    _kt = HEAD_PROPERTIES['body_arm_kt']
    _nu = HEAD_PROPERTIES['body_arm_nu']
    for arm_i in range(n_arm):
        simulator.connect(
            first_rod=rigid_rod,
            second_rod=shearable_rods[arm_i],
            first_connect_idx=-1,
            second_connect_idx=0
        ).using(
            FixedJoint2Rigid,
            k=_k,
            nu=_nu,
            kt=_kt,
            angle=angles[arm_i],
            radius=head_radius
        )

    """ Add drag force """
    """
    dl = base_length_of_arm / n_elem # TODO: base length of arm
    fluid_factor = 1
    r_bar = (radius_base + radius_tip) / 2
    sea_water_dentsity = 1022
    c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
    c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor

    simulator.add_forcing_to(self.shearable_rod).using(
        DragForce,
        rho_environment=sea_water_dentsity,
        c_per=c_per,
        c_tan=c_tan,
        system=self.shearable_rod,
        step_skip=self.step_skip,
        callback_params=self.rod_parameters_dict
    )
    """

    """ Add muscle actuation """
    arm_activations = []
    for i in range(n_arm):
        muscle_layers = create_es_muscle_layers(
            shearable_rods[i].radius,
            DEFAULT_SCALE_LENGTH["base_radius"]
        )
        simulator.add_forcing_to(shearable_rods[i]).using(
            ApplyMuscle,
            muscles=muscle_layers,
            step_skip=10000,  # Not relavent
            callback_params_list=[],
        )
        arm_activations.append(muscle_layers)  # 2 for TM

    return shearable_rods, rigid_rod, arm_activations
