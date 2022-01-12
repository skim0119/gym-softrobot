"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

from collections import defaultdict

from numba import njit
from tqdm import tqdm

import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from elastica._linalg import _batch_cross, _batch_matvec, _batch_matrix_transpose
from elastica._calculus import quadrature_kernel
from elastica.external_forces import inplace_addition

from gym_softrobot.utils.actuation.algorithms.algorithm import Algorithm
import gym_softrobot.utils.actuation.objects
# from objects.point import PointTarget
# from objects.cylinder import CylinderConstraint, CylinderTarget
from gym_softrobot.utils.actuation.rod_tools import (
    forward_path,
    backward_path,
    calculate_dilatation,
    sigma_to_shear,
    kappa_to_curvature,
    # calculate_potential_energy
)

from gym_softrobot.utils.actuation.actuations.muscles import TransverseMuscle
from gym_softrobot.utils.actuation.actuations.actuation import _force_induced_couple


@njit(cache=True)
def backward_path_2(dl, director, sigma, kappa, internal_force, internal_couple):

    blocksize = internal_force.shape[1]

    _, voronoi_dilatation = calculate_dilatation(sigma)
    shear = sigma_to_shear(sigma)
    curvature = kappa_to_curvature(kappa, voronoi_dilatation)

    for i in range(blocksize-1):
        internal_force[:, -1-i-1] = internal_force[:, -1-i] + (
            np.cross(curvature[:, -1-i], internal_force[:, -1-i])
            ) * dl
    nu_cross_n = _batch_cross(shear, internal_force)

    for i in range(blocksize-2):
        # maybe kappa need to be devided by voroni_dilatation?
        kappa_cross_m = np.cross(curvature[:, -1-i], internal_couple[:, -1-i])
        internal_couple[:, -1-i-1] = internal_couple[:, -1-i] + (
            kappa_cross_m
            + (nu_cross_n[:, -1-i] + nu_cross_n[:, -1-i-1])/2
            ) * dl
    pass

@njit(cache=True)
def backward_path_3(dl, director, sigma, ns, ms, n1, m1, internal_force, internal_couple):
    shear = np.zeros(sigma.shape)
    # print(shear.shape)
    shear[:, :] = sigma_to_shear(sigma)
    
    n = np.zeros(internal_force.shape)
    n[:, -1] = n1
    for i in range(n.shape[1]-1):
        n[:, -1-i-1] = n[:, -1-i] - (ns[:, -1-i]+ns[:, -1-i-1])*dl[-1-i]/2
    internal_force[:, :] = _batch_matvec(director, n)

    m = np.zeros(internal_force.shape)
    m[:, -1] = m1
    ms[:, :] = -_batch_cross(_batch_matvec(_batch_matrix_transpose(director), shear), n)
    for i in range(m.shape[1]-1):
        m[:, -1-i-1] = m[:, -1-i] - (ms[:, -1-i]+ms[:, -1-i-1])*dl[-1-i]/2
    internal_couple[:, :] = quadrature_kernel(_batch_matvec(director, m))[:, 1:-1]
    return

@njit(cache=True)
def muscle_to_strain(muscle_force, muscle_couple, sigma, kappa, shear_matrix, bend_matrix):

    dilatation, voronoi_dilatation = calculate_dilatation(sigma)

    sigma[0, :] = - muscle_force[0, :] / (shear_matrix[0, 0, :] / dilatation)
    sigma[1, :] = - muscle_force[1, :] / (shear_matrix[1, 1, :] / dilatation)
    sigma[2, :] = - muscle_force[2, :] / (shear_matrix[2, 2, :] / dilatation)

    kappa[0, :] = - muscle_couple[0, :] / (bend_matrix[0, 0, :] / voronoi_dilatation**3)
    kappa[1, :] = - muscle_couple[1, :] / (bend_matrix[1, 1, :] / voronoi_dilatation**3)
    kappa[2, :] = - muscle_couple[2, :] / (bend_matrix[2, 2, :] / voronoi_dilatation**3)
    pass

@njit(cache=True)
def update_muscle_actuation(
    activation, muscle_force, muscle_couple,
    sigma, kappa, internal_force, internal_couple,
    shear_matrix, bend_matrix, eta_activation, pre_activation
    ):

    blocksize = activation.shape[0]

    alpha_bend = np.zeros((3, blocksize))
    alpha = np.zeros((3, blocksize+1))

    for i in range(blocksize):
        for j in range(3):
            alpha_bend[1, i] -= internal_couple[j, i] * muscle_couple[j, i] / bend_matrix[j, j, i]
            alpha_bend[2, i] += (muscle_couple[j, i])**2 / bend_matrix[j, j, i]

    for i in range(blocksize+1):
        for j in range(3):
            alpha[1, i] -= internal_force[j, i] * muscle_force[j, i] / shear_matrix[j, j, i]
            alpha[2, i] += (muscle_force[j, i])**2 / shear_matrix[j, j, i]

    for i in range(blocksize):
        alpha_bend[1, i] += (alpha[1, i] + alpha[1, i+1]) / 2
        alpha_bend[2, i] += (alpha[2, i] + alpha[2, i+1]) / 2

    activation[:] += eta_activation * (-alpha_bend[2, :]*activation + alpha_bend[1, :])

    for i in range(blocksize):
        if activation[i] < 0:
            activation[i] = 0
        elif activation[i] > 1:
            activation[i] = 1
        else:
            pass

    pass

class ForwardBackwardMuscle(Algorithm):
    def __init__(self, rod, muscles, algo_config):
        Algorithm.__init__(self, rod, algo_config)
        self.target = algo_config.get('target', None)
        self.prev_target = None
        self.target = algo_config.get('target', None)
        self.prev_target = None

        self.muscles = muscles
        self.activations = []
        self.pre_activations = []
        for muscle in self.muscles:
            self.activations.append(muscle.activation.copy())
            self.pre_activations.append(muscle.activation.copy())
        self.eta = algo_config.get('eta', 1e-8)

    def update(self, update_weight):

        muscle_forces = np.zeros((3, self.n_elems))
        muscle_couples = np.zeros((3, self.n_elems-1))

        dilatation, _ = calculate_dilatation(self.sigma)
        for muscle, activation in zip(self.muscles, self.activations):
            muscle.set_activation(activation)
            muscle(self.rod, count_flag=False)
            if not (type(muscle) is TransverseMuscle):
                _force_induced_couple(
                    muscle.internal_forces,
                    muscle.muscle_radius_ratio * self.rod.radius / np.sqrt(dilatation),
                    muscle.internal_couples
                )
            inplace_addition(muscle_forces, muscle.internal_forces)
            inplace_addition(muscle_couples, muscle.internal_couples)
        
        # TODO: fix the area and second moment of area change
        muscle_to_strain(
            muscle_forces, muscle_couples,
            self.sigma, self.kappa,
            self.shear_matrix, self.bend_matrix
        )
        
        forward_path(self.dl, self.sigma, self.kappa, self.position, self.director)

        # if type(self.target) is PointTarget:
        #     terminal_cost_partial_end_position, terminal_cost_partial_end_director = (
        #         self.target.terminal_cost_partial_end_state(
        #             self.position[:, -1],
        #             end_director=self.director[:, :, -1]
        #         )
        #     )
        # else:
        #     pass

        # self.internal_force[:, -1] = - self.director[:, :, -1] @ terminal_cost_partial_end_position
        # if not (terminal_cost_partial_end_director == None):
        #     self.internal_couple[:, -1] = terminal_cost_partial_end_director


        midpoint_position = (self.position[:, :-1] + self.position[:, 1:]) / 2
        ns = np.zeros(midpoint_position.shape)
        ms = np.zeros(midpoint_position.shape)
        n1 = np.zeros(3)
        m1 = np.zeros(3)

        # for constraint in self.constraints:
        #     constraint.calculate_cost(midpoint_position, self.radii)
        #     ns[:, :] -= constraint.running_cost_partial_position(midpoint_position)

        # self.target.calculate_cost(self.position[:, -1], midpoint_position, self.radius)
        terminal_cost_partial_end_position, terminal_cost_partial_end_director = (
                self.target.terminal_cost_partial_end_state(
                    self.position[:, -1],
                    end_director=self.director[:, :, -1]
                )
            )
        n1[:] = - terminal_cost_partial_end_position

        if objects.object.Constraint in self.target.__class__.__mro__:
            self.target.running_cost(midpoint_position, self.radius)
            ns[:, :] -= self.target.running_cost_partial_position(midpoint_position)
        # print(self.dl)
        # print(ns.shape, self.sigma.shape)

        backward_path_3(self.dl, self.director, self.sigma, ns, ms, n1, m1, self.internal_force, self.internal_couple)

        # backward_path(self.dl, self.director, self.sigma, ns, ms, n1, m1, self.internal_force, self.internal_couple)

        dilatation, _ = calculate_dilatation(self.sigma)
        for i, muscle in enumerate(self.muscles):
            muscle.set_activation(np.ones(self.n_elems-1))
            muscle(self.rod, count_flag=False)
            if not (type(muscle) is TransverseMuscle):
                _force_induced_couple(
                    muscle.internal_forces,
                    muscle.muscle_radius_ratio * self.rod.radius / np.sqrt(dilatation),
                    muscle.internal_couples
                )

            update_muscle_actuation(
                self.activations[i],
                muscle.internal_forces, muscle.internal_couples,
                self.sigma, self.kappa,
                self.internal_force, self.internal_couple,
                self.shear_matrix, self.bend_matrix,
                self.eta, #max(1e-10/update_weight, 1e-15)
                self.pre_activations[i]
            )

        return update_weight+1

    def run(self, continuum_target_cost_weight=None, plot_flag=False, iter_number=100000):
        print("Running algorithm with target:", self.target)
        update_weight = 1
        for _ in tqdm(range(iter_number)):
            update_weight = self.update(update_weight)
        return
