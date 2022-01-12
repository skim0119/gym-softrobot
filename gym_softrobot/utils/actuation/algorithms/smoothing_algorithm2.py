"""
Created on Apr. 27, 2021
@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function

from collections import defaultdict

from numba import njit
from numpy.core.shape_base import block
from tqdm import tqdm

import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors

from elastica._linalg import _batch_cross, _batch_matvec, _batch_matrix_transpose
from elastica._calculus import quadrature_kernel
from elastica._rotations import _rotate, _inv_rotate
from elastica.external_forces import inplace_addition

from gym_softrobot.utils.actuation.algorithms.algorithm import Algorithm
from gym_softrobot.utils.actuation.rod_tools import (
    forward_path,
    calculate_dilatation,
    sigma_to_shear,
    kappa_to_curvature,
    _lab_to_material,
    _material_to_lab,
    _batch_cross,
    # _trapezoidal,
    average
)

class ForwardBackwardSmooth(Algorithm):
    def __init__(self, rod, algo_config, data):
        Algorithm.__init__(self, rod, algo_config)

        self.argument_rotational = np.zeros((3, rod.n_elems-1))
        self.argument_translational = np.zeros((3, rod.n_elems))
        self.costate_rotational = np.zeros((3, rod.n_elems-1))
        self.costate_translational = np.zeros((3, rod.n_elems))

        self.data = data
        
        self.s_position_jump_index = np.full(rod.n_elems+1, -1, dtype=int)
        self.s_director_jump_index = np.full(rod.n_elems, -1, dtype=int)
        for index, s in enumerate(data.s_position):
            for s_index in range(self.s_position.shape[0]):
                if s_index == 0:
                    s_range_plus = self.s_position[1]/2
                    s_range_minus = 0
                elif s_index == self.s_position.shape[0]-1:
                    s_range_plus = 0
                    s_range_minus = (self.s_position[-1]-self.s_position[-2])/2
                else:
                    s_range_plus = (self.s_position[s_index+1]-self.s_position[s_index])/2
                    s_range_minus = (self.s_position[s_index]-self.s_position[s_index-1])/2
                s_plus = self.s_position[s_index] + s_range_plus
                s_minus = self.s_position[s_index] - s_range_minus
                
                if (s_plus-s)*(s-s_minus) >= -1e-5:
                    self.s_position_jump_index[s_index] = index
                    break        
        self.director_flag = data.director_flag
        if self.director_flag:
            for index, s in enumerate(data.s_director):
                for s_index in range(self.s_director.shape[0]):
                    if s_index == 0:
                        s_range_plus = self.s_director[1]/2
                        s_range_minus = 0
                    elif s_index == self.s_director.shape[0]-1:
                        s_range_plus = 0
                        s_range_minus = (self.s_director[-1]-self.s_director[-2])/2
                    else:
                        s_range_plus = (self.s_director[s_index+1]-self.s_director[s_index])/2
                        s_range_minus = (self.s_director[s_index]-self.s_director[s_index-1])/2
                    s_plus = self.s_director[s_index] + s_range_plus
                    s_minus = self.s_director[s_index] - s_range_minus
                    
                    if (s_plus-s)*(s-s_minus) >= -1e-5:
                        self.s_director_jump_index[s_index] = index
                        break
            self.update = self.update_with_director_jump
        else:
            self.update = self.update_without_director_jump

    def calculate_cost(self,):
        cost = 0
        cost += self.energy_cost(
            self.dl, self.shear, self.kappa, self.rest_shear, self.rest_kappa,
            self.shear_matrix, self.bend_matrix,
        )
        cost += self.argument_cost(
            self.dl, self.argument_rotational, self.argument_translational, self.config.argument_weight
        )
        cost += self.data_position_cost(
            self.position,
            self.s_position_jump_index,
            self.data.noisy_position,
            self.config.data_deviation_weight_cost_position
        )
        if self.director_flag:
            cost += self.data_director_cost(
                self.director,
                self.s_director_jump_index,
                self.data.noisy_director,
                self.config.data_deviation_weight_cost_director
            )

        return cost

    @staticmethod
    @njit(cache=True)
    def energy_cost(
        dl, shear, kappa, rest_shear, rest_kappa,
        shear_matrix, bend_matrix
    ):
        cost = 0
        for n in range(dl.shape[0]):
            for i in range(3):
                cost += (shear[i, n]-rest_shear[i, n])**2 * shear_matrix[i, i, n] * dl[i]
        voronoi_dl = (dl[:-1] + dl[1:]) / 2
        for n in range(voronoi_dl.shape[0]):
            for i in range(3):
                cost += (kappa[i, n]-rest_kappa[i, n])**2 * bend_matrix[i, i, n] * voronoi_dl[i]
        return 0.5*cost

    @staticmethod
    @njit(cache=True)
    def argument_cost(
        dl, argument_rotational, argument_translational, argument_weight
    ):
        cost = 0
        for n in range(argument_translational.shape[1]):
            for i in range(argument_translational.shape[0]):
                cost += argument_translational[i, n] ** 2 * dl[n]
        voronoi_dl = (dl[:-1] + dl[1:]) / 2
        for n in range(argument_rotational.shape[1]):
            for i in range(argument_rotational.shape[0]):
                cost += argument_rotational[i, n] ** 2 * voronoi_dl[n]
        return 0.5*cost*argument_weight

    @staticmethod
    @njit(cache=True)
    def data_position_cost(
        position,
        s_position_jump_index,
        data_position,
        data_deviation_weight_cost_position,
    ):
        cost = 0
        for n, index in enumerate(s_position_jump_index):
            if index != -1:
                cost += data_deviation_weight_cost_position * position_cost(position[:, n], data_position[:, index])
        return cost

    @staticmethod
    @njit(cache=True)
    def data_director_cost(
        director,
        s_director_jump_index,
        data_director,
        data_deviation_weight_cost_director
    ):
        cost = 0
        for n, index in enumerate(s_director_jump_index):
            if index != -1:
                cost += data_deviation_weight_cost_director * director_cost(director[:, :, n], data_director[:, :, index])
        return cost

    def update_strain_and_forward_path(self,):
        self.update_argument(
            self.argument_translational, self.argument_rotational,
            self.costate_translational, self.costate_rotational,
            self.config.argument_weight, self.config.step_size
        )
        self.forward_path_strain(
            self.dl, self.shear, self.kappa,
            self.argument_rotational, self.argument_translational,
        )
        forward_path(
            self.dl, self.shear, self.kappa, 
            self.position, self.director,
        )

    def update_without_director_jump(self, ):
        self.backward_path_without_director_jump(
            self.dl, self.position, self.director, self.shear, self.kappa,
            self.internal_force, self.internal_couple,
            self.s_position_jump_index, self.data.noisy_position,
            self.config.data_deviation_weight_cost_position,
        )
        self.costate_backward_path(
            self.dl, self.kappa, self.shear,
            self.rest_kappa, self.rest_shear,
            self.bend_matrix, self.shear_matrix,
            self.costate_rotational, self.costate_translational,
            self.internal_couple, self.internal_force,
        )
        self.update_strain_and_forward_path()

    def update_with_director_jump(self, ):
        self.backward_path_with_director_jump(
            self.dl, self.position, self.director, self.shear, self.kappa,
            self.internal_force, self.internal_couple,
            self.s_position_jump_index, self.data.noisy_position,
            self.config.data_deviation_weight_cost_position,
            self.s_director_jump_index, self.data.noisy_director,
            self.config.data_deviation_weight_cost_director
        )
        self.costate_backward_path(
            self.dl, self.kappa, self.shear,
            self.rest_kappa, self.rest_shear,
            self.bend_matrix, self.shear_matrix,
            self.costate_rotational, self.costate_translational,
            self.internal_couple, self.internal_force,
        )
        self.update_strain_and_forward_path()

    def run(self, iter_number=1_000_000):
        print("Running forward-backward algorithm to smooth the data")
        cost = np.zeros(iter_number)
        for k in tqdm(range(iter_number)):
            self.update()
            cost[k] = self.calculate_cost()
        return cost

    @staticmethod
    @njit(cache=True)
    def forward_path_strain(
        dl, shear, kappa,
        argument_rotational, argument_translational,
    ):
        blocksize = dl.shape[0]
        for n in range(blocksize-2):
            kappa[:, n+1] = kappa[:, n] + argument_rotational[:, n]*dl[n]
            shear[:, n+1] = shear[:, n] + argument_translational[:, n]*dl[n]
        shear[:, -1] = shear[:, -2] + argument_translational[:, -2]*dl[-2]

    @staticmethod
    @njit(cache=True)
    def update_argument(
        argument_translational, argument_rotational,
        costate_translational, costate_rotational,
        argument_weight, step_size
    ):
        for n in range(argument_rotational.shape[1]):
            argument_rotational[:, n] += (
                step_size * (costate_rotational[:, n] - argument_weight*argument_rotational[:, n])
            )
            argument_translational[:, n] += (
                step_size * (costate_translational[:, n] - argument_weight*argument_translational[:, n])
            )
        argument_translational[:, -1] += (
            step_size * (costate_translational[:, -1] - argument_weight*argument_translational[:, -1])
        )

    @staticmethod
    @njit(cache=True)
    def costate_backward_path(
        dl, kappa, shear,
        rest_kappa, rest_shear,
        bend_matrix, shear_matrix,
        costate_rotational, costate_translational,
        internal_couple, internal_force,
    ):
        blocksize = dl.shape[0]
        for i in range(3):
            costate_rotational[i, -1] = - (
                -internal_couple[i, -1] + bend_matrix[i, i, -1]*(kappa[i, -1]-rest_kappa[i, -1])
            )*dl[-1]
            costate_translational[i, -1] = - (
                -internal_force[i, -1] + shear_matrix[i, i, -1]*(shear[i, -1]-rest_shear[i, -1])
            )*dl[-1]

        for n in range(blocksize-2):
            for i in range(3):
                costate_rotational[i, -1-n-1] = costate_rotational[i, -1-n] - (
                    -internal_couple[i, -1-n-1] + bend_matrix[i, i, -1-n-1]*(kappa[i, -1-n-1]-rest_kappa[i, -1-n-1])
                )*dl[-1-n-1]
                costate_translational[i, -1-n-1] = costate_translational[i, -1-n] - (
                    -internal_force[i, -1-n-1] + shear_matrix[i, i, -1-n-1]*(shear[i, -1-n-1]-rest_shear[i, -1-n-1])
                )*dl[-1-n-1]
        
        for i in range(3):
            costate_translational[i, 0] = costate_translational[i, 1] - (
                -internal_force[i, 0] + shear_matrix[i, i, 0]*(shear[i, 0]-rest_shear[i, 0])
            )*dl[0]
        pass

    @staticmethod
    @njit(cache=True)
    def backward_path_without_director_jump(
        dl, position, director, shear, kappa,
        internal_force, internal_couple,
        s_position_jump_index, data_position, data_deviation_weight_cost_position,
    ):
        # calculate internal force
        internal_force_lab_frame = (
            calculate_internal_force_with_jumping_condition(
                position, director, internal_force,
                s_position_jump_index, data_position, data_deviation_weight_cost_position,
            )
        )

        # calculate internal couple
        internal_couple_lab_frame = (
            calculate_internal_couple_without_jumping_condition(
                dl, director, shear, internal_force_lab_frame, internal_couple,
            )
        )
    
    @staticmethod
    @njit(cache=True)
    def backward_path_with_director_jump(
        dl, position, director, shear, kappa,
        internal_force, internal_couple,
        s_position_jump_index, data_position, data_deviation_weight_cost_position,
        s_director_jump_index, data_director, data_deviation_weight_cost_director,
    ):
        # calculate internal force
        internal_force_lab_frame = (
            calculate_internal_force_with_jumping_condition(
                position, director, internal_force,
                s_position_jump_index, data_position, data_deviation_weight_cost_position,
            )
        )

        # calculate internal couple
        internal_couple_lab_frame = (
            calculate_internal_couple_with_jumping_condition(
                dl, director, shear, internal_force_lab_frame, internal_couple,
                s_director_jump_index, data_director, data_deviation_weight_cost_director,
            )
        )

@njit(cache=True)
def calculate_internal_force_with_jumping_condition(
    position, director, internal_force,
    s_position_jump_index, data_position, data_deviation_weight_cost_position,
):
    blocksize = internal_force.shape[1]
    internal_force_lab_frame = np.zeros((3, blocksize))
    # print(internal_force_lab_frame[:,0])
    # calculate internal force in lab frame
    for i in range(blocksize-1):
        # this is the jumping condition
        if s_position_jump_index[-1-i] != -1:
            internal_force_lab_frame[:, -1-i] -= data_deviation_weight_cost_position * (
                position[:, -1-i] - data_position[:, s_position_jump_index[-1-i]]
            )
        internal_force_lab_frame[:, -1-i-1] = internal_force_lab_frame[:, -1-i].copy()

    internal_force[:, :] = _lab_to_material(director, internal_force_lab_frame)
    return internal_force_lab_frame

@njit(cache=True)
def calculate_internal_couple_with_jumping_condition(
    dl, director, shear, internal_force_lab_frame, internal_couple,
    s_director_jump_index, data_director, data_deviation_weight_cost_director,
):
    blocksize = internal_couple.shape[1]+1
    internal_couple_lab_frame = np.zeros((3, blocksize))

    ms = - _batch_cross(_material_to_lab(director, shear), internal_force_lab_frame)
    for i in range(blocksize-1):
        # this is the jumping condition
        if s_director_jump_index[-1-i] != -1:
            internal_couple_lab_frame[:, -1-i] += data_deviation_weight_cost_director * (
                difference_between_directors(
                    director[:, :, -1-i], data_director[:, :, s_director_jump_index[-1-i]]
                )
            )

        internal_couple_lab_frame[:, -1-i-1] = (
            internal_couple_lab_frame[:, -1-i] - (ms[:, -1-i]+ms[:, -1-i-1])*dl[-1-i]/2
        )

    internal_couple[:, :] = couple_in_lab_transform_to_material(
        internal_couple_lab_frame, director
    )

    return internal_couple_lab_frame

@njit(cache=True)
def calculate_internal_couple_without_jumping_condition(
    dl, director, shear, internal_force_lab_frame, internal_couple,
):
    blocksize = internal_couple.shape[1]+1
    internal_couple_lab_frame = np.zeros((3, blocksize))

    ms = - _batch_cross(_material_to_lab(director, shear), internal_force_lab_frame)
    for i in range(blocksize-1):
        internal_couple_lab_frame[:, -1-i-1] = (
            internal_couple_lab_frame[:, -1-i] - (ms[:, -1-i]+ms[:, -1-i-1])*dl[-1-i]/2
        )

    internal_couple[:, :] = couple_in_lab_transform_to_material(
        internal_couple_lab_frame, director
    )
    return internal_couple_lab_frame

@njit(cache=True)
def difference_between_directors(director1, director2):
    difference = _mat1mat2t(director1, director2) - _mat1mat2t(director2, director1)
    return _matTvec(director1, _vec(difference))

@njit(cache=True)
def couple_in_lab_transform_to_material(internal_couple_lab_frame, director):
    blocksize = internal_couple_lab_frame.shape[1]
    internal_couple = np.zeros((3, blocksize-1))
    for i in range(blocksize-1):
        middle_couple = (internal_couple_lab_frame[:, i] + internal_couple_lab_frame[:, i+1]) / 2
        middle_director = calculate_middle_director(director[:, :, i:i+2])
        internal_couple[:, i] = _matvec(middle_director, middle_couple)
    return internal_couple

@njit(cache=True)
def calculate_middle_director(directors):
    vector = _inv_rotate(directors)
    return _rotate(directors[:, :, 0:1], 0.5, vector)[:, :, 0]

@njit(cache=True)
def _matvec(matrix, vec):
    result = np.zeros(3,)
    for i in range(3):
        for j in range(3):
            result[i] += matrix[i][j] * vec[j]
    return result

@njit(cache=True)
def _matTvec(matrix, vec):
    result = np.zeros(3,)
    for i in range(3):
        for j in range(3):
            result[i] += matrix[j][i] * vec[j]
    return result

@njit(cache=True)
def _mat1mat2t(matrix1, matrix2):
    result = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i, j] += matrix1[i][k] * matrix2[j][k]
    return result

@njit(cache=True)
def _vec(skew_matrix):
    return np.array([
        -skew_matrix[1, 2],
        skew_matrix[0, 2],
        -skew_matrix[0,1]
    ])

@njit(cache=True)
def position_cost(position1, position2):
    cost = 0
    for i in range(3):
        cost += (position1[i]-position2[i])**2
    cost = 0.5*np.sqrt(cost)
    return cost

@njit(cache=True)
def director_cost(director1, director2):
    identity = np.eye(3)
    matrix = identity - _mat1mat2t(director1, director2)
    cost = np.trace(_mat1mat2t(matrix, matrix))
    return cost
