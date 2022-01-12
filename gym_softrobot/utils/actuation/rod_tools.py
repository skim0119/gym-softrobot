"""
Created on Oct. 19, 2020
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit, jit

from elastica._linalg import _batch_matvec, _batch_cross, _batch_norm, _batch_matrix_transpose
from elastica._calculus import quadrature_kernel, difference_kernel

from elastica.external_forces import NoForces

@njit(cache=True)
def inverse_rigidity_matrix(matrix):
    inverse_matrix = np.zeros(matrix.shape)
    for i in range(inverse_matrix.shape[2]):
        inverse_matrix[:, :, i] = np.linalg.inv(matrix[:, :, i])
    return inverse_matrix
    
def strain_to_shear_and_curvature(strain):
    n_elems = int((strain.size + 3) / 6)
    shear = np.zeros((3, n_elems))
    curvature = np.zeros((3, n_elems-1))
    index = 0
    for i in range(3):
        shear[i, :] = strain[index:index+n_elems]
        index += n_elems
    for i in range(3):
        curvature[i] = strain[index:index+n_elems-1]
        index += (n_elems-1)
    return shear, curvature

def shear_and_curvature_to_strain(shear, curvature):
    strain = np.zeros(shear.size + curvature.size)
    n_elems = shear.shape[1]
    index = 0
    for i in range(3):
        strain[index:index+n_elems] = shear[i, :]
        index += n_elems
    for i in range(3):
        strain[index:index+n_elems-1] = curvature[i]
        index += (n_elems-1)
    return strain

@njit(cache=True)
def _lab_to_material(directors, lab_vectors):
    return _batch_matvec(directors, lab_vectors)

@njit(cache=True)
def _material_to_lab(directors, material_vectors):

    blocksize = material_vectors.shape[1]
    output_vector = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                output_vector[i, k] += (
                    directors[j, i, k] * material_vectors[j, k]
                )

    return output_vector

@njit(cache=True)
def average(vector_collection):

    blocksize = vector_collection.shape[1]-1
    output_vector = np.zeros((3, blocksize))

    for k in range(blocksize):
        for i in range(3):
            output_vector[i, k] = (vector_collection[i, k]+vector_collection[i, k+1])/2

    return output_vector

@njit(cache=True)
def average1D(vector_collection):

    blocksize = vector_collection.shape[0]-1
    output_vector = np.zeros(blocksize)

    for k in range(blocksize):
        output_vector[k] = (vector_collection[k]+vector_collection[k+1])/2
    return output_vector


@njit(cache=True)
def distance(position_collection, target_position):
    blocksize = position_collection.shape[1]
    distance_collection = np.zeros(blocksize)
    for k in range(blocksize):
        distance_collection[k] = (
            (position_collection[0, k]-target_position[0])**2 +
            (position_collection[1, k]-target_position[1])**2 +
            (position_collection[2, k]-target_position[2])**2
        )**0.5
    return distance_collection

@njit(cache=True)
def forward_path(dl, sigma, kappa, position_collection, director_collection, r0=np.zeros(3), Q0=np.array([[0,0,-1],[0,1,0],[1,0,0]])):
    _, voronoi_dilatation = calculate_dilatation(sigma)
    shear = sigma_to_shear(sigma)
    curvature = kappa_to_curvature(kappa, voronoi_dilatation)

    # position_collection[:,0] = r0
    # director_collection[:,:,0] = Q0
    for i in range(kappa.shape[1]):
        # next_position(
        #     director_collection[:, :, i],
        #     (sigma[:, i]+np.array([0, 0, 1]))*dl,
        #     position_collection[:, i:i+2]
        #     )
        next_position(
            director_collection[:, :, i],
            shear[:, i] * dl,
            position_collection[:, i:i+2]
            )
        # next_director(
        #     voronoi_dilatation[i]*curvature[:, i]*dl,
        #     director_collection[:, :, i:i+2]
        #     )
        # next_director(
        #     kappa[:, i] * dl,
        #     director_collection[:, :, i:i+2]
        #     )
        next_director(
            kappa[:, i] * dl,
            director_collection[:, :, i:i+2]
            )

    next_position(
        director_collection[:, :, -1],
        shear[:, -1] * dl,
        position_collection[:, -2:]
        )
    pass

@njit(cache=True)
def next_position(director, delta, positions):
    positions[:, 1] = positions[:, 0]
    for index_i in range(3):
        for index_j in range(3):
            positions[index_i, 1] += director[index_j, index_i] * delta[index_j]
    return

@njit(cache=True)
def next_director(rotation, directors):
    Rotation = get_rotation_matrix(rotation)
    for index_i in range(3):
        for index_j in range(3):
            directors[index_i, index_j, 1] = 0
            for index_k in range(3):
                directors[index_i, index_j, 1] += (
                    Rotation[index_k, index_i] * directors[index_k, index_j, 0]
                )
    return

@njit(cache=True)
def get_rotation_matrix(rotation):
    Rotation = np.identity(3)
    angle = np.sqrt(rotation[0]**2+rotation[1]**2+rotation[2]**2)
    if angle == 0:
        return Rotation
    axis = rotation/angle
    K = np.zeros((3, 3))
    K[2, 1] = axis[0]
    K[1, 2] = - axis[0]
    K[0, 2] = axis[1]
    K[2, 0] = - axis[1]
    K[1, 0] = axis[2]
    K[0, 1] = - axis[2]

    K2 = np.zeros((3, 3))
    K2[0, 0] = -(axis[1]*axis[1]+axis[2]*axis[2])
    K2[1, 1] = -(axis[2]*axis[2]+axis[0]*axis[0])
    K2[2, 2] = -(axis[0]*axis[0]+axis[1]*axis[1])
    K2[0, 1] = axis[0]*axis[1]
    K2[1, 0] = axis[0]*axis[1]
    K2[0, 2] = axis[0]*axis[2]
    K2[2, 0] = axis[0]*axis[2]
    K2[1, 2] = axis[1]*axis[2]
    K2[2, 1] = axis[1]*axis[2]

    Rotation[:, :] += np.sin(angle)*K + (1-np.cos(angle))*K2
    return Rotation

@njit(cache=True)
def backward_path(dl, director, shear, ns, ms, n1, m1, internal_force, internal_couple):
    n = np.zeros(internal_force.shape)
    n[:, -1] = n1
    for i in range(n.shape[1]-1):
        n[:, -1-i-1] = n[:, -1-i] - (ns[:, -1-i]+ns[:, -1-i-1])*dl/2
    internal_force[:, :] = _batch_matvec(director, n)

    m = np.zeros(internal_force.shape)
    m[:, -1] = m1
    ms[:, :] = -_batch_cross(_batch_matvec(_batch_matrix_transpose(director), shear), n)
    for i in range(m.shape[1]-1):
        m[:, -1-i-1] = m[:, -1-i] - (ms[:, -1-i]+ms[:, -1-i-1])*dl/2
    internal_couple[:, :] = quadrature_kernel(_batch_matvec(director, m))[:, 1:-1]
    return

# @njit(cache=True)
# def update_deformation(internal_force, internal_couple, shear_matrix, bend_matrix, 
#             step_strain, step_curvature, strain, curvature, next_strain, next_curvature):
#     dilatation, voronoi_dilatation = calculate_dilatation(strain)
#     intrinsic_strain = np.array([0, 0, 1])
#     for i in range(next_strain.shape[1]):
#         for index_i in range(3):
#             next_strain[index_i, i] = strain[index_i, i] \
#                 + step_strain[index_i, i] * (internal_force[index_i, i] - (strain[index_i, i] - intrinsic_strain[index_i])*(shear_matrix[index_i, index_i, i]/dilatation[i]))
#         if next_strain[2, i] < 0:
#             next_strain[2, i] = 0

#     for i in range(next_curvature.shape[1]):
#         for index_i in range(3):
#             next_curvature[index_i, i] = curvature[index_i, i] \
#                 + step_curvature[index_i, i] * (internal_couple[index_i, i] - curvature[index_i, i]*(bend_matrix[index_i, index_i, i]/(voronoi_dilatation[i]**2)))
#     return

@njit(cache=True)
def calculate_dilatation(sigma):
    shear = sigma.copy()
    for i in range(shear.shape[1]):
        shear[2, i] += 1
    dilatation = _batch_norm(shear)
    voronoi_dilatation = (dilatation[:-1] + dilatation[1:])/2
    return dilatation, voronoi_dilatation

@njit(cache=True)
def compare_difference(position, previous_position):
    difference = np.zeros(position.shape[1])
    for i in range(difference.shape[0]):
        difference[i] = np.sqrt(
            (position[0, i]-previous_position[0, i])**2 +
            (position[1, i]-previous_position[1, i])**2 +
            (position[2, i]-previous_position[2, i])**2
        )
    return np.sum(difference)

@njit(cache=True)
def calculate_length(position):
    length = 0
    for i in range(position.shape[1]-1):
        length += np.sqrt(
            (position[0, i+1] - position[0, i])**2 +
            (position[1, i+1] - position[1, i])**2 +
            (position[2, i+1] - position[2, i])**2
        )
    return length

# @njit(cache=True)
# def calculate_potential_energy(strain, curvature, shear_matrix, bend_matrix):
#     dilatation, voronoi_dilatation = calculate_dilatation(strain)
#     energy = 0
#     intrinsic_strain = np.array([0, 0, 1])
#     for i in range(strain.shape[1]):
#         for index_i in range(3):
#             energy += (
#                 shear_matrix[index_i, index_i, i]/dilatation[i]
#                 * (strain[index_i, i]-intrinsic_strain[index_i])**2
#             )
#     for i in range(curvature.shape[1]):
#         for index_i in range(3):
#             energy += (
#                 bend_matrix[index_i, index_i, i]/(voronoi_dilatation[i]**2)
#                 * curvature[index_i, i]**2
#             )
#     return energy/2

# @njit(cache=True)
# def calculate_desired_potential_energy(
#     strain, curvature, shear_matrix, bend_matrix,
#     intrinsic_strain, intrinsic_curvature
# ):
#     dilatation, voronoi_dilatation = calculate_dilatation(strain)
#     energy = 0
#     for i in range(strain.shape[1]):
#         for index_i in range(3):
#             energy += (
#                 shear_matrix[index_i, index_i, i]/dilatation[i]
#                 * (strain[index_i, i]-intrinsic_strain[index_i, i])**2
#             )
#     for i in range(curvature.shape[1]):
#         for index_i in range(3):
#             energy += (
#                 bend_matrix[index_i, index_i, i]/(voronoi_dilatation[i]**2)
#                 * (curvature[index_i, i]-intrinsic_curvature[index_i, i])**2
#             )
#     return energy/2

@njit(cache=True)
def calculate_kinetic_energy(velocity, angular_velocity, mass, inertia):
    energy = 0
    for i in range(velocity.shape[1]):
        for index_i in range(3):
            energy += mass[i]*(velocity[index_i, i]**2)
        for index_i in range(3):
            energy += inertia[index_i, index_i, i] * (angular_velocity[index_i, i]**2)
    return energy/2

@njit(cache=True)
def _vector_function_innder_product(vec_func_1, vec_func_2):
    result = 0
    for i in range(3):
        result += np.trapz(vec_func_1[i]*vec_func_2[i])
    return result

@njit(cache=True)
def sigma_to_shear(sigma):
    shear = np.zeros(sigma.shape)
    for i in range(shear.shape[1]):
        shear[0, i] = sigma[0, i]
        shear[1, i] = sigma[1, i]
        shear[2, i] = sigma[2, i] + 1
    return shear

@njit(cache=True)
def kappa_to_curvature(kappa, voronoi_dilatation):
    curvature = np.zeros(kappa.shape)
    for i in range(curvature.shape[1]):
        curvature[0, i] = kappa[0, i] / voronoi_dilatation[i]
        curvature[1, i] = kappa[1, i] / voronoi_dilatation[i]
        curvature[2, i] = kappa[2, i] / voronoi_dilatation[i]
    return curvature

class CalculateStrainRate(NoForces):
    def __init__(self, sigma_rate, kappa_rate):
        self.sigma_rate = sigma_rate
        self.kappa_rate = kappa_rate
        self.prev_sigma = np.zeros(self.sigma_rate.shape)
        self.prev_kappa = np.zeros(self.kappa_rate.shape)
        self.prev_time = -np.inf

    def apply_torques(self, system, time: np.float = 0.0):
        # dt = time - self.prev_time
        # self.sigma_rate[:, :] = (system.sigma - self.prev_sigma) / dt
        # self.kappa_rate[:, :] = (system.kappa - self.prev_kappa) / dt

        # self.prev_sigma = system.sigma.copy()
        # self.prev_kappa = system.kappa.copy()
        # self.prev_time = time

        self.calculate_sigma_rate(
            system.rest_lengths, system.dilatation, system.director_collection,
            system.velocity_collection, system.omega_collection,
            system.sigma, self.sigma_rate
        )
        self.calculate_kappa_rate(
            system.rest_voronoi_lengths, system.voronoi_dilatation,
            system.omega_collection, system.kappa, self.kappa_rate
        )
        pass

    @staticmethod
    @njit(cache=True)
    def calculate_sigma_rate(
        rest_lengths, dilatation, director,
        velocity, angular_velocity,
        sigma, sigma_rate
    ):
        sigma_rate[:, :] = _batch_matvec(
            director,
            difference_kernel(velocity)[:, 1:-1]/(rest_lengths*dilatation)
        ) - _batch_cross(angular_velocity, sigma_to_shear(sigma))

    @staticmethod
    @njit(cache=True)
    def calculate_kappa_rate(
        rest_voronoi_lengths, voronoi_dilatation,
        angular_velocity, kappa, kappa_rate
    ):
        kappa_rate[:, :] = (
            difference_kernel(angular_velocity)[:, 1:-1] / (rest_voronoi_lengths*voronoi_dilatation)
             - 2 * _batch_cross(
                quadrature_kernel(angular_velocity)[:, 1:-1],
                kappa
             )
        )
