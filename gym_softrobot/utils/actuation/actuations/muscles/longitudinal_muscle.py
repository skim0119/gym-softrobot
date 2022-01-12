"""
Created on Jan. 07, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit

from elastica._linalg import _batch_cross
from elastica._calculus import quadrature_kernel

from gym_softrobot.utils.actuation.actuations.actuation import _internal_to_external_load, _force_induced_couple
from gym_softrobot.utils.actuation.actuations.muscles.muscle import (
    local_strain,
    MuscleForce,
    MuscleFibers,
)
# from actuations.muscles.activations_function import (
#     set_guassian_activations,
#     GuassianFunction,
#     UniformFunction
# )

# from actuations.muscles.weighted_function import weighted_load
from gym_softrobot.utils.actuation.rod_tools import sigma_to_shear, kappa_to_curvature

class LongitudinalMuscle(MuscleForce):
    def __init__(
        self,
        muscle_radius_ratio,
        max_force,
        strain_weighted_function=None,
        strain_rate_weighted=None
    ):
        MuscleForce.__init__(self, muscle_radius_ratio.shape[1])
        self.muscle_radius_ratio = np.zeros((3, self.n_elements))
        self.muscle_radius_ratio[:2, :] = muscle_radius_ratio.copy()
        self.max_force = max_force.copy()
        self.strain_weighted_function = strain_weighted_function
        self.strain_rate_weighted = False
        if strain_rate_weighted is not None:
            self.strain_rate_weighted = True
            self.strain_rate_weighted_function = strain_rate_weighted['function']
            self.sigma_rate = strain_rate_weighted['sigma_rate']
            self.kappa_rate = strain_rate_weighted['kappa_rate']
        self.count = 0


    def __call__(self, system, count_flag=True):

        magnitude_for_force = self.get_activation() * self.max_force

        weight = self.calculate_force_length_velocity_weight(system)

        # weighted_load(
        #     load=magnitude_for_force,
        #     weight=weight
        # )

        longitudinal_muscle_function(
            magnitude_for_force, self.muscle_radius_ratio*system.radius,
            system.director_collection, system.kappa, system.tangents,
            system.rest_lengths, system.rest_voronoi_lengths,
            system.dilatation, system.voronoi_dilatation,
            self.internal_forces, self.internal_couples,
            self.external_forces, self.external_couples
        )
        if count_flag:
            self.count += 1

    def calculate_force_length_velocity_weight(self, system):
        weight = np.ones(self.activation.shape)

        if self.strain_weighted_function is not None:
            weight[:] = self.strain_weighted_function(
                stretch_mag=self.longitudinal_muscle_stretch_mag(
                    off_center_displacement=self.muscle_radius_ratio*system.radius,
                    strain=sigma_to_shear(system.sigma),
                    curvature=kappa_to_curvature(system.kappa, system.voronoi_dilatation)
                )
            )
        
        # if self.strain_rate_weighted:
        #     stretch_rate = self.longitudinal_muscle_stretch_rate_mag(
        #         off_center_displacement=self.muscle_radius_ratio*system.radius,
        #         sigma_rate=self.sigma_rate,
        #         curvature_rate=kappa_to_curvature(self.kappa_rate, system.voronoi_dilatation)
        #     )
        #     # if (self.count % 1000) == 0:
        #     #     print(np.max(weight))
        #     weight[:] *= self.strain_rate_weighted_function(stretch_rate)

        return weight

    @staticmethod
    @njit(cache=True)
    def longitudinal_muscle_stretch_mag(off_center_displacement, strain, curvature):
        nu = local_strain(
            off_center_displacement, strain, curvature
        )
        blocksize = nu.shape[1]
        stretch_mag = np.ones(blocksize)
        for i in range(blocksize):
            stretch_mag[i] = np.linalg.norm(nu[:, i])
        return stretch_mag

    @staticmethod
    @njit(cache=True)
    def longitudinal_muscle_stretch_rate_mag(off_center_displacement, sigma_rate, curvature_rate):
        nu_rate = local_strain(
            off_center_displacement, sigma_rate, curvature_rate
        )
        blocksize = nu_rate.shape[1]
        stretch_mag = np.ones(blocksize)
        for i in range(blocksize):
            stretch_mag[i] = nu_rate[2, i]
        return stretch_mag

    # @staticmethod
    # @njit(cache=True)
    # def longitudinal_muscle_function(
    #     magnitude_for_force, r_m,
    #     director_collection, kappa, tangents,
    #     rest_lengths, rest_voronoi_lengths,
    #     dilatation, voronoi_dilatation,
    #     internal_forces, internal_couples,
    #     external_forces, external_couples
    # ):

    #     internal_forces[2, :] = magnitude_for_force.copy()
    #     internal_couples[:, :] = quadrature_kernel(
    #         _batch_cross(r_m, internal_forces)
    #     )[:, 1:-1]

    #     _internal_to_external_load(
    #         director_collection, kappa, tangents,
    #         rest_lengths, rest_voronoi_lengths,
    #         dilatation, voronoi_dilatation,
    #         internal_forces, internal_couples,
    #         external_forces, external_couples
    #     )

@njit(cache=True)
def longitudinal_muscle_function(
    magnitude_for_force, r_m,
    director_collection, kappa, tangents,
    rest_lengths, rest_voronoi_lengths,
    dilatation, voronoi_dilatation,
    internal_forces, internal_couples,
    external_forces, external_couples
):

    internal_forces[2, :] = magnitude_for_force.copy()
    _force_induced_couple(internal_forces, r_m, internal_couples)
    # internal_couples[:, :] = quadrature_kernel(
    #     _batch_cross(r_m, internal_forces)
    # )[:, 1:-1]

    _internal_to_external_load(
        director_collection, kappa, tangents,
        rest_lengths, rest_voronoi_lengths,
        dilatation, voronoi_dilatation,
        internal_forces, internal_couples,
        external_forces, external_couples
    )


# class GuassianLongitudinalMuscle(LongitudinalMuscle, GuassianFunction):
#     def __init__(self, muscle_radius_ratio, max_force, variance, **kwargs):
#         LongitudinalMuscle.__init__(self, muscle_radius_ratio, max_force, **kwargs)
#         GuassianFunction.__init__(self, variance)

#     def set_controls(self, controls, positions):
#         self.set_activations(
#             controls, positions, self.s, self.variance, self.activations
#         )

# class UniformLongitudinalMuscle(LongitudinalMuscle, UniformFunction):
#     def __init__(self, muscle_radius_ratio, max_force, width):
#         LongitudinalMuscle.__init__(self, muscle_radius_ratio, max_force)
#         UniformFunction.__init__(self, width)

#     def set_controls(self, controls, positions):
#         self.set_activations(
#             controls, positions, self.s, self.width, self.activations
#         )

# class GuassianLongitudinalMuscleFibers(GuassianLongitudinalMuscle, MuscleFibers):
#     def __init__(self, muscle_radius_ratio, max_force, variance, positions, **kwargs):
#         GuassianLongitudinalMuscle.__init__(self, muscle_radius_ratio, max_force, variance, **kwargs)
#         self.positions = np.array(positions)
#         MuscleFibers.__init__(self, self.n_elements, self.positions.shape[0])

#     def __call__(self, system):
#         if self.G_flag:
#             self.internal_forces[:, :] *= 0
#             self.internal_couples[:, :] *= 0
#             self.external_forces[:, :] *= 0
#             self.external_couples[:, :] *= 0

#             self.muscle_fibers_function(
#                 self.controls,
#                 self.G_internal_forces, self.G_internal_couples,
#                 self.G_external_forces, self.G_external_couples,
#                 self.internal_forces, self.internal_couples,
#                 self.external_forces, self.external_couples
#             )

#             # self.G_activations[:, :] *= 0
#             self.G_internal_forces[:, :, :] *= 0
#             self.G_internal_couples[:, :, :] *= 0
#             self.G_external_forces[:, :, :] *= 0
#             self.G_external_couples[:, :, :] *= 0
#             self.G_flag = False

#         else:
#             GuassianLongitudinalMuscle.__call__(self, system)

#     def set_controls(self, controls):
#         self.controls[:] = np.array(controls)
#         GuassianLongitudinalMuscle.set_controls(self, self.controls, self.positions)
    
#     def calculate_G(self, system):
#         # for i in range(self.positions.shape[0]):
#         #     self.set_activations(
#         #         np.ones(1), self.positions[i:i+1], self.s, self.variance, self.activations
#         #     )
#         #     self(system)
#         #     # self.G_activations[i, :] = self.activations.copy()
#         #     self.G_internal_forces[i, :, :] = self.internal_forces.copy()
#         #     self.G_internal_couples[i, :, :] = self.internal_couples.copy()
#         #     self.G_external_forces[i, :, :] = self.external_forces.copy()
#         #     self.G_external_couples[i, :, :] = self.external_couples.copy()

#         weight = self.calculate_force_length_velocity_weight(system)

#         self.calculate_G_numba(
#             self.positions, self.s, self.variance, self.activations,
#             self.max_force, self.muscle_radius_ratio*system.radius, weight,
#             system.director_collection, system.kappa, system.tangents,
#             system.rest_lengths, system.rest_voronoi_lengths,
#             system.dilatation, system.voronoi_dilatation,
#             self.internal_forces, self.internal_couples,
#             self.external_forces, self.external_couples,
#             self.G_internal_forces, self.G_internal_couples,
#             self.G_external_forces, self.G_external_couples
#         )
        
#         self.G_flag = True
    
#     @staticmethod
#     @njit(cache=True)
#     def calculate_G_numba(
#         positions, s, variance, activations,
#         max_force, off_center_displacement, weight,
#         director_collection, kappa, tangents,
#         rest_lengths, rest_voronoi_lengths,
#         dilatation, voronoi_dilatation,
#         internal_forces, internal_couples,
#         external_forces, external_couples,
#         G_internal_forces, G_internal_couples,
#         G_external_forces, G_external_couples,
#     ):
#         blocksize = positions.shape[0]
#         for i in range(blocksize):
#             set_guassian_activations(
#                 np.ones(1), positions[i: i+1], s, variance, activations
#             )
#             magnitude_for_force = activations * max_force * weight
#             longitudinal_muscle_function(
#                 magnitude_for_force, off_center_displacement,
#                 director_collection, kappa, tangents,
#                 rest_lengths, rest_voronoi_lengths,
#                 dilatation, voronoi_dilatation,
#                 internal_forces, internal_couples,
#                 external_forces, external_couples
#             )
#             G_internal_forces[i, :, :] = internal_forces.copy()
#             G_internal_couples[i, :, :] = internal_couples.copy()
#             G_external_forces[i, :, :] = external_forces.copy()
#             G_external_couples[i, :, :] = external_couples.copy()
#         pass

# class UniformLongitudinalMuscleFibers(UniformLongitudinalMuscle, MuscleFibers):
#     def __init__(self, muscle_radius_ratio, max_force, width, positions):
#         UniformLongitudinalMuscle.__init__(self, muscle_radius_ratio, max_force, width)
#         self.positions = np.array(positions)
#         MuscleFibers.__init__(self, self.n_elements, self.positions.shape[0])

#     def set_controls(self, controls):
#         self.controls[:] = np.array(controls)
#         UniformLongitudinalMuscle.set_controls(self, self.controls, self.positions)
