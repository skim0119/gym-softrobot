"""
Created on Jan. 04, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from numba import njit

from elastica._linalg import _batch_matvec, _batch_cross
from elastica._calculus import quadrature_kernel, difference_kernel
from elastica.external_forces import inplace_addition, NoForces

@njit(cache=True)
def _material_to_lab(director_collection, vectors):
    blocksize = vectors.shape[1]
    lab_frame_vectors = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                lab_frame_vectors[i, n] += (
                    director_collection[j, i, n] * vectors[j, n]
                )
    return lab_frame_vectors

@njit(cache=True)
def _lab_to_material(director_collection, vectors):
    return _batch_matvec(director_collection, vectors)

@njit(cache=True)
def _internal_to_external_load(
    director_collection, kappa, tangents,
    rest_lengths, rest_voronoi_lengths,
    dilatation, voronoi_dilatation,
    internal_forces, internal_couples,
    external_forces, external_couples
    ):

    external_forces[:, :] = difference_kernel(
        _material_to_lab(director_collection, internal_forces)
        )

    external_couples[:, :] = (
        difference_kernel(internal_couples) +
        quadrature_kernel(
            _batch_cross(kappa, internal_couples) * rest_voronoi_lengths
        ) +
        _batch_cross(
                _lab_to_material(director_collection, tangents * dilatation),
                internal_forces
            )  * rest_lengths
    )
    return

@njit(cache=True)
def _force_induced_couple(internal_forces, distance, internal_couples):
    internal_couples[:, :] = quadrature_kernel(
        _batch_cross(distance, internal_forces)
    )[:, 1:-1]
    return

class ContinuousActuation(object):
    """
    This class and the classes ingerited from this should contain the parameters
    and functions that describe and calculate the forces/couples generated from
    the actuation.
    """

    def __init__(self, n_elements: int):
        self.internal_forces = np.zeros((3, n_elements))       # material frame
        self.external_forces = np.zeros((3, n_elements+1))     # global frame
        self.internal_couples = np.zeros((3, n_elements-1))    # material frame
        self.external_couples = np.zeros((3, n_elements))      # material frame
    
    def __call__(self, system):
        raise NotImplementedError
    
class ApplyActuation(NoForces):
    def __init__(self, actuation, step_skip: int, callback_params: dict):
        self.current_step = 0
        self.actuation = actuation
        self.every = step_skip
        self.callback_params = callback_params

    def apply_torques(self, system, time: np.float = 0.0):
        
        self.actuation(system)
        inplace_addition(system.external_forces, self.actuation.external_forces)
        inplace_addition(system.external_torques, self.actuation.external_couples)

        self.make_callback()

    def make_callback(self,):
        if self.current_step % self.every == 0:
            self.callback_func()
        self.current_step += 1

    def callback_func(self):
        self.callback_params['internal_forces'].append(self.actuation.internal_forces.copy())
        self.callback_params['internal_couples'].append(self.actuation.internal_couples.copy())
        self.callback_params['external_forces'].append(self.actuation.external_forces.copy())
        self.callback_params['external_couples'].append(self.actuation.external_couples.copy())
