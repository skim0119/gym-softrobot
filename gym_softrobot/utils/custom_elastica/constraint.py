from dataclasses import dataclass
import numpy as np
from numba import njit
from elastica._rotations import _rotate

from elastica.boundary_conditions import FreeRod


class BodyBoundaryCondition(FreeRod):
    """
    This boundary condition class fixes body orientation

        Attributes
        ----------
        fixed_positions : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
    """

    def __init__(self, fixed_position, fixed_director):
        FreeRod.__init__(self)
        self.fixed_position = fixed_position
        self.fixed_director = fixed_director

    def constrain_values(self, rod, time):
        self.compute_contrain_values(
            rod.position_collection,
            self.fixed_position,
            rod.director_collection,
            self.fixed_director,
        )

    def constrain_rates(self, rod, time):
        self.compute_constrain_rates(
            rod.velocity_collection,
            rod.omega_collection,
            rod.acceleration_collection,
            rod.alpha_collection,
        )

    @staticmethod
    @njit(cache=True)
    def compute_contrain_values(
        position_collection, fixed_position, director_collection, fixed_director
    ):
        # Position
        position_collection[2, 0] = fixed_position[2]
        # Director
        director_collection[2, 0, 0] = 0.0
        director_collection[2, 1, 0] = 0.0
        director_collection[2, 2, 0] = 1.0
        for i in range(2):
            length = np.sqrt(
                director_collection[i, 0, 0] ** 2 + director_collection[i, 1, 0] ** 2
            )
            for j in range(2):
                director_collection[i, j, 0] /= length
            director_collection[i, 2, 0] = 0.0

    @staticmethod
    @njit(cache=True)
    def compute_constrain_rates(velocity, omega, acceleration, alpha):
        """
        Compute contrain rates in numba njit decorator
        Parameters
        ----------
        velocity : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        omega : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        acceleration : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.
        alpha : numpy.ndarray
            2D (dim, blocksize) array containing data with `float` type.

        Returns
        -------

        """
        # Translational
        velocity[2, :] = 0.0
        acceleration[2, :] = 0.0
        # Rotational
        omega[:2, 0] = 0.0
        alpha[:2, 0] = 0.0


class ControllableFixConstraint(FreeRod):
    @dataclass
    class _Controller:
        index: int
        flag: bool = True

        def __bool__(self):
            return self.flag

        def turn_on(self):
            self.flag = True

        def turn_off(self):
            self.flag = False

    """ Modelled after sucker on octopus arm"""

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.controller = self._Controller(index=index)

    @property
    def get_controller(self):
        return self.controller

    def constrain_values(self, rod, time):
        pass

    def constrain_rates(self, rod, time):
        if self.controller.flag:
            self.nb_compute_constrain_rates(
                rod.velocity_collection,
                rod.omega_collection,
                self.controller.index
            )

    @staticmethod
    @njit(cache=True)
    def nb_compute_contrain_values(
        position_collection, fixed_position, director_collection, fixed_directors, index
    ):
        position_collection[..., index] = fixed_position
        director_collection[..., index] = fixed_directors

    @staticmethod
    @njit(cache=True)
    def nb_compute_constrain_rates(velocity_collection, omega_collection, index):
        velocity_collection[..., index] = 0.0
        omega_collection[..., index] = 0.0
