from dataclasses import dataclass
import numpy as np
from numba import njit
from elastica._rotations import _rotate

from elastica.boundary_conditions import FreeRod

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
