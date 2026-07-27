from dataclasses import dataclass
import numpy as np
from numba import njit

from elastica import FreeBC


@dataclass
class SuckerController:
    index: int
    flag: bool = True
    reduction_ratio: float = 1.0

    def __bool__(self):
        return self.flag

    def turn_on(self):
        self.flag = True

    def turn_off(self):
        self.flag = False


class ControllableFixConstraint(FreeBC):
    """ Modelled after sucker on octopus arm"""

    def __init__(self, index, reduction_ratio=1.0, controller=None, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller or SuckerController(
            index=index, reduction_ratio=reduction_ratio
        )

    @property
    def get_controller(self):
        return self.controller

    def constrain_values(self, system, time):
        return
        # if self.controller.flag:
        #    self.nb_compute_constrain_values(
        #        rod.position_collection,
        #        rod.director_collection,
        #        self.controller.index
        #    )

    def constrain_rates(self, system, time):
        if self.controller.flag:
            self.nb_compute_constrain_rates(
                system.velocity_collection,
                system.omega_collection,
                self.controller.index,
                self.controller.reduction_ratio,
            )

    # @staticmethod
    # @njit(cache=True)
    # def nb_compute_constrain_values(
    #    position_collection, director_collection, index
    # ):
    #    position_collection[2, index] = 0
    #    director_collection[1, :, index] = np.array([0.0,0.0,1.0])

    @staticmethod
    @njit(cache=True)
    def nb_compute_constrain_rates(
        velocity_collection, omega_collection, index, reduction_ratio
    ):
        velocity_collection[..., index] *= 1.0 - reduction_ratio
        omega_collection[..., index] *= 1.0 - reduction_ratio
