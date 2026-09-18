from typing import Any

import numpy as np
from numpy.typing import NDArray
from numba import njit

import elastica as ea
from elastica.typing import RodType, RigidBodyType


class FixSphere(ea.ConstraintBase):
    def __init__(
        self,
        fixed_position: tuple[int, ...] = (0,),
        fixed_directors: tuple[int, ...] = (0,),
        **kwargs: Any,
    ) -> None:
        """

        Initialization of the constraint. Any parameter passed to 'using' will be available in kwargs.

        Parameters
        ----------
        constrained_position_idx : tuple
            Tuple of position-indices that will be constrained
        constrained_director_idx : tuple
            Tuple of director-indices that will be constrained
        """
        super().__init__(**kwargs)
        self.fixed_position_collection = np.array(fixed_position)
        self.fixed_directors_collection = np.array(fixed_directors)

    def constrain_values(self, system: "RodType | RigidBodyType", time: np.float64) -> None:
        # system.position_collection[..., 0] = self.fixed_position
        # system.director_collection[..., 0] = self.fixed_directors
        _compute_constrain_values(
            system.position_collection,
            self.fixed_position_collection,
            system.director_collection,
            self.fixed_directors_collection,
        )

    def constrain_rates(self, system: "RodType | RigidBodyType", time: np.float64) -> None:
        # system.velocity_collection[..., 0] = 0.0
        # system.omega_collection[..., 0] = 0.0
        _compute_constrain_rates(
            system.velocity_collection,
            system.omega_collection,
        )


@staticmethod
@njit(cache=True)  # type: ignore
def _compute_constrain_values(
    position_collection: NDArray[np.float64],
    fixed_position_collection: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    fixed_directors_collection: NDArray[np.float64],
) -> None:
    """
    Computes constrain values in numba njit decorator.

    Parameters
    ----------
    position_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    fixed_position_collection : numpy.ndarray
        2D (dim, 1) array containing data with 'float' type.
    director_collection : numpy.ndarray
        3D (dim, dim, blocksize) array containing data with 'float' type.
    fixed_directors_collection : numpy.ndarray
        3D (dim, dim, 1) array containing data with 'float' type.
    """
    position_collection[..., 0] = fixed_position_collection
    director_collection[..., 0] = fixed_directors_collection


@staticmethod
@njit(cache=True)  # type: ignore
def _compute_constrain_rates(
    velocity_collection: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
) -> None:
    """
    Compute constrain rates in numba njit decorator

    Parameters
    ----------
    velocity_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    omega_collection : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    """
    velocity_collection[..., 0] = 0.0
    omega_collection[..., 0] = 0.0

