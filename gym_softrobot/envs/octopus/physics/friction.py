from __future__ import annotations
from typing import TypeAlias
import numpy as np
import numpy.typing as npt

from numpy.typing import NDArray

import elastica as ea
from gym_softrobot.utils.custom_elastica.compat import (
    _calculate_contact_forces_rod_plane_with_anisotropic_friction,
)

Vector3: TypeAlias = tuple[float, float, float] | npt.NDArray[np.float64]


class AnisotropicFriction(ea.NoForces):
    """
    This class is for applying contact forces between rod-plane with friction.
    For more details regarding the contact module refer to
    Eqn 4.8 of Gazzola et al. RSoS (2018).

    Examples
    --------
    How to define contact between rod and plane.

    >>> simulator.detect_contact_between(rod).using(
    ...    k=1e4,
    ...    nu=10,
    ...    slip_velocity_tol = 1e-4,
    ...    static_mu_array = np.array([0.0,0.0,0.0]),
    ...    kinetic_mu_array = np.array([1.0,2.0,3.0]),
    ...    plane_origin = (0.0, 0.0, 0.0),
    ...    plane_normal = (0.0, 1.0, 0.0),
    ...    surface_tol = 1.0e-4,
    ... )
    """

    def __init__(
        self,
        k: float,
        nu: float,
        slip_velocity_tol: float,
        static_mu_array: NDArray[np.float64],
        kinetic_mu_array: NDArray[np.float64],
        plane_origin: Vector3 = (0.0, 0.0, 0.0),
        plane_normal: Vector3 = (0.0, 1.0, 0.0),
        surface_tol: float = 1.0e-4,
    ) -> None:
        """

        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        plane_origin: tuple[float, float, float]
            Origin of the plane.
        plane_normal: tuple[float, float, float]
            Normal of the plane.
        surface_tol: float
            Surface tolerance.
        """
        super().__init__()
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.surface_tol = np.float64(surface_tol)
        self.plane_origin_array = np.asarray(plane_origin, dtype=np.float64).reshape(3, 1)
        self.plane_normal_array = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        self.slip_velocity_tol = slip_velocity_tol
        (
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
        ) = static_mu_array
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array

    def apply_forces(
        self,
        system: ea.RodType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between RodType object and Plane object with anisotropic friction.

        Parameters
        ----------
        system_one : RodType
            Rod object.
        """
        _calculate_contact_forces_rod_plane_with_anisotropic_friction(
            self.plane_origin_array,
            self.plane_normal_array,
            self.surface_tol,
            self.slip_velocity_tol,
            self.k,
            self.nu,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
            system.radius,
            system.mass,
            system.tangents,
            system.position_collection,
            system.director_collection,
            system.velocity_collection,
            system.omega_collection,
            system.internal_forces,
            system.external_forces,
            system.internal_torques,
            system.external_torques,
        )
