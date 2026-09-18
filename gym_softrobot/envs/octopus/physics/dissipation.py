from typing import Any

import elastica as ea
import numpy as np
from elastica.typing import RodType
from numba import njit


# TODO: Rename, exponential damping
class RayleighDamping(ea.DamperBase):
    """
    Rayleigh damping model (old model)

    .. math::

        \\mathbf{F}_{damp} = -\\nu \\mathbf{v}

        \\boldsymbol{\\tau}_{damp} = -\\nu_t \\boldsymbol{\\omega}

    Parameters
    ----------
    damping_constant : float
        Damping coefficient :math:`\\nu` (per unit length). Units: [1/s] or [kg/(m·s)]
    rotational_damping_constant  : float
        Rotational damping coefficient :math:`\\nu_t` (per unit length). Units: [1/s] or [kg/(m·s)]

    Examples
    --------
    .. code-block:: python

        simulator.dampen(rod).using(
            RayleighDissipation,
            damping_constant=0.1,
            rotational_damping_constant =0.1,
        )

    See Also
    --------
    AnalyticalLinearDamper : Recommended alternative with better stability
    LaplaceDissipationFilter : Alternative filtering-based dissipation
    """

    def __init__(
        self,
        damping_constant: np.float64,
        rotational_damping_constant: np.float64,
        time_step: np.float64,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nu = np.exp(-damping_constant * time_step)
        self.nu_t = np.exp(-rotational_damping_constant * time_step)

    def dampen_rates(self, system: RodType, time: np.float64) -> None:
        """
        Apply Rayleigh dissipation forces and torques.

        Parameters
        ----------
        system : RodType
            Rod system to apply damping to
        time : float
            Current simulation time
        """

        _rayleigh_dissipate(
            self.nu,
            self.nu_t,
            system.velocity_collection,  # shape: (3, N)
            system.omega_collection,  # shape: (3, N-1)
        )


@njit(cache=True)  # type: ignore
def _rayleigh_dissipate(
    nu: float,
    nu_tau: float,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> None:
    """
    Numba-optimized implementation of Rayleigh dissipation.
    """
    velocity *= nu
    angular_velocity *= nu_tau

