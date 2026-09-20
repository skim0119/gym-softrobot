"""Rod–rod contact with optional skip of base elements (shared hub penetration)."""

from __future__ import annotations

import elastica as ea
import numpy as np
from elastica._linalg import _batch_product_k_ik_to_ik
from elastica.contact_utils import _find_min_dist, _prune_using_aabbs_rod_rod
from elastica.typing import RodType
from numba import njit
from numpy.typing import NDArray


class RodRodContactSkipBaseElements(ea.NoContact):
    """
    ``RodRodContact`` that ignores the first ``skip_elements`` on each rod.
    Often, the first few elements coincide and create a large penetration force.
    """

    def __init__(
        self,
        k: float,
        nu: float,
        *,
        skip_elements: int = 0,
    ) -> None:
        super().__init__()
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.skip_elements = int(skip_elements)

    def apply_contact(
        self,
        system_one: RodType,
        system_two: RodType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        del time
        if _prune_using_aabbs_rod_rod(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection,
            system_two.radius,
            system_two.lengths,
        ):
            return

        _calculate_contact_forces_rod_rod_skip_base(
            system_one.position_collection[..., :-1],
            system_one.radius,
            system_one.lengths,
            system_one.tangents,
            system_one.velocity_collection,
            system_one.internal_forces,
            system_one.external_forces,
            system_two.position_collection[..., :-1],
            system_two.radius,
            system_two.lengths,
            system_two.tangents,
            system_two.velocity_collection,
            system_two.internal_forces,
            system_two.external_forces,
            self.k,
            self.nu,
            self.skip_elements,
            self.skip_elements,
        )


@njit(cache=True)  # type: ignore
def _calculate_contact_forces_rod_rod_skip_base(
    x_collection_rod_one: NDArray[np.float64],
    radius_rod_one: NDArray[np.float64],
    length_rod_one: NDArray[np.float64],
    tangent_rod_one: NDArray[np.float64],
    velocity_rod_one: NDArray[np.float64],
    internal_forces_rod_one: NDArray[np.float64],
    external_forces_rod_one: NDArray[np.float64],
    x_collection_rod_two: NDArray[np.float64],
    radius_rod_two: NDArray[np.float64],
    length_rod_two: NDArray[np.float64],
    tangent_rod_two: NDArray[np.float64],
    velocity_rod_two: NDArray[np.float64],
    internal_forces_rod_two: NDArray[np.float64],
    external_forces_rod_two: NDArray[np.float64],
    contact_k: np.float64,
    contact_nu: np.float64,
    skip_i: int,
    skip_j: int,
) -> None:
    """Elastica rod–rod contact, skipping the first ``skip_elements`` on each rod."""
    n_points_rod_one = x_collection_rod_one.shape[1]
    n_points_rod_two = x_collection_rod_two.shape[1]

    edge_collection_rod_one = _batch_product_k_ik_to_ik(length_rod_one, tangent_rod_one)
    edge_collection_rod_two = _batch_product_k_ik_to_ik(length_rod_two, tangent_rod_two)

    for i in range(skip_i, n_points_rod_one):
        for j in range(skip_j, n_points_rod_two):
            radii_sum = radius_rod_one[i] + radius_rod_two[j]
            length_sum = length_rod_one[i] + length_rod_two[j]
            x_selected_rod_one = x_collection_rod_one[..., i]
            x_selected_rod_two = x_collection_rod_two[..., j]

            del_x = x_selected_rod_one - x_selected_rod_two
            norm_del_x = np.sqrt(np.dot(del_x, del_x))
            if norm_del_x >= (radii_sum + length_sum):
                continue

            distance_vector, _, _ = _find_min_dist(
                x_selected_rod_one,
                edge_collection_rod_one[..., i],
                x_selected_rod_two,
                edge_collection_rod_two[..., j],
            )
            distance_vector_length = np.sqrt(np.dot(distance_vector, distance_vector))
            distance_vector /= distance_vector_length
            gamma = radii_sum - distance_vector_length
            if gamma < -1e-5:
                continue

            rod_one_elemental_forces = 0.5 * (
                external_forces_rod_one[..., i]
                + external_forces_rod_one[..., i + 1]
                + internal_forces_rod_one[..., i]
                + internal_forces_rod_one[..., i + 1]
            )
            rod_two_elemental_forces = 0.5 * (
                external_forces_rod_two[..., j]
                + external_forces_rod_two[..., j + 1]
                + internal_forces_rod_two[..., j]
                + internal_forces_rod_two[..., j + 1]
            )
            equilibrium_forces = -rod_one_elemental_forces + rod_two_elemental_forces
            normal_force = abs(min(np.dot(equilibrium_forces, distance_vector), 0.0))

            mask = (gamma > 0.0) * 1.0
            contact_force = contact_k * gamma
            interpenetration_velocity = 0.5 * (
                (velocity_rod_one[..., i] + velocity_rod_one[..., i + 1])
                - (velocity_rod_two[..., j] + velocity_rod_two[..., j + 1])
            )
            contact_damping_force = contact_nu * np.dot(
                interpenetration_velocity, distance_vector
            )
            net_contact_force = (
                normal_force + 0.5 * mask * (contact_damping_force + contact_force)
            ) * distance_vector

            if i == 0:
                external_forces_rod_one[..., i] -= net_contact_force * 2 / 3
                external_forces_rod_one[..., i + 1] -= net_contact_force * 4 / 3
            elif i == n_points_rod_one - 1:
                external_forces_rod_one[..., i] -= net_contact_force * 4 / 3
                external_forces_rod_one[..., i + 1] -= net_contact_force * 2 / 3
            else:
                external_forces_rod_one[..., i] -= net_contact_force
                external_forces_rod_one[..., i + 1] -= net_contact_force

            if j == 0:
                external_forces_rod_two[..., j] += net_contact_force * 2 / 3
                external_forces_rod_two[..., j + 1] += net_contact_force * 4 / 3
            elif j == n_points_rod_two - 1:
                external_forces_rod_two[..., j] += net_contact_force * 4 / 3
                external_forces_rod_two[..., j + 1] += net_contact_force * 2 / 3
            else:
                external_forces_rod_two[..., j] += net_contact_force
                external_forces_rod_two[..., j + 1] += net_contact_force
