"""Numba-accelerated tendon routing forces for the continuum arm."""

from __future__ import annotations

import numpy as np
from elastica._linalg import _batch_matvec
from elastica.external_forces import NoForces
from numba import njit


@njit(cache=True)
def _tendon_directions(
    positions: np.ndarray,
    directors: np.ndarray,
    contact_nodes: np.ndarray,
    routing_offsets: np.ndarray,
) -> np.ndarray:
    n_tendons = routing_offsets.shape[0]
    directions = np.zeros((n_tendons, len(contact_nodes) + 1, 3))
    route_nodes = np.empty(len(contact_nodes) + 1, dtype=np.int64)
    route_nodes[0] = 0
    route_nodes[1:] = contact_nodes
    route_elements = np.minimum(route_nodes, positions.shape[1] - 2)
    route_directors = directors[:, :, route_elements]
    inverse_directors = np.transpose(route_directors, (1, 0, 2))
    epsilon = np.finfo(np.float64).eps
    for tendon_index in range(n_tendons):
        route_offsets = routing_offsets[tendon_index].T
        points = positions[:, route_nodes] + _batch_matvec(
            inverse_directors, route_offsets
        )
        for route_index in range(len(contact_nodes)):
            delta = points[:, route_index + 1] - points[:, route_index]
            norm = np.linalg.norm(delta)
            directions[tendon_index, route_index] = delta / max(norm, epsilon)
    return directions


@njit(cache=True)
def _apply_tendon_forces(
    tensions: np.ndarray,
    directions: np.ndarray,
    contact_nodes: np.ndarray,
    external_forces: np.ndarray,
) -> np.ndarray:
    n_tendons = tensions.shape[0]
    tendon_forces = np.zeros((n_tendons, len(contact_nodes), 3))
    for tendon_index in range(n_tendons):
        for route_index in range(len(contact_nodes)):
            tendon_forces[tendon_index, route_index] = tensions[tendon_index] * (
                directions[tendon_index, route_index + 1]
                - directions[tendon_index, route_index]
            )
    for route_index, node_index in enumerate(contact_nodes):
        for tendon_index in range(n_tendons):
            external_forces[:, node_index] += tendon_forces[tendon_index, route_index]
    return tendon_forces


@njit(cache=True)
def _apply_tendon_torques(
    routing_offsets: np.ndarray,
    contact_nodes: np.ndarray,
    tendon_forces: np.ndarray,
    directors: np.ndarray,
    external_torques: np.ndarray,
) -> None:
    for tendon_index in range(routing_offsets.shape[0]):
        for route_index, node_index in enumerate(contact_nodes):
            element_index = node_index - 1
            local_force = _batch_matvec(
                directors[:, :, element_index : element_index + 1],
                tendon_forces[tendon_index, route_index : route_index + 1].T,
            )
            external_torques[:, element_index] += np.cross(
                routing_offsets[tendon_index, route_index + 1], local_force[:, 0]
            )


class TendonActuation(NoForces):
    """Apply routed tendon tensions at a fraction of the local rod radius."""

    def __init__(
        self,
        *,
        contact_nodes: np.ndarray,
        tensions: np.ndarray,
        rod_radii: np.ndarray,
        radial_fraction: float = 0.75,
    ) -> None:
        super().__init__()
        self.contact_nodes = np.asarray(contact_nodes, dtype=np.int64)
        self.tensions = tensions
        element_radii = np.asarray(rod_radii)
        node_radii = np.empty(element_radii.size + 1)
        node_radii[0] = element_radii[0]
        node_radii[-1] = element_radii[-1]
        node_radii[1:-1] = 0.5 * (element_radii[:-1] + element_radii[1:])
        route_radii = (
            radial_fraction
            * node_radii[
                np.concatenate((np.array((0,), dtype=np.int64), self.contact_nodes))
            ]
        )
        tendon_angles = 2.0 * np.pi * np.arange(self.tensions.size) / self.tensions.size
        tendon_directions = np.column_stack(
            (np.cos(tendon_angles), np.sin(tendon_angles), np.zeros(self.tensions.size))
        )
        self.routing_offsets = (
            tendon_directions[:, None, :] * route_radii[None, :, None]
        )
        self._tendon_forces = np.zeros(
            (self.tensions.size, len(self.contact_nodes), 3)
        )

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        directions = _tendon_directions(
            system.position_collection,
            system.director_collection,
            self.contact_nodes,
            self.routing_offsets,
        )
        self._tendon_forces = _apply_tendon_forces(
            self.tensions,
            directions,
            self.contact_nodes,
            system.external_forces,
        )

    def apply_torques(self, system, time: np.float64 = np.float64(0.0)) -> None:
        _apply_tendon_torques(
            self.routing_offsets,
            self.contact_nodes,
            self._tendon_forces,
            system.director_collection,
            system.external_torques,
        )
