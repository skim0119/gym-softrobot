"""Spirob-derived tapered geometry for the tendon-driven arm."""

from __future__ import annotations

import numpy as np
from elastica import CosseratRod


# Fit on a 25-element mesh for a 0.30 m arm, 3.5 degree taper, and a 0.023 m
# base radius. The theta interval retains the 15 outer spiral sections used by
# the Octopus arm.
SPIROB_A = 0.010839912
SPIROB_B = 0.100482603
SPIROB_BASE_LENGTH = 0.30
SPIROB_BASE_RADIUS = 0.023
SPIROB_THETA_MAX = 4.0 * np.pi
SPIROB_ACTIVE_SECTION_FRACTION = 15.0 / 24.0


def spirob_taper_profile(
    n_elements: int,
    *,
    base_length: float = SPIROB_BASE_LENGTH,
    base_radius: float = SPIROB_BASE_RADIUS,
    a: float = SPIROB_A,
    b: float = SPIROB_B,
) -> tuple[np.ndarray, np.ndarray]:
    """Return straight-element lengths and radii from the fitted spiral strip."""
    if n_elements < 3:
        raise ValueError("n_elements must be at least 3")
    if base_length <= 0.0 or base_radius <= 0.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("length, radius, a, and b must be positive")

    theta_start = SPIROB_THETA_MAX * (1.0 - SPIROB_ACTIVE_SECTION_FRACTION)
    theta = np.linspace(theta_start, SPIROB_THETA_MAX, n_elements + 1)
    inner_radius = a * np.exp(b * theta)
    curled_radius = 0.5 * a * (np.exp(b * theta) + np.exp(b * (theta + 2.0 * np.pi)))
    inner_positions = np.vstack(
        (inner_radius * np.cos(theta), inner_radius * np.sin(theta))
    )
    curled_positions = np.vstack(
        (curled_radius * np.cos(theta), curled_radius * np.sin(theta))
    )

    lengths = np.linalg.norm(np.diff(curled_positions, axis=1), axis=0)[::-1]
    radii = np.linalg.norm(
        np.diff(curled_positions, axis=1) - np.diff(inner_positions, axis=1),
        axis=0,
    )[::-1]
    lengths *= base_length / lengths.sum()
    radii *= base_radius / radii[0]
    return lengths, radii


def create_spirob_rod(
    *,
    n_elements: int,
    start: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    density: float,
    youngs_modulus: float,
    base_length: float = SPIROB_BASE_LENGTH,
    base_radius: float = SPIROB_BASE_RADIUS,
) -> CosseratRod:
    """Create a straight, tapered Cosserat rod from the fitted Spirob profile."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    lengths, radii = spirob_taper_profile(
        n_elements, base_length=base_length, base_radius=base_radius
    )
    rod = CosseratRod.straight_rod(
        n_elements=n_elements,
        start=np.asarray(start, dtype=np.float64),
        direction=direction,
        normal=np.asarray(normal, dtype=np.float64),
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
    )

    rod.position_collection[:, 0] = start
    rod.position_collection[:, 1:] = (
        np.asarray(start)[:, None] + np.cumsum(lengths)[None, :] * direction[:, None]
    )
    rod.lengths[:] = lengths
    rod.rest_lengths[:] = lengths
    rod.tangents[:] = direction[:, None]
    rod.rest_voronoi_lengths[:] = 0.5 * (lengths[:-1] + lengths[1:])
    rod.radius[:] = radii

    area = np.pi * radii**2
    second_moment = area**2 / (4.0 * np.pi)
    polar_moment = 2.0 * second_moment
    rod.volume[:] = area * lengths
    rod.mass.fill(0.0)
    rod.mass[:-1] += 0.5 * density * rod.volume
    rod.mass[1:] += 0.5 * density * rod.volume

    inertia = rod.mass_second_moment_of_inertia
    inertia.fill(0.0)
    inertia[0, 0] = density * lengths * second_moment
    inertia[1, 1] = density * lengths * second_moment
    inertia[2, 2] = density * lengths * polar_moment
    rod.inv_mass_second_moment_of_inertia[:] = np.moveaxis(
        np.linalg.inv(np.moveaxis(inertia, 2, 0)), 0, 2
    )

    rod.shear_matrix.fill(0.0)
    rod.shear_matrix[0, 0] = (27.0 / 28.0) * (youngs_modulus / 3.0) * area
    rod.shear_matrix[1, 1] = (27.0 / 28.0) * (youngs_modulus / 3.0) * area
    rod.shear_matrix[2, 2] = youngs_modulus * area

    element_bend = np.zeros((3, 3, n_elements))
    element_bend[0, 0] = youngs_modulus * second_moment
    element_bend[1, 1] = youngs_modulus * second_moment
    element_bend[2, 2] = (youngs_modulus / 3.0) * polar_moment
    rod.bend_matrix[:] = (
        element_bend[..., 1:] * lengths[1:] + element_bend[..., :-1] * lengths[:-1]
    ) / rod.rest_voronoi_lengths

    return rod
