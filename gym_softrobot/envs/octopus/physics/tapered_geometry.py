__doc__ = """ Factory function to allocate variables for Cosserat Rod, keeping tapered arm shape"""
import logging

import numpy as np
from gym_softrobot.utils.custom_elastica.compat import (
    _assert_dim,
    _assert_shape,
    _batch_cross,
    _batch_dot,
    _directors_validity_checker,
    _position_validity_checker,
)
from elastica.utils import MaxDimension, Tolerance
from numpy.testing import assert_allclose
from numpy.typing import NDArray

def polar2xy(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def allocate(
    n_elements: int,
    rod_origin_position: np.ndarray,
    direction: NDArray[np.float64],
    normal: NDArray[np.float64],
    base_radius: np.float64,
    density: np.float64,
    youngs_modulus: np.float64,
    *,
    taper_angle_in_deg: float = 3,  # Needed to compute radius
    # Geometry parameter. It is from some optimization schinanigans to match the length and good taper angle
    a_val=0.016700421,
    b_val=0.092864105,
    division_angle=float(np.deg2rad(30)),
    theta_max=4 * np.pi,  # number of rotation the spiral shape could make
):
    log = logging.getLogger()

    N = 24
    discrete_theta = np.arange(N + 1) * division_angle
    inner_r = a_val * np.exp(b_val * discrete_theta)
    inner_positions = np.vstack(polar2xy(inner_r, discrete_theta))
    curled_r = (
        0.5
        * a_val
        * (np.exp(b_val * discrete_theta) + np.exp(b_val * (discrete_theta + 2 * np.pi)))
    )
    curled_positions = np.vstack(polar2xy(curled_r, discrete_theta))
    lengths = np.linalg.norm(np.diff(curled_positions), axis=0)[::-1][:n_elements]
    # print(f"lengths: {lengths}")
    radius = np.linalg.norm(np.diff(curled_positions) - np.diff(inner_positions), axis=0)[::-1][
        :n_elements
    ]
    ratio = base_radius / radius[0]
    radius *= ratio
    # print(f"radius: {radius}")

    # sanity checks here
    assert np.sqrt(np.dot(normal, normal)) > Tolerance.atol()
    assert np.sqrt(np.dot(direction, direction)) > Tolerance.atol()

    # define the number of nodes and voronoi elements based on if rod is
    n_nodes = n_elements + 1
    n_voronoi_elements = n_elements - 1

    # check if position is given.
    # Set the position array
    position = np.zeros((MaxDimension.value(), n_nodes))
    start = rod_origin_position
    position[:, 0] = start
    for i in range(1, n_nodes):
        position[:, i] = position[:, i - 1] + direction * lengths[i - 1]
    _position_validity_checker(position, start, n_elements)

    # Compute rest lengths and tangents
    position_diff = np.diff(position, axis=-1)
    rest_lengths = lengths.copy()
    tangents = position_diff / rest_lengths
    normal /= np.linalg.norm(normal)

    # Set the directors matrix
    directors = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements))
    # Construct directors using tangents and normal
    normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
    # Check if rod normal and rod tangent are perpendicular to each other otherwise
    # directors will be wrong!!
    assert_allclose(
        _batch_dot(normal_collection, tangents),
        0,
        atol=Tolerance.atol(),
        err_msg=(" Rod normal and tangent are not perpendicular to each other!"),
    )
    directors[0, ...] = normal_collection
    directors[1, ...] = _batch_cross(tangents, normal_collection)
    directors[2, ...] = tangents
    _directors_validity_checker(directors, tangents, n_elements)

    # Set radius array
    # Check if the elements of radius are greater than tolerance
    assert np.all(radius > Tolerance.atol()), " Radius has to be greater than 0."

    # Set density array
    density_array = np.zeros((n_elements))
    # Check if the user input density is valid
    density_temp = np.array(density)
    _assert_dim(density_temp, 2, "density")
    density_array[:] = density_temp
    # Check if the elements of density are greater than tolerance
    assert np.all(density_array > Tolerance.atol()), " Density has to be greater than 0."

    # Second moment of inertia
    A0 = np.pi * radius * radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3]).transpose()
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements), np.float64
    )

    mass_second_moment_of_inertia_temp = np.einsum("ij,i->ij", I0, density * rest_lengths)

    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )
    # sanity check of mass second moment of inertia
    if (mass_second_moment_of_inertia < Tolerance.atol()).all():
        message = "Mass moment of inertia matrix smaller than tolerance, please check provided radius, density and length."
        log.warning(message)

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_elements)
    )
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i]) == MaxDimension.value()
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    shear_matrix = np.zeros((MaxDimension.value(), MaxDimension.value(), n_elements), np.float64)
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i],
            ],
        )

    # Bend/Twist matrix
    bend_matrix = np.zeros(
        (MaxDimension.value(), MaxDimension.value(), n_voronoi_elements + 1),
        np.float64,
    )
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
                shear_modulus * I0_3[i],
            ],
        )
    for i in range(0, MaxDimension.value()):
        assert np.all(bend_matrix[i, i, :] > Tolerance.atol()), (
            " Bend matrix has to be greater than 0."
        )

    # Compute bend matrix in Voronoi Domain
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths[1:] + bend_matrix[..., :-1] * rest_lengths[0:-1]
    ) / (rest_lengths[1:] + rest_lengths[:-1])

    # Compute volume of elements
    volume = np.pi * radius**2 * rest_lengths

    # Compute mass of elements
    mass = np.zeros(n_nodes)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    # Generate rest sigma and rest kappa, use user input if defined
    # set rest strains and curvature to be  zero at start
    rest_sigma = np.zeros((MaxDimension.value(), n_elements))
    _assert_shape(rest_sigma, (MaxDimension.value(), n_elements), "rest_sigma")

    rest_kappa = np.zeros((MaxDimension.value(), n_voronoi_elements))
    _assert_shape(rest_kappa, (MaxDimension.value(), n_voronoi_elements), "rest_kappa")

    # Compute rest voronoi length
    rest_voronoi_lengths = 0.5 * (rest_lengths[1:] + rest_lengths[:-1])

    # Allocate arrays for Cosserat Rod equations
    velocities = np.zeros((MaxDimension.value(), n_nodes))
    omegas = np.zeros((MaxDimension.value(), n_elements))
    accelerations = 0.0 * velocities
    angular_accelerations = 0.0 * omegas

    internal_forces = 0.0 * accelerations
    internal_torques = 0.0 * angular_accelerations

    external_forces = 0.0 * accelerations
    external_torques = 0.0 * angular_accelerations

    lengths = np.zeros((n_elements))
    tangents = np.zeros((3, n_elements))

    dilatation = np.zeros((n_elements))
    voronoi_dilatation = np.zeros((n_voronoi_elements))
    dilatation_rate = np.zeros((n_elements))

    sigma = np.zeros((3, n_elements))
    kappa = np.zeros((3, n_voronoi_elements))

    internal_stress = np.zeros((3, n_elements))
    internal_couple = np.zeros((3, n_voronoi_elements))

    return (
        n_elements,
        position,
        velocities,
        omegas,
        accelerations,
        angular_accelerations,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density_array,
        volume,
        mass,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
    )
