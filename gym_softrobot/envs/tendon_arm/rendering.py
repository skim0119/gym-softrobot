"""Radius-aware rendering helpers for the tendon-driven arm."""

from __future__ import annotations

import numpy as np


def add_tapered_rod(
    axis,
    positions: np.ndarray,
    element_radii: np.ndarray,
    *,
    color: str = "tab:blue",
    alpha: float = 0.95,
    n_sides: int = 12,
):
    """Draw a tube following the centerline with the rod's element radii.

    ``positions`` has shape ``(3, n_nodes)`` and ``element_radii`` has one
    radius per adjacent node pair. A circular cross-section is constructed
    perpendicular to the local centerline tangent.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgb

    centerline = np.asarray(positions, dtype=np.float64).T
    element_radii = np.asarray(element_radii, dtype=np.float64)
    if centerline.shape[0] != element_radii.size + 1:
        raise ValueError("positions must have exactly one more node than radii")

    node_radii = np.empty(centerline.shape[0])
    node_radii[0] = element_radii[0]
    node_radii[-1] = element_radii[-1]
    node_radii[1:-1] = 0.5 * (element_radii[:-1] + element_radii[1:])

    tangents = np.gradient(centerline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(tangent_norms[:, None], np.finfo(np.float64).eps)
    reference = np.tile((0.0, 0.0, 1.0), (len(centerline), 1))
    parallel = np.abs(tangents[:, 2]) > 0.95
    reference[parallel] = (1.0, 0.0, 0.0)
    basis_1 = np.cross(tangents, reference)
    basis_1 /= np.maximum(
        np.linalg.norm(basis_1, axis=1)[:, None], np.finfo(np.float64).eps
    )
    basis_2 = np.cross(tangents, basis_1)

    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    ring = centerline[:, None, :] + node_radii[:, None, None] * (
        np.cos(angles)[None, :, None] * basis_1[:, None, :]
        + np.sin(angles)[None, :, None] * basis_2[:, None, :]
    )
    faces = []
    for node_index in range(len(centerline) - 1):
        for side_index in range(n_sides):
            next_side = (side_index + 1) % n_sides
            faces.append(
                (
                    ring[node_index, side_index],
                    ring[node_index, next_side],
                    ring[node_index + 1, next_side],
                    ring[node_index + 1, side_index],
                )
            )
    faces.extend((ring[0], ring[-1]))
    light_direction = np.array((-0.45, 0.55, 0.70))
    light_direction /= np.linalg.norm(light_direction)
    base_color = np.asarray(to_rgb(color))
    face_colors = []
    for face in faces:
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal_norm = np.linalg.norm(normal)
        illumination = (
            abs(float(np.dot(normal / normal_norm, light_direction)))
            if normal_norm > np.finfo(np.float64).eps
            else 0.0
        )
        face_colors.append(base_color * (0.35 + 0.65 * illumination))
    collection = Poly3DCollection(
        faces,
        facecolor=face_colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=alpha,
    )
    axis.add_collection3d(collection)
    return collection
