from __future__ import annotations

import elastica as ea
import numpy as np
from numba import njit
from numpy import arccos, maximum, minimum, sin


class BaseSphereTether(ea.NoContact):
    """Attach a rod node to a sphere with translational and rotational springs.

    The tether applies a linear force from rod node `rod_node_index` toward
    ``sphere_center - rest_offset`` using stiffness `k`, plus a restoring torque
    that penalizes orientation error from `relative_rotation`. Optional `nut`
    adds rotational damping from angular-velocity mismatch.

    ``rest_offset`` is a world-frame vector from the attachment point to the
    sphere center. The default is zero (attach to the center).
    """

    def __init__(
        self,
        k: float,
        rod_node_index: int,
        relative_rotation: np.ndarray,
        *,
        k_rot: float | None = None,
        nut: float = 0.0,
        rest_offset: np.ndarray | None = None,
    ) -> None:
        super().__init__()

        self.k = float(k)
        self.idx = int(rod_node_index)
        self.k_rot = float(k) if k_rot is None else float(k_rot)
        self.nut = float(nut)
        if rest_offset is None:
            self.rest_offset = np.zeros(3, dtype=np.float64)
        else:
            self.rest_offset = np.asarray(rest_offset, dtype=np.float64).reshape(3)

        self._relative_rotation = relative_rotation
        if self._relative_rotation.shape != (3, 3):
            raise ValueError("relative_rotation must have shape (3, 3)")

    @property
    def _allowed_system_two(self) -> list[type]:
        return [ea.Sphere]

    def apply_contact(
        self,
        system_one: object,
        system_two: object,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        _apply_base_sphere_tether_translation(
            self.k,
            self.idx,
            system_one.position_collection,
            system_one.external_forces,
            system_two.position_collection[:, 0],
            system_two.external_forces,
            self.rest_offset,
        )

        _apply_base_sphere_tether_rotation(
            self.k_rot,
            self.nut,
            self.idx,
            system_one.director_collection,
            system_two.director_collection,
            self._relative_rotation,
            system_one.omega_collection,
            system_two.omega_collection,
            system_one.external_torques,
            system_two.external_torques,
        )


class SphereHeadTether(ea.NoContact):
    """Hard-constrain the rod head pose to a sphere pose each contact step.

    This contact adapter copies the sphere position to rod node 0 and resets the
    head director frame to a provided orientation matrix.

    This is one-way only (sphere -> rod) boundary condition.
    It is only used to update the location of an object into the desired position.
    The object should be appropriately damped to prevent it from oscillating.
    This kinds of one-way BC is tends to be unstable.
    """

    def __init__(self, head_orientation: np.ndarray) -> None:
        super().__init__()
        self.head_orientation = head_orientation

    @property
    def _allowed_system_one(self) -> list[type]:
        return [ea.Sphere]

    def apply_contact(
        self,
        system_one: object,
        system_two: object,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        self._reset_head(
            system_one.position_collection,
            system_two.position_collection,
            system_two.director_collection,
            self.head_orientation,
        )

    @staticmethod
    @njit(cache=True)
    def _reset_head(
        sphere_position: np.ndarray,
        position_collection: np.ndarray,
        director_collection: np.ndarray,
        head_orientation: np.ndarray,
    ) -> None:
        position_collection[..., 0] = sphere_position[:, 0]
        director_collection[..., 0] = head_orientation


@njit(cache=True)
def _elastica_inv_rotate_identity_target(
    B: np.ndarray,
) -> np.ndarray:
    r"""Inverse-rotate matrix ``B`` toward identity and return axis-angle vector.

    This is a specialization of the Elastica inverse-rotation map with target
    matrix fixed to identity, i.e. solve

    $$
    \exp([\mathbf{w}]_\times)\,B \approx I.
    $$

    Using the standard skew extraction and trace identity for ``A = I``:

    $$
    \mathbf{v} =
    \begin{bmatrix}
        B_{2,1} - B_{1,2} \\
        B_{0,2} - B_{2,0} \\
        B_{1,0} - B_{0,1}
    \end{bmatrix},\quad
    \theta = \arccos\left(\frac{\operatorname{tr}(B)-1}{2}\right),\quad
    \mathbf{w} = -\frac{1}{2}\frac{\theta}{\sin\theta}\,\mathbf{v}.
    $$
    """
    v0 = B[2, 1] - B[1, 2]
    v1 = B[0, 2] - B[2, 0]
    v2 = B[1, 0] - B[0, 1]
    trace = B[0, 0] + B[1, 1] + B[2, 2]
    trace = minimum(trace, 3.0)
    trace = maximum(trace, -1.0)
    theta = arccos(0.5 * trace - 0.5) + 1e-14
    magnitude = -0.5 * theta / sin(theta)
    return magnitude * np.array([v0, v1, v2])


@njit(cache=True)
def _apply_base_sphere_tether_translation(
    k: float,
    idx: int,
    rod_positions: np.ndarray,
    rod_external_forces: np.ndarray,
    sphere_positions: np.ndarray,
    sphere_external_forces: np.ndarray,
    rest_offset: np.ndarray,
) -> None:
    fx = k * (sphere_positions[0] - rest_offset[0] - rod_positions[0, idx])
    fy = k * (sphere_positions[1] - rest_offset[1] - rod_positions[1, idx])
    fz = k * (sphere_positions[2] - rest_offset[2] - rod_positions[2, idx])

    rod_external_forces[0, idx] += fx
    rod_external_forces[1, idx] += fy
    rod_external_forces[2, idx] += fz
    sphere_external_forces[0, 0] -= fx
    sphere_external_forces[1, 0] -= fy
    sphere_external_forces[2, 0] -= fz


@njit(cache=True)
def _apply_base_sphere_tether_rotation(
    k_rot: float,
    nut: float,
    idx: int,
    rod_directors: np.ndarray,
    sphere_directors: np.ndarray,
    rest_rotation: np.ndarray,
    rod_omega: np.ndarray,
    sphere_omega: np.ndarray,
    rod_external_torques: np.ndarray,
    sphere_external_torques: np.ndarray,
) -> None:
    r"""Apply rotational spring-damper tether torques between rod node and sphere.

    Let ``R_r`` be the rod director at node ``idx`` and ``R_s`` the sphere director.
    The relative orientation and deviation from rest are

    $$
    R_{\mathrm{rel}} = R_r R_s^T,\quad
    R_{\mathrm{dev}} = R_{\mathrm{rel}}^T R_{\mathrm{rest}}.
    $$

    Then the axis-angle error vector is computed as

    $$
    \boldsymbol{\theta} = \mathrm{inv\_rotate}(I, R_{\mathrm{dev}}^T),
    $$

    Then expressed in inertial coordinates and combined with angular-velocity
    mismatch to form a spring-damper torque:

    $$
    \Delta\boldsymbol{\omega}
    = \left(R_s^T \boldsymbol{\omega}_s\right)
      - \left(R_r^T \boldsymbol{\omega}_r\right),
    $$
    $$
    \boldsymbol{\tau}
    = k_{\mathrm{rot}}\,\boldsymbol{\theta}
      - \nu_t\,\Delta\boldsymbol{\omega}.
    $$

    Equal-and-opposite torques are applied to rod and sphere in their respective
    local frames, preserving action-reaction coupling.
    """
    rel_rot = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for t in range(3):
                rel_rot[i, j] += rod_directors[i, t, idx] * sphere_directors[j, t, 0]

    dev_rot_T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for t in range(3):
                dev_rot_T[j, i] += rel_rot[t, i] * rest_rotation[t, j]

    # Rotation vector in inertial coordinates
    rot_vec = _elastica_inv_rotate_identity_target(dev_rot_T)
    rot_vec_inertial = np.zeros(3)
    for i in range(3):
        for j in range(3):
            rot_vec_inertial[i] += sphere_directors[j, i, 0] * rot_vec[j]

    # Dissipation by angular velocity mismatch
    dev_omega = np.zeros(3)
    for i in range(3):
        for j in range(3):
            dev_omega[i] += sphere_directors[j, i, 0] * sphere_omega[j, 0]
            dev_omega[i] -= rod_directors[j, i, idx] * rod_omega[j, idx]

    # Total torque applied
    torque = k_rot * rot_vec_inertial - nut * dev_omega

    # Torque applied in local coordinates
    for i in range(3):
        for j in range(3):
            sphere_external_torques[i, 0] += sphere_directors[i, j, 0] * torque[j]
            rod_external_torques[i, idx] -= rod_directors[i, j, idx] * torque[j]
