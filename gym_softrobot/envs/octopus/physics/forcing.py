from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numba import njit

if not hasattr(np, "typing"):
    import numpy.typing as np_typing

    np.typing = np_typing  # type: ignore[attr-defined]

from elastica import NoForces
from elastica.typing import RigidBodyType, RodType
from gym_softrobot.utils.custom_elastica.compat import _calculate_contact_forces_rod_plane


def _normalize_sucker_intervals(
    start_index: int | Sequence[int],
    end_index: int | None | Sequence[int | None],
) -> tuple[list[int], list[int | None]]:
    """Expand scalar or parallel list pairs into (starts, ends)."""
    if isinstance(start_index, (int, np.integer)):
        return [int(start_index)], [None if end_index is None else int(end_index)]
    starts = [int(s) for s in start_index]
    if end_index is None:
        ends: list[int | None] = [None] * len(starts)
    elif isinstance(end_index, (int, np.integer)):
        ends = [int(end_index)] * len(starts)
    else:
        ends = [None if e is None else int(e) for e in end_index]
    return starts, ends


class SuckerActuation(NoForces):
    def __init__(
        self,
        k: float,
        nu: float,
        k_c: float,
        nu_c: float,
        trigger: Callable[[], tuple[float, float]],
        *,
        capture_distance: float = 0.5,
        min_distance: float = 1.0e-6,
        plane_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        plane_normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
        start_index: int | Sequence[int] = 0,
        end_index: int | None | Sequence[int | None] = None,
        forces_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.forces_enabled = bool(forces_enabled)
        self.k = float(k)
        self.nu = float(nu)
        self.capture_distance = float(capture_distance)
        self.min_distance = float(min_distance)
        self.k_c = float(k_c)
        self.nu_c = float(nu_c)
        self.trigger = trigger
        self._starts, self._ends = _normalize_sucker_intervals(start_index, end_index)

        plane_origin_array = np.asarray(plane_origin, dtype=np.float64).reshape(3, 1)
        plane_normal_array = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(plane_normal_array))
        if not np.isclose(normal_norm, 1.0, atol=1e-8):
            raise ValueError("plane_normal must be a unit vector")
        self.plane_origin = plane_origin_array
        self.plane_normal = plane_normal_array
        self._plane_origin_flat = np.ascontiguousarray(plane_origin_array.reshape(3))
        self._plane_normal_flat = np.ascontiguousarray(plane_normal_array)

    def apply_forces(
        self,
        system: RodType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        if not self.forces_enabled:
            return

        triggers = self.trigger()
        if all(t < 1e-3 for t in triggers):
            return

        surface_tol = 1e-4
        if (self.k_c > 0.0 or self.nu_c > 0.0) and any(t >= 1e-3 for t in triggers):
            _calculate_contact_forces_rod_plane(
                self.plane_origin,
                self.plane_normal,
                surface_tol,
                self.k_c,
                self.nu_c,
                system.radius,
                system.mass,
                system.position_collection,
                system.velocity_collection,
                system.internal_forces,
                system.external_forces,
            )

        for trig, si, ei in zip(triggers, self._starts, self._ends):
            if trig < 1e-3:
                continue
            _compute_sucker_plane_force_direct(
                system.position_collection,
                system.velocity_collection,
                system.director_collection,
                system.radius,
                self._plane_origin_flat,
                self._plane_normal_flat,
                self.k,
                self.nu,
                trig,
                self.min_distance,
                self.capture_distance,
                si,
                -1 if ei is None else ei,
                system.external_forces,
                system.external_torques,
            )


@njit(cache=True)
def _compute_sucker_plane_force_direct(
    rod_positions: np.ndarray,
    rod_velocities: np.ndarray,
    rod_directors: np.ndarray,
    rod_radii: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    stiffness: float,
    transverse_damping: float,
    trigger_strength: float,
    min_distance: float,
    capture_distance: float,
    start_index: int,
    end_index: int,
    rod_external_forces: np.ndarray,
    rod_external_torques: np.ndarray,
) -> None:
    element_count = rod_radii.shape[0]
    start = start_index if start_index >= 0 else element_count + start_index
    start = 0 if start < 0 else start if start < element_count else element_count

    if end_index < 0:
        stop = element_count if end_index == -1 else element_count + end_index
    else:
        stop = end_index + 1 if end_index < element_count else element_count

    for idx in range(start, stop):
        element_center_x = 0.5 * (rod_positions[0, idx + 1] + rod_positions[0, idx])
        element_center_y = 0.5 * (rod_positions[1, idx + 1] + rod_positions[1, idx])
        element_center_z = 0.5 * (rod_positions[2, idx + 1] + rod_positions[2, idx])

        sucker_axis_x = rod_directors[0, 0, idx]
        sucker_axis_y = rod_directors[0, 1, idx]
        sucker_axis_z = rod_directors[0, 2, idx]

        radius = rod_radii[idx]
        lever_arm_x = sucker_axis_x * radius
        lever_arm_y = sucker_axis_y * radius
        lever_arm_z = sucker_axis_z * radius
        sucker_position_x = element_center_x + lever_arm_x
        sucker_position_y = element_center_y + lever_arm_y
        sucker_position_z = element_center_z + lever_arm_z

        signed_dist = (
            (sucker_position_x - plane_origin[0]) * plane_normal[0]
            + (sucker_position_y - plane_origin[1]) * plane_normal[1]
            + (sucker_position_z - plane_origin[2]) * plane_normal[2]
        )
        positive_dist = max(signed_dist, 0.0)
        if positive_dist > capture_distance:
            continue

        scale = capture_distance if capture_distance > min_distance else min_distance
        weight = np.exp(-((positive_dist / scale) ** 2) * 9.0)

        velocity_x = 0.5 * (rod_velocities[0, idx + 1] + rod_velocities[0, idx])
        velocity_y = 0.5 * (rod_velocities[1, idx + 1] + rod_velocities[1, idx])
        velocity_z = 0.5 * (rod_velocities[2, idx + 1] + rod_velocities[2, idx])

        normal_speed = (
            velocity_x * plane_normal[0]
            + velocity_y * plane_normal[1]
            + velocity_z * plane_normal[2]
        )
        tangent_x = velocity_x - normal_speed * plane_normal[0]
        tangent_y = velocity_y - normal_speed * plane_normal[1]
        tangent_z = velocity_z - normal_speed * plane_normal[2]

        force_mag = trigger_strength * stiffness * weight * positive_dist
        force_x = -force_mag * plane_normal[0] - trigger_strength * transverse_damping * tangent_x
        force_y = -force_mag * plane_normal[1] - trigger_strength * transverse_damping * tangent_y
        force_z = -force_mag * plane_normal[2] - trigger_strength * transverse_damping * tangent_z

        rod_external_forces[0, idx] += 0.5 * force_x
        rod_external_forces[1, idx] += 0.5 * force_y
        rod_external_forces[2, idx] += 0.5 * force_z
        rod_external_forces[0, idx + 1] += 0.5 * force_x
        rod_external_forces[1, idx + 1] += 0.5 * force_y
        rod_external_forces[2, idx + 1] += 0.5 * force_z

        # r x F is in lab frame; external_torques are material-frame (director) components.
        torque_lab_x = lever_arm_y * force_z - lever_arm_z * force_y
        torque_lab_y = lever_arm_z * force_x - lever_arm_x * force_z
        torque_lab_z = lever_arm_x * force_y - lever_arm_y * force_x
        rod_external_torques[0, idx] += (
            rod_directors[0, 0, idx] * torque_lab_x
            + rod_directors[0, 1, idx] * torque_lab_y
            + rod_directors[0, 2, idx] * torque_lab_z
        )
        rod_external_torques[1, idx] += (
            rod_directors[1, 0, idx] * torque_lab_x
            + rod_directors[1, 1, idx] * torque_lab_y
            + rod_directors[1, 2, idx] * torque_lab_z
        )
        rod_external_torques[2, idx] += (
            rod_directors[2, 0, idx] * torque_lab_x
            + rod_directors[2, 1, idx] * torque_lab_y
            + rod_directors[2, 2, idx] * torque_lab_z
        )


class YSurfaceBallwGravity(NoForces):
    """Rigid sphere on a horizontal floor with gravity and dry friction.

    Applies standard gravity in ``-Y``. Contact uses a Hertz-style normal law
    against the plane ``y = plane_origin`` (penetration when the sphere center
    is below ``plane_origin + radius``). Tangential Coulomb friction opposes
    motion in the ``XZ`` plane when in contact. Scalar ``plane_origin`` is the
    world ``y`` coordinate of the surface; the plane normal is fixed to ``+Y``.
    """

    def __init__(
        self,
        k_c: float,
        nu_c: float,
        *,
        mu: float = 0.1,
        plane_origin: float = 0.0,
        min_normal: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.k_c = float(k_c)
        self.nu_c = float(nu_c)
        self.mu = float(mu)
        self.plane_origin = float(plane_origin)
        self.min_normal = float(min_normal)

    def apply_forces(
        self,
        system: RigidBodyType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        system.external_forces[1, 0] -= 9.81 * system.mass[0]
        self._apply_sphere_plane_hertz(
            system.position_collection,
            system.velocity_collection,
            system.external_forces,
            float(system.radius),
            self.plane_origin,
            self.k_c,
            self.nu_c,
            self.mu,
        )

    @staticmethod
    @njit(cache=True)
    def _apply_sphere_plane_hertz(
        sphere_center: np.ndarray,
        sphere_velocity: np.ndarray,
        sphere_external_force: np.ndarray,
        sphere_radius: float,
        plane_origin: float,
        k_c: float,
        nu_c: float,
        mu: float,
    ) -> None:
        signed_distance = sphere_center[1, 0] - plane_origin
        penetration = sphere_radius - signed_distance
        if penetration <= 0.0:
            return

        normal_speed = sphere_velocity[1, 0]
        contact_magnitude = k_c * penetration ** (3 / 2) - nu_c * normal_speed
        if contact_magnitude > 0.0:
            sphere_external_force[1, 0] += contact_magnitude

        # Transverse friction
        tx = sphere_velocity[0, 0]
        tz = sphere_velocity[2, 0]
        speed_t_sq = tx * tx + tz * tz
        if speed_t_sq > 1.0e-9:
            speed_t = np.sqrt(speed_t_sq)
            f_t = mu * contact_magnitude
            s = -f_t / speed_t
            sphere_external_force[0, 0] += s * tx
            sphere_external_force[2, 0] += s * tz


class YSurfaceRodwGravity(NoForces):
    """Cosserat rod on a horizontal floor with gravity and dry friction.

    Applies gravity in ``-Y`` at rod nodes. Contact uses the existing
    rod-plane normal response against ``y = plane_origin`` with fixed ``+Y``
    surface normal. A local Coulomb-style tangential friction term is applied
    on contacting elements and distributed to adjacent nodes.
    """

    def __init__(
        self,
        k_c: float,
        nu_c: float,
        *,
        mu: float = 5.0,
        plane_origin: float = 0.0,
    ) -> None:
        super().__init__()
        self.k_c = float(k_c)
        self.nu_c = float(nu_c)
        self.mu = float(mu)
        self.plane_origin = float(plane_origin)
        self._plane_origin = np.array([[0.0], [self.plane_origin], [0.0]], dtype=np.float64)
        self._plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self._surface_tol = 1e-4

    def apply_forces(
        self,
        system: RodType,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        _apply_uniform_gravity_to_nodes(system.external_forces, system.mass, 9.81)
        _calculate_contact_forces_rod_plane(
            self._plane_origin,
            self._plane_normal,
            self._surface_tol,
            self.k_c,
            self.nu_c,
            system.radius,
            system.mass,
            system.position_collection,
            system.velocity_collection,
            system.internal_forces,
            system.external_forces,
        )
        _apply_rod_plane_coulomb_friction(
            system.position_collection,
            system.velocity_collection,
            system.radius,
            system.external_forces,
            self.plane_origin,
            self.k_c,
            self.nu_c,
            self.mu,
        )


@njit(cache=True)
def _apply_uniform_gravity_to_nodes(
    external_forces: np.ndarray,
    nodal_mass: np.ndarray,
    gravity_mag: float,
) -> None:
    for i in range(nodal_mass.shape[0]):
        external_forces[1, i] -= gravity_mag * nodal_mass[i]


@njit(cache=True)
def _apply_rod_plane_coulomb_friction(
    rod_positions: np.ndarray,
    rod_velocities: np.ndarray,
    rod_radii: np.ndarray,
    rod_external_forces: np.ndarray,
    plane_origin: float,
    k_c: float,
    nu_c: float,
    mu: float,
) -> None:
    element_count = rod_radii.shape[0]
    for idx in range(element_count):
        center_y = 0.5 * (rod_positions[1, idx] + rod_positions[1, idx + 1])
        penetration = rod_radii[idx] - (center_y - plane_origin)
        if penetration <= 0.0:
            continue

        vy = 0.5 * (rod_velocities[1, idx] + rod_velocities[1, idx + 1])
        normal_force = k_c * penetration ** (3.0 / 2.0) - nu_c * vy
        if normal_force <= 0.0:
            continue

        vx = 0.5 * (rod_velocities[0, idx] + rod_velocities[0, idx + 1])
        vz = 0.5 * (rod_velocities[2, idx] + rod_velocities[2, idx + 1])
        speed_t_sq = vx * vx + vz * vz
        if speed_t_sq <= 1.0e-12:
            continue

        speed_t = np.sqrt(speed_t_sq)
        scale = -(mu * normal_force) / speed_t
        fx = scale * vx
        fz = scale * vz

        rod_external_forces[0, idx] += 0.5 * fx
        rod_external_forces[2, idx] += 0.5 * fz
        rod_external_forces[0, idx + 1] += 0.5 * fx
        rod_external_forces[2, idx + 1] += 0.5 * fz
