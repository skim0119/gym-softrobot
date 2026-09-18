"""Legacy phase-Gaussian octopus simulation (60 Hz, open-loop ``OctoArmPolicy``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeAlias, TypeVar

import elastica as ea
import numpy as np
import numpy.typing as npt

from gym_softrobot.envs.octopus.phase_physics.crawling import SegmentExtensionActuation
from gym_softrobot.envs.octopus.phase_physics.config import OctopusV1Config
from gym_softrobot.envs.octopus.phase_physics.policy import OctoArmPolicy
from gym_softrobot.envs.octopus.phase_physics.forcing import SuckerActuation, YSurfaceBallwGravity
from gym_softrobot.envs.octopus.phase_physics.dissipation import RayleighDamping
from gym_softrobot.envs.octopus.phase_physics.tapered_arm import create_arm
from gym_softrobot.envs.octopus.phase_physics.contacts import BaseSphereTether
from gym_softrobot.envs.octopus.phase_physics.friction import AnisotropicFriction
from gym_softrobot.envs.octopus.phase_physics.rod_rod_contact import (
    RodRodContactSkipBaseElements,
)

Vector3: TypeAlias = tuple[float, float, float] | npt.NDArray[np.float64]


def all_finite(array: npt.ArrayLike) -> np.bool_:
    return np.all(np.isfinite(array))


def rotation_delta(
    current_director: npt.NDArray[np.float64],
    initial_director: npt.NDArray[np.float64],
) -> float:
    relative_rotation = (
        np.asarray(current_director, dtype=np.float64)
        @ np.asarray(initial_director, dtype=np.float64).T
    )
    trace_value = float(np.trace(relative_rotation))
    return float(np.arccos(np.clip(0.5 * (trace_value - 1.0), -1.0, 1.0)))


class _Simulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


@dataclass(frozen=True)
class PhaseInstanceRollout:
    """Heading-agnostic physics trace for one octopus instance."""

    observation: npt.NDArray[np.float32]
    failed: bool
    horizontal: npt.NDArray[np.float64]
    vertical: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    displacement: float


ACTUATION_CHANNEL_NAMES: tuple[str, ...] = (
    "stiffness",
    "extension",
    "base_suction",
    "middle_suction",
    "bend",
)


@dataclass
class PhaseEpisodeDiagnostics:
    time: list[float] = field(default_factory=list)
    sphere_position: list[npt.NDArray[np.float64]] = field(default_factory=list)
    sphere_director: list[npt.NDArray[np.float64]] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    forward_progress: list[float] = field(default_factory=list)
    lateral_deviation: list[float] = field(default_factory=list)
    vertical_deviation: list[float] = field(default_factory=list)
    rotation_delta: list[float] = field(default_factory=list)
    strain_energy: list[float] = field(default_factory=list)
    rod_centerlines: list[npt.NDArray[np.float64]] = field(default_factory=list)
    rod_directors: list[npt.NDArray[np.float64]] = field(default_factory=list)
    rod_radius: list[npt.NDArray[np.float64]] = field(default_factory=list)
    actuation: list[npt.NDArray[np.float64]] = field(default_factory=list)


@dataclass
class _OctopusInstance:
    policy: Any
    base_sphere: Any
    rods: list[Any]
    base_position: npt.NDArray[np.float64]
    initial_sphere_position: npt.NDArray[np.float64]
    initial_sphere_director: npt.NDArray[np.float64]


C = TypeVar("C", bound=OctopusV1Config)


class PhaseOctopusSimulation(Generic[C]):
    """One or more octopuses driven by open-loop policies in a shared Elastica world."""

    PolicyCls: type = OctoArmPolicy

    def __init__(
        self,
        config: C,
        *,
        n_instances: int = 1,
        fix_sphere: bool = False,
    ) -> None:
        if n_instances < 1:
            raise ValueError(f"n_instances must be >= 1, got {n_instances}")
        self.config = config
        self.n_instances = int(n_instances)
        self.time = 0.0
        self.timestepper = ea.PositionVerlet()
        self.simulator = _Simulator()

        self.base_range = (1, 3)
        self.middle_range = (6, 8)
        self.instances: list[_OctopusInstance] = []
        self._build_simulation(fix_sphere)
        self._last_diagnostics: list[PhaseEpisodeDiagnostics | None] = [
            None for _ in range(self.n_instances)
        ]

    @property
    def policies(self) -> list[Any]:
        return [instance.policy for instance in self.instances]

    @property
    def policy(self) -> Any:
        return self.instances[0].policy

    @property
    def base_sphere(self) -> Any:
        return self.instances[0].base_sphere

    @property
    def rods(self) -> list[Any]:
        return self.instances[0].rods

    def _build_simulation(self, fix_sphere: bool = False) -> None:
        cfg = self.config
        static_mu, kinetic_mu = cfg.rod_friction_mu_arrays()
        plane_y = -cfg.base_radius
        plane_origin = np.array([0.0, plane_y, 0.0], dtype=np.float64)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        for instance_index in range(self.n_instances):
            base_position = np.zeros(3, dtype=np.float64)
            policy = self.PolicyCls.default(T_L=float(cfg.T_L), n_arms=cfg.n_arms)
            base_sphere = ea.Sphere(
                base_position.copy(),
                cfg.base_sphere_radius,
                cfg.sphere_density,
            )
            self.simulator.append(base_sphere)

            if fix_sphere:
                from gym_softrobot.envs.octopus.phase_physics.boundary_condition import FixSphere

                self.simulator.constrain(base_sphere).using(
                    FixSphere,
                    fixed_position_idx=(0,),
                    fixed_director_idx=(0,),
                )
            else:
                self.simulator.add_forcing_to(base_sphere).using(
                    YSurfaceBallwGravity,
                    k_c=cfg.sphere_plane_k,
                    nu_c=cfg.sphere_plane_nu,
                    mu=cfg.sphere_plane_mu,
                    plane_origin=-cfg.base_sphere_radius,
                )
                self.simulator.dampen(base_sphere).using(
                    RayleighDamping,
                    damping_constant=cfg.sphere_damping_constant,
                    rotational_damping_constant=cfg.sphere_rotational_damping_constant,
                    time_step=cfg.time_step,
                )

            rods: list[Any] = []
            for arm_index in range(cfg.n_arms):
                angle = 2.0 * np.pi * (arm_index + cfg.rotation_offset) / cfg.n_arms
                direction = np.array([np.sin(angle), 0.0, -np.cos(angle)], dtype=np.float64)
                normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
                rod = create_arm(
                    cfg.n_elem,
                    base_position.copy(),
                    direction,
                    normal,
                    cfg.base_length,
                    cfg.base_radius,
                    cfg.density,
                    cfg.youngs_modulus,
                )
                rods.append(rod)
                self.simulator.append(rod)

                self.simulator.detect_contact_between(rod, base_sphere).using(
                    BaseSphereTether,
                    k=cfg.tether_k,
                    k_rot=cfg.tether_k_rot,
                    rod_node_index=0,
                    relative_rotation=rod.director_collection[..., 0]
                    @ base_sphere.director_collection[..., 0].T,
                )
                if cfg.arm_friction_active(arm_index):
                    self.simulator.add_forcing_to(rod).using(
                        ea.GravityForces,
                        acc_gravity=np.array(
                            [0.0, float(cfg.gravitational_acc), 0.0],
                            dtype=np.float64,
                        ),
                    )
                    self.simulator.add_forcing_to(rod).using(
                        AnisotropicFriction,
                        k=cfg.contact_k,
                        nu=cfg.contact_nu,
                        slip_velocity_tol=cfg.slip_velocity_tol,
                        static_mu_array=static_mu,
                        kinetic_mu_array=kinetic_mu,
                        plane_origin=plane_origin,
                        plane_normal=plane_normal,
                        surface_tol=1.0e-4,
                    )
                self.simulator.dampen(rod).using(
                    ea.AnalyticalLinearDamper,
                    translational_damping_constant=cfg.rod_translational_damping_constant,
                    rotational_damping_constant=cfg.rod_rotational_damping_constant,
                    time_step=cfg.time_step,
                )

                self._add_actuators(
                    self.simulator,
                    rod,
                    cfg,
                    instance_index,
                    arm_index,
                    plane_origin,
                    plane_normal,
                    policy,
                )

            if cfg.arm_rod_contact_enabled:
                for arm_i in range(len(rods)):
                    for arm_j in range(arm_i + 1, len(rods)):
                        self.simulator.detect_contact_between(
                            rods[arm_i],
                            rods[arm_j],
                        ).using(
                            RodRodContactSkipBaseElements,
                            k=float(cfg.rod_rod_contact_k),
                            nu=float(cfg.rod_rod_contact_nu),
                            skip_first_elements=int(cfg.rod_rod_contact_skip_first_elements),
                        )

            initial_sphere_position = np.asarray(
                base_sphere.position_collection[..., 0], dtype=np.float64
            ).copy()
            initial_sphere_director = np.asarray(
                base_sphere.director_collection[..., 0], dtype=np.float64
            ).copy()
            self.instances.append(
                _OctopusInstance(
                    policy=policy,
                    base_sphere=base_sphere,
                    rods=rods,
                    base_position=base_position,
                    initial_sphere_position=initial_sphere_position,
                    initial_sphere_director=initial_sphere_director,
                )
            )

        self.simulator.finalize()

    @property
    def locomotion_phase(self) -> float:
        """Phase in [0, 1] for the current time."""
        return float((self.time % self.config.T_L) / self.config.T_L)

    def advance(self, duration: float, *, apply_phase: bool = True) -> None:
        if apply_phase:
            phase = self.locomotion_phase
            for instance in self.instances:
                instance.policy.apply_phase(phase)
        substeps = max(1, int(np.ceil(duration / self.config.time_step)))
        step_dt = duration / substeps
        for _ in range(substeps):
            self.time = self.timestepper.step(self.simulator, self.time, step_dt)

    def sphere_position(self, instance_index: int = 0) -> npt.NDArray[np.float64]:
        sphere = self.instances[instance_index].base_sphere
        return np.asarray(sphere.position_collection[..., 0], dtype=np.float64)

    def sphere_velocity(self, instance_index: int = 0) -> npt.NDArray[np.float64]:
        sphere = self.instances[instance_index].base_sphere
        return np.asarray(sphere.velocity_collection[..., 0], dtype=np.float64)

    def observation(self, instance_index: int = 0) -> npt.NDArray[np.float64]:
        instance = self.instances[instance_index]
        displacement = self.sphere_position(instance_index) - instance.initial_sphere_position
        return np.concatenate(
            [
                displacement.astype(np.float32),
                self.sphere_velocity(instance_index).astype(np.float32),
            ]
        )

    def strain_energy(self, instance_index: int = 0) -> float:
        total = 0.0
        for rod in self.instances[instance_index].rods:
            total += float(rod.compute_bending_energy()) + float(rod.compute_shear_energy())
        return total

    def is_finite(self) -> bool:
        for instance in self.instances:
            for rod in instance.rods:
                if not all_finite(rod.position_collection):
                    return False
                if not all_finite(rod.velocity_collection):
                    return False
                if not all_finite(rod.sigma):
                    return False
                if not all_finite(rod.kappa):
                    return False
        return True

    def _failed_rollout(self, instance_index: int) -> PhaseInstanceRollout:
        return PhaseInstanceRollout(
            observation=self.observation(instance_index).astype(np.float32),
            failed=True,
            horizontal=np.zeros((0, 2), dtype=np.float64),
            vertical=np.zeros(0, dtype=np.float64),
            rotation=np.zeros(0, dtype=np.float64),
            strain=np.zeros(0, dtype=np.float64),
            displacement=0.0,
        )

    def run_episode(
        self,
        *,
        record_diagnostics: bool = False,
    ) -> list[PhaseInstanceRollout]:
        """Integrate for ``episode_duration_cycles * T_L`` at legacy ``control_dt``."""
        cfg = self.config
        duration = cfg.episode_duration_s
        diags = [
            PhaseEpisodeDiagnostics() if record_diagnostics else None
            for _ in range(self.n_instances)
        ]
        horizontal_steps: list[list[npt.NDArray[np.float64]]] = [
            [] for _ in range(self.n_instances)
        ]
        vertical_steps: list[list[float]] = [[] for _ in range(self.n_instances)]
        rotation_steps: list[list[float]] = [[] for _ in range(self.n_instances)]
        strain_steps: list[list[float]] = [[] for _ in range(self.n_instances)]

        elapsed = 0.0
        while elapsed < duration - 1.0e-12:
            self.advance(cfg.control_dt)
            if not self.is_finite():
                self._last_diagnostics = diags
                return [self._failed_rollout(index) for index in range(self.n_instances)]

            for index, instance in enumerate(self.instances):
                displacement = self.sphere_position(index) - instance.initial_sphere_position
                horizontal = np.asarray([displacement[0], displacement[2]], dtype=np.float64)
                vertical_deviation = float(abs(displacement[1]))
                step_rotation_delta = rotation_delta(
                    np.asarray(
                        instance.base_sphere.director_collection[..., 0],
                        dtype=np.float64,
                    ),
                    instance.initial_sphere_director,
                )
                strain = self.strain_energy(index)
                horizontal_steps[index].append(horizontal.copy())
                vertical_steps[index].append(vertical_deviation)
                rotation_steps[index].append(step_rotation_delta)
                strain_steps[index].append(strain)

                diag = diags[index]
                if diag is not None:
                    diag.time.append(float(self.time))
                    diag.sphere_position.append(self.sphere_position(index).copy())
                    diag.sphere_director.append(
                        np.asarray(
                            instance.base_sphere.director_collection[..., 0],
                            dtype=np.float64,
                        ).copy()
                    )
                    diag.vertical_deviation.append(vertical_deviation)
                    diag.rotation_delta.append(step_rotation_delta)
                    diag.strain_energy.append(strain)
                    diag.rod_centerlines.append(
                        np.stack(
                            [
                                np.asarray(rod.position_collection, dtype=np.float64).copy()
                                for rod in instance.rods
                            ],
                            axis=0,
                        )
                    )
                    diag.rod_directors.append(
                        np.stack(
                            [
                                np.asarray(rod.director_collection, dtype=np.float64).copy()
                                for rod in instance.rods
                            ],
                            axis=0,
                        )
                    )
                    diag.rod_radius.append(
                        np.stack(
                            [
                                np.asarray(rod.radius, dtype=np.float64).copy()
                                for rod in instance.rods
                            ],
                            axis=0,
                        )
                    )
                    diag.actuation.append(instance.policy.actuation_vector())

            elapsed += cfg.control_dt

        self._last_diagnostics = diags
        rollouts: list[PhaseInstanceRollout] = []
        for index in range(self.n_instances):
            end_disp = (
                self.sphere_position(index) - self.instances[index].initial_sphere_position
            )
            planar_disp = float(np.linalg.norm(end_disp[[0, 2]]))
            rollouts.append(
                PhaseInstanceRollout(
                    observation=self.observation(index).astype(np.float32),
                    failed=False,
                    horizontal=np.stack(horizontal_steps[index], axis=0),
                    vertical=np.asarray(vertical_steps[index], dtype=np.float64),
                    rotation=np.asarray(rotation_steps[index], dtype=np.float64),
                    strain=np.asarray(strain_steps[index], dtype=np.float64),
                    displacement=planar_disp,
                )
            )
        return rollouts

    def _add_actuators(
        self,
        simulator,
        rod,
        cfg,
        instance_index: int,
        arm_index: int,
        plane_origin,
        plane_normal,
        policy,
    ) -> None:
        del policy
        simulator.add_forcing_to(rod).using(
            SuckerActuation,
            k=cfg.sucker_k,
            nu=cfg.sucker_nu,
            k_c=0.0,  # cfg.sucker_contact_k,
            nu_c=0.0,  # cfg.sucker_contact_nu,
            forces_enabled=cfg.arm_sucker_active(arm_index),
            trigger=lambda inst=instance_index, arm=arm_index: (
                self.instances[inst].policy.middle_suction_active[arm],
                self.instances[inst].policy.base_suction_active[arm],
            ),
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            start_index=[self.middle_range[0], self.base_range[0]],
            end_index=[self.middle_range[1], self.base_range[1]],
        )
        simulator.add_forcing_to(rod).using(
            SegmentExtensionActuation,
            original_shear_matrix=rod.shear_matrix.copy(),
            original_bend_matrix=rod.bend_matrix.copy(),
            start_index=self.base_range[1],
            end_index=self.middle_range[0],
            amplitude=lambda inst=instance_index, arm=arm_index: (
                self.instances[inst].policy.target_extension[arm],
                self.instances[inst].policy.target_stiffness[arm],
                self.instances[inst].policy.target_bend[arm],
            ),
        )
