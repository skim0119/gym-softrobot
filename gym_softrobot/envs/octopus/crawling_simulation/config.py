"""Physics and rollout parameters for the Octopus-v1 bandit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from gym_softrobot.envs.octopus.control.phase_policy import ArmControlPolicy, OctoArmPolicy
from gym_softrobot.envs.octopus.control.muscle_policy import MuscleArmControlPolicy

RewardMode = Literal["forward", "three_heading_avg"]
FrictionMode = Literal["no-friction", "anisotropic", "isotropic"]

ArmInteractionMask = bool | Sequence[bool] | None


@dataclass(frozen=True, kw_only=True)
class _OctopusMaterialProperties:
    # Octopus morphology
    n_arms: int = 8
    n_channels: int = 6
    rotation_offset: int = 0.5

    # Octopus simulation geometry
    n_elem: int = 15
    base_length: float = 0.45
    base_radius: float = 0.02
    density: float = 5500.0
    youngs_modulus: float = 5.0e5
    base_sphere_radius: float = 0.09
    sphere_density: float = 500.0
    sphere_plane_k: float = 1.0e4
    sphere_plane_nu: float = 0.0  # 0.5
    sphere_plane_mu: float = 0.1
    tether_k: float = 1.0e3
    tether_k_rot: float = 1.0
    sucker_k: float = 1.0e2
    sucker_nu: float = 5.0e2
    sucker_contact_k: float = 0.0  # Not used
    sucker_contact_nu: float = 0.0  # Not used

    sphere_damping_constant: float = 0.05
    sphere_rotational_damping_constant: float = 1.0e-2
    rod_translational_damping_constant: float = 3.0
    rod_rotational_damping_constant: float = 3.0e-3

    # Environment
    rod_surface_friction: FrictionMode = "anisotropic"
    # rod_surface_friction: FrictionMode = "no-friction"
    gravitational_acc: float = -9.80665
    slip_velocity_tol: float = 1e-8
    contact_k: float = 1.0
    contact_nu: float = 1e-6
    froude: float = 1.0
    arm_friction_enabled_list: ArmInteractionMask = field(default=None, repr=False)

    @property
    def rod_friction_enabled(self) -> bool:
        return self.rod_surface_friction != "no-friction"

    @property
    def arm_friction_enabled(self) -> bool:
        return _normalize_arm_interaction_mask(
            self.arm_friction_enabled_list,
            n_arms=self.n_arms,
            default=True,
        )

    def arm_friction_active(self, arm_index: int) -> bool:
        return self.rod_friction_enabled and self.arm_friction_enabled[arm_index]

    def rod_friction_mu_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        gravity = float(self.gravitational_acc)
        mu = self.base_length / (self.T_L**2 * gravity * self.froude)
        mu *= 0.5
        static_mu = np.zeros(3, dtype=np.float64)
        if self.rod_surface_friction == "isotropic":
            kinetic_mu = np.array([mu, mu, mu], dtype=np.float64)
        else:
            kinetic_mu = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
        return static_mu, kinetic_mu

    # Rod-Rod contact
    arm_rod_contact_enabled: bool = False
    rod_rod_contact_k: float = 1.0e4
    rod_rod_contact_nu: float = 3.0
    rod_rod_contact_skip_first_elements: int = 3

    # Numerical control setting
    T_L: float = 2.4
    time_step: float = 3.0e-4  # Simulation timestep
    control_dt: float = 1.0 / 60.0  # Control frequency
    episode_duration_cycles: float = 5.0  #  Duration of the episode in the single "episode"
    episode_horizon: int = 3  #  Temporary value for RL horizon

    @property
    def episode_duration_s(self) -> float:
        return self.episode_duration_cycles * self.T_L

    # Reward settings
    # FIXME: Could be relocated
    reward_mode: RewardMode = "three_heading_avg"
    lateral_penalty: float = 0.5
    vertical_penalty: float = 0.25
    rotation_penalty: float = 0.25
    energy_penalty: float = 0.001
    failure_penalty: float = -100.0

    @property
    def reward_scale(self) -> float:
        """Scale raw displacement fitness so 1.0 ≈ 80% arm length per cycle."""
        return 1.0 / (self.episode_duration_cycles * 0.8 * self.base_length)


# --- Actuations configurations ---


@dataclass(frozen=True, kw_only=True)
class _SuckerControl:
    """Sucker-actuated octopus"""

    n_arms: int

    sucker_enabled: bool = True
    arm_sucker_enabled_list: ArmInteractionMask = field(default=None, repr=False)

    @property
    def arm_sucker_enabled(self) -> bool:
        return _normalize_arm_interaction_mask(
            self.arm_sucker_enabled_list,
            n_arms=self.n_arms,
            default=True,
        )

    def arm_sucker_active(self, arm_index: int) -> bool:
        return self.sucker_enabled and self.arm_sucker_enabled[arm_index]


@dataclass(frozen=True, kw_only=True)
class _OctopusControlSettings(_SuckerControl): ...


# --- Combined configurations ----


@dataclass(frozen=True, kw_only=True)
class OctopusV1Config(_OctopusMaterialProperties, _OctopusControlSettings):
    """Legacy phase-Gaussian octopus (144-dim open-loop policy, 60 Hz control)."""

    @property
    def policy_param_shape(self) -> tuple[int, int]:
        return (self.n_arms, ArmControlPolicy.vector_size())

    @property
    def policy_param_count(self) -> int:
        return int(OctoArmPolicy.vector_size())


@dataclass(frozen=True, kw_only=True)
class OctopusMuscleConfig(_OctopusMaterialProperties, _OctopusControlSettings):
    n_channels: int = 9

    @property
    def policy_param_shape(self) -> tuple[int, int]:
        return (self.n_arms, MuscleArmControlPolicy.vector_size())

    @property
    def policy_param_count(self) -> int:
        return int(self.n_arms * MuscleArmControlPolicy.vector_size())


# --- Utility functions ---


def _normalize_arm_interaction_mask(
    value: ArmInteractionMask,
    *,
    n_arms: int,
    default: bool,
) -> tuple[bool, ...]:
    """
    Expand a scalar or length-``n_arms`` sequence into per-arm booleans.

    If ``value`` is ``None``, return a tuple of ``default`` values for each arm.
    If ``value`` is a boolean, return a tuple of ``value`` values for each arm.
    If ``value`` is a sequence, return a tuple of ``value`` values for each arm.

    Parameters
    ----------
    value : ArmInteractionMask
        Scalar or length-``n_arms`` sequence of booleans
    n_arms : int
        Number of arms
    name : str
        Name of the interaction mask
    default : bool
        Default value for the interaction mask

    Returns
    -------
    tuple[bool, ...]
        Per-arm booleans

    Raises
    ------
    ValueError
        If the length of the interaction mask is not equal to ``n_arms``
    """
    if value is None:
        return tuple(default for _ in range(n_arms))
    if isinstance(value, (bool, np.bool_)):
        return tuple(bool(value) for _ in range(n_arms))
    flags = [bool(v) for v in value]
    if len(flags) != n_arms:
        raise ValueError(f"interaction mask must have length {n_arms}, got {len(flags)}")
    return tuple(flags)
