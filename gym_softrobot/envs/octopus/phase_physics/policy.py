"""Open-loop phase-Gaussian policies for the legacy octopus crawler."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit


@dataclass
class ArmControlPolicy:
    stiffness_center: float = 0.33
    stiffness_deviation: float = 0.2
    stiffness_scale: float = 0.0
    extension_center: float = 0.33
    extension_deviation: float = 0.10
    extension_scale: float = 0.5
    contraction_center: float = 0.66
    contraction_deviation: float = 0.20
    contraction_scale: float = 0.2
    base_suction_center: float = 0.33
    base_suction_deviation: float = 0.2
    base_suction_scale: float = 0.7
    middle_suction_center: float = 0.66
    middle_suction_deviation: float = 0.20
    middle_suction_scale: float = 0.7
    bend_center: float = 0.5
    bend_deviation: float = 0.10
    bend_scale: float = 0.0

    PARAMETER_NAMES = (
        "stiffness_center",
        "stiffness_deviation",
        "stiffness_scale",
        "extension_center",
        "extension_deviation",
        "extension_scale",
        "contraction_center",
        "contraction_deviation",
        "contraction_scale",
        "base_suction_center",
        "base_suction_deviation",
        "base_suction_scale",
        "middle_suction_center",
        "middle_suction_deviation",
        "middle_suction_scale",
        "bend_center",
        "bend_deviation",
        "bend_scale",
    )

    @classmethod
    def vector_size(cls) -> int:
        return len(cls.PARAMETER_NAMES)

    def lower_bounds(self) -> np.ndarray:
        """Per-arm box constraints for CMA-ES / clipping (phase in [0, 1], positive deviations)."""
        return np.asarray(
            [
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                -np.pi / 2,
            ],
            dtype=np.float64,
        )

    def upper_bounds(self) -> np.ndarray:
        return np.asarray(
            [
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                np.pi / 2,
            ],
            dtype=np.float64,
        )

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in self.PARAMETER_NAMES],
            dtype=np.float64,
        )

    def __getitem__(self, index: int) -> float:
        return float(getattr(self, self.PARAMETER_NAMES[index]))

    def __setitem__(self, index: int, value: float) -> None:
        setattr(self, self.PARAMETER_NAMES[index], float(value))

    @classmethod
    def from_vector(cls, values: np.ndarray) -> ArmControlPolicy:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        kwargs = {name: float(value) for name, value in zip(cls.PARAMETER_NAMES, flat)}
        return cls(**kwargs)


@dataclass(slots=True)
class OctoArmPolicy:
    T_L: float = 2.4
    num_arms: int = 8
    arm_policies: tuple[ArmControlPolicy, ...] = ()

    target_stiffness: np.ndarray = field(init=False)
    target_extension: np.ndarray = field(init=False)
    base_suction_active: np.ndarray = field(init=False)
    middle_suction_active: np.ndarray = field(init=False)
    target_bend: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.target_stiffness = np.ones(self.num_arms, dtype=np.float64)
        self.target_extension = np.zeros(self.num_arms, dtype=np.float64)
        self.base_suction_active = np.zeros(self.num_arms, dtype=np.float64)
        self.middle_suction_active = np.zeros(self.num_arms, dtype=np.float64)
        self.target_bend = np.zeros(self.num_arms, dtype=np.float64)

        if not self.arm_policies:
            object.__setattr__(
                self,
                "arm_policies",
                tuple(ArmControlPolicy() for _ in range(self.num_arms)),
            )

    def actuation_vector(self) -> np.ndarray:
        """Applied per-arm targets after the latest phase update."""
        return np.stack(
            [
                self.target_stiffness.copy(),
                self.target_extension.copy(),
                self.base_suction_active.copy(),
                self.middle_suction_active.copy(),
                self.target_bend.copy(),
            ],
            axis=1,
            dtype=np.float64,
        )

    def apply_phase(self, phase: float) -> None:
        for arm_index, arm_policy in enumerate(self.arm_policies):
            self.target_stiffness[arm_index] = current_activation(
                phase,
                arm_policy.stiffness_center,
                arm_policy.stiffness_deviation,
                arm_policy.stiffness_scale,
            )
            self.target_extension[arm_index] = current_activation(
                phase,
                arm_policy.extension_center,
                arm_policy.extension_deviation,
                arm_policy.extension_scale,
            ) - current_activation(
                phase,
                arm_policy.contraction_center,
                arm_policy.contraction_deviation,
                arm_policy.contraction_scale,
            )
            self.base_suction_active[arm_index] = current_activation(
                phase,
                arm_policy.base_suction_center,
                arm_policy.base_suction_deviation,
                arm_policy.base_suction_scale,
            )
            self.middle_suction_active[arm_index] = current_activation(
                phase,
                arm_policy.middle_suction_center,
                arm_policy.middle_suction_deviation,
                arm_policy.middle_suction_scale,
            )
            self.target_bend[arm_index] = current_activation(
                phase,
                arm_policy.bend_center,
                arm_policy.bend_deviation,
                arm_policy.bend_scale,
            )

    @classmethod
    def vector_size(cls) -> int:
        return cls.num_arms() * ArmControlPolicy.vector_size()

    def lower_bounds(self) -> np.ndarray:
        return np.tile(ArmControlPolicy().lower_bounds(), self.num_arms)

    def upper_bounds(self) -> np.ndarray:
        return np.tile(ArmControlPolicy().upper_bounds(), self.num_arms)

    @classmethod
    def default(cls, T_L: float = 2.4, *, n_arms: int | None = None) -> OctoArmPolicy:
        arm_count = int(n_arms if n_arms is not None else 8)
        return cls(T_L=float(T_L), num_arms=arm_count)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([policy.to_vector() for policy in self.arm_policies]).astype(
            np.float64
        )

    @classmethod
    def from_vector(
        cls,
        values: np.ndarray,
        T_L: float = 2.4,
        *,
        n_arms: int | None = None,
        normalized: bool = False,
    ) -> OctoArmPolicy:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        arm_count = int(n_arms if n_arms is not None else 8)
        arm_width = ArmControlPolicy.vector_size()
        expected = arm_count * arm_width
        if flat.size != expected:
            raise ValueError(f"Expected {expected} policy parameters, got {flat.size}")
        if normalized:
            bounds = cls(T_L=float(T_L), num_arms=arm_count)
            low = bounds.lower_bounds()
            high = bounds.upper_bounds()
            clipped = np.clip(flat, -1.0, 1.0)
            flat = low + (clipped + 1.0) * 0.5 * (high - low)
            flat = np.clip(flat, low, high)
        arms = [
            ArmControlPolicy.from_vector(flat[arm_index * arm_width : (arm_index + 1) * arm_width])
            for arm_index in range(arm_count)
        ]
        return cls(T_L=float(T_L), num_arms=arm_count, arm_policies=tuple(arms))

    def update_from(self, source: "OctoArmPolicy") -> None:
        """Copy open-loop arm parameters in place (keeps pre-bound actuator arrays)."""
        if self.num_arms != source.num_arms:
            raise ValueError(
                f"num_arms mismatch: destination={self.num_arms}, source={source.num_arms}"
            )
        for arm_index, src_arm in enumerate(source.arm_policies):
            dst_arm = self.arm_policies[arm_index]
            for name in ArmControlPolicy.PARAMETER_NAMES:
                setattr(dst_arm, name, getattr(src_arm, name))

    def idle(self) -> None:
        for arm in self.arm_policies:
            arm.stiffness_scale = 0.0
            arm.extension_scale = 0.0
            arm.contraction_scale = 0.0
            arm.base_suction_scale = 0.0
            arm.middle_suction_scale = 0.0
            arm.bend_scale = 0.0


def rotate_policy_half_step(policy):
    """Interpolate each arm policy halfway toward the previous arm (+22.5 degrees)."""
    arm_vectors = [arm_policy.to_vector() for arm_policy in policy.arm_policies]
    arm_count = len(arm_vectors)
    rotated_vectors = [
        0.5 * (arm_vectors[arm_index] + arm_vectors[(arm_index - 1) % arm_count])
        for arm_index in range(arm_count)
    ]
    return type(policy).from_vector(
        np.concatenate(rotated_vectors, axis=0),
        T_L=policy.T_L,
        n_arms=arm_count,
    )


def rotate_policy_negative_half_step(policy):
    """Interpolate each arm policy halfway toward the next arm (-22.5 degrees)."""
    arm_vectors = [arm_policy.to_vector() for arm_policy in policy.arm_policies]
    arm_count = len(arm_vectors)
    rotated_vectors = [
        0.5 * (arm_vectors[arm_index] + arm_vectors[(arm_index + 1) % arm_count])
        for arm_index in range(arm_count)
    ]
    return type(policy).from_vector(
        np.concatenate(rotated_vectors, axis=0),
        T_L=policy.T_L,
        n_arms=arm_count,
    )


def rotate_policy_by_angle(policy, heading_angle: float):
    arm_vectors = [arm_policy.to_vector() for arm_policy in policy.arm_policies]
    arm_count = len(arm_vectors)
    arm_step = 2.0 * np.pi / arm_count
    shift = float(heading_angle / arm_step)

    rotated_vectors: list[np.ndarray] = []
    for arm_index in range(arm_count):
        source_index = arm_index - shift
        lower_index = int(np.floor(source_index)) % arm_count
        upper_index = (lower_index + 1) % arm_count
        upper_weight = source_index - np.floor(source_index)
        lower_weight = 1.0 - upper_weight
        rotated_vectors.append(
            lower_weight * arm_vectors[lower_index] + upper_weight * arm_vectors[upper_index]
        )

    return type(policy).from_vector(
        np.concatenate(rotated_vectors, axis=0),
        T_L=policy.T_L,
        n_arms=arm_count,
    )


@njit(cache=True)
def current_activation(
    phase: float,
    center: float,
    deviation: float,
    scale: float,
) -> float:
    return (
        scale * np.exp(-((phase - center) ** 2) / (2 * deviation**2)) * (1 - (2 * phase - 1) ** 8)
    )

