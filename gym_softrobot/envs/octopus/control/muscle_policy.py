"""Open-loop phase-Gaussian policies for independently controlled octopus muscle groups."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gym_softrobot.envs.octopus.control.phase_policy import (
    current_activation,
)


@dataclass
class MuscleArmControlPolicy:
    tm_center: float = 0.33
    tm_deviation: float = 0.20
    tm_scale: float = 0.5
    lm0_center: float = 0.66
    lm0_deviation: float = 0.20
    lm0_scale: float = 0.2
    lm1_center: float = 0.66
    lm1_deviation: float = 0.20
    lm1_scale: float = 0.2
    lm2_center: float = 0.66
    lm2_deviation: float = 0.20
    lm2_scale: float = 0.2
    lm3_center: float = 0.66
    lm3_deviation: float = 0.20
    lm3_scale: float = 0.2
    base_suction_center: float = 0.33
    base_suction_deviation: float = 0.2
    base_suction_scale: float = 0.7
    middle_suction_center: float = 0.66
    middle_suction_deviation: float = 0.20
    middle_suction_scale: float = 0.7
    om_positive_center: float = 0.5
    om_positive_deviation: float = 0.10
    om_positive_scale: float = 0.0
    om_negative_center: float = 0.5
    om_negative_deviation: float = 0.10
    om_negative_scale: float = 0.0

    PARAMETER_NAMES = (
        "tm_center",
        "tm_deviation",
        "tm_scale",
        "lm0_center",
        "lm0_deviation",
        "lm0_scale",
        "lm1_center",
        "lm1_deviation",
        "lm1_scale",
        "lm2_center",
        "lm2_deviation",
        "lm2_scale",
        "lm3_center",
        "lm3_deviation",
        "lm3_scale",
        "base_suction_center",
        "base_suction_deviation",
        "base_suction_scale",
        "middle_suction_center",
        "middle_suction_deviation",
        "middle_suction_scale",
        "om_positive_center",
        "om_positive_deviation",
        "om_positive_scale",
        "om_negative_center",
        "om_negative_deviation",
        "om_negative_scale",
    )

    @classmethod
    def vector_size(cls) -> int:
        return len(cls.PARAMETER_NAMES)

    def lower_bounds(self) -> np.ndarray:
        values: list[float] = []
        for _ in range(9):
            values.extend((0.0, 1.0e-6, 0.0))
        return np.asarray(values, dtype=np.float64)

    def upper_bounds(self) -> np.ndarray:
        values: list[float] = []
        for _ in range(9):
            values.extend((1.0, 0.5, 1.0))
        return np.asarray(values, dtype=np.float64)

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
    def from_vector(cls, values: np.ndarray) -> "MuscleArmControlPolicy":
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        kwargs = {name: float(value) for name, value in zip(cls.PARAMETER_NAMES, flat)}
        return cls(**kwargs)


@dataclass(slots=True)
class OctoArmMusclePolicy:
    T_L: float = 2.4
    n_arms: int = 8
    arm_policies: tuple[MuscleArmControlPolicy, ...] = ()

    base_suction_active: np.ndarray = field(init=False)
    middle_suction_active: np.ndarray = field(init=False)
    target_muscle_groups: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.base_suction_active = np.zeros(self.n_arms, dtype=np.float64)
        self.middle_suction_active = np.zeros(self.n_arms, dtype=np.float64)
        self.target_muscle_groups = np.zeros((self.n_arms, 7), dtype=np.float64)
        if not self.arm_policies:
            object.__setattr__(
                self,
                "arm_policies",
                tuple(MuscleArmControlPolicy() for _ in range(self.n_arms)),
            )

    def actuation_vector(self) -> np.ndarray:
        return np.stack(
            [
                self.target_muscle_groups[:, 0].copy(),
                self.target_muscle_groups[:, 1].copy(),
                self.target_muscle_groups[:, 2].copy(),
                self.target_muscle_groups[:, 3].copy(),
                self.target_muscle_groups[:, 4].copy(),
                self.base_suction_active.copy(),
                self.middle_suction_active.copy(),
                self.target_muscle_groups[:, 5].copy(),
                self.target_muscle_groups[:, 6].copy(),
            ],
            axis=1,
            dtype=np.float64,
        )

    def apply_phase(self, phase: float) -> None:
        for arm_index, arm_policy in enumerate(self.arm_policies):
            self.target_muscle_groups[arm_index, 0] = current_activation(
                phase, arm_policy.tm_center, arm_policy.tm_deviation, arm_policy.tm_scale
            )
            self.target_muscle_groups[arm_index, 1] = max(
                0.0,
                current_activation(
                    phase, arm_policy.lm0_center, arm_policy.lm0_deviation, arm_policy.lm0_scale
                ),
            )
            self.target_muscle_groups[arm_index, 2] = max(
                0.0,
                current_activation(
                    phase, arm_policy.lm1_center, arm_policy.lm1_deviation, arm_policy.lm1_scale
                ),
            )
            self.target_muscle_groups[arm_index, 3] = max(
                0.0,
                current_activation(
                    phase, arm_policy.lm2_center, arm_policy.lm2_deviation, arm_policy.lm2_scale
                ),
            )
            self.target_muscle_groups[arm_index, 4] = max(
                0.0,
                current_activation(
                    phase, arm_policy.lm3_center, arm_policy.lm3_deviation, arm_policy.lm3_scale
                ),
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
            self.target_muscle_groups[arm_index, 5] = max(
                0.0,
                current_activation(
                    phase,
                    arm_policy.om_positive_center,
                    arm_policy.om_positive_deviation,
                    arm_policy.om_positive_scale,
                ),
            )
            self.target_muscle_groups[arm_index, 6] = max(
                0.0,
                current_activation(
                    phase,
                    arm_policy.om_negative_center,
                    arm_policy.om_negative_deviation,
                    arm_policy.om_negative_scale,
                ),
            )

    @classmethod
    def num_arms(cls) -> int:
        return 8

    @classmethod
    def vector_size(cls) -> int:
        return cls.num_arms() * MuscleArmControlPolicy.vector_size()

    def lower_bounds(self) -> np.ndarray:
        return np.tile(MuscleArmControlPolicy().lower_bounds(), self.n_arms)

    def upper_bounds(self) -> np.ndarray:
        return np.tile(MuscleArmControlPolicy().upper_bounds(), self.n_arms)

    @classmethod
    def default(
        cls,
        T_L: float = 2.4,
        *,
        n_arms: int | None = None,
    ) -> "OctoArmMusclePolicy":
        arm_count = int(n_arms if n_arms is not None else cls.num_arms())
        return cls(T_L=float(T_L), n_arms=arm_count)

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
    ) -> "OctoArmMusclePolicy":
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        arm_count = int(n_arms if n_arms is not None else cls.num_arms())
        arm_width = MuscleArmControlPolicy.vector_size()
        expected = arm_count * arm_width
        if flat.size != expected:
            raise ValueError(f"Expected {expected} policy parameters, got {flat.size}")
        if normalized:
            bounds = cls(T_L=float(T_L), n_arms=arm_count)
            low = bounds.lower_bounds()
            high = bounds.upper_bounds()
            clipped = np.clip(flat, -1.0, 1.0)
            flat = low + (clipped + 1.0) * 0.5 * (high - low)
            flat = np.clip(flat, low, high)
        arms = [
            MuscleArmControlPolicy.from_vector(
                flat[arm_index * arm_width : (arm_index + 1) * arm_width]
            )
            for arm_index in range(arm_count)
        ]
        return cls(T_L=float(T_L), n_arms=arm_count, arm_policies=tuple(arms))

    def update_from(self, source: "OctoArmMusclePolicy") -> None:
        """Copy open-loop arm parameters in place (keeps pre-bound actuator arrays)."""
        if self.n_arms != source.n_arms:
            raise ValueError(f"n_arms mismatch: destination={self.n_arms}, source={source.n_arms}")
        for arm_index, src_arm in enumerate(source.arm_policies):
            dst_arm = self.arm_policies[arm_index]
            for name in MuscleArmControlPolicy.PARAMETER_NAMES:
                setattr(dst_arm, name, getattr(src_arm, name))

    def idle(self) -> None:
        """
        Set scales to be zero
        """
        for arm in self.arm_policies:
            arm.tm_scale = 0.0
            arm.lm0_scale = 0.0
            arm.lm1_scale = 0.0
            arm.lm2_scale = 0.0
            arm.lm3_scale = 0.0
            arm.base_suction_scale = 0.0
            arm.middle_suction_scale = 0.0
            arm.om_positive_scale = 0.0
            arm.om_negative_scale = 0.0
