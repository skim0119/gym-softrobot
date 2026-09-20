from __future__ import annotations

import numpy as np
import pytest

from gym_softrobot.envs.octopus.control.muscle_policy import OctoArmMusclePolicy
from gym_softrobot.envs.octopus.control.phase_policy import OctoArmPolicy
from gym_softrobot.envs.octopus.crawling_simulation.config import (
    OctopusMuscleConfig,
    OctopusV1Config,
)
from gym_softrobot.envs.octopus.crawling_simulation.muscle_simulation import (
    PhaseOctopusMuscleSimulation,
)
from gym_softrobot.envs.octopus.crawling_simulation.phase_simulation import (
    PhaseOctopusSimulation,
)


def _small_config(config_type):
    return config_type(
        n_elem=5,
        episode_duration_cycles=0.01,
        control_dt=0.01,
        time_step=0.001,
    )


def test_phase_policy_vector_round_trip() -> None:
    policy = OctoArmPolicy.default()
    vector = policy.to_vector()

    assert vector.shape == (144,)
    restored = OctoArmPolicy.from_vector(vector)
    np.testing.assert_array_equal(restored.to_vector(), vector)


def test_muscle_policy_vector_round_trip() -> None:
    policy = OctoArmMusclePolicy.default()
    vector = policy.to_vector()

    assert vector.shape == (216,)
    restored = OctoArmMusclePolicy.from_vector(vector)
    np.testing.assert_array_equal(restored.to_vector(), vector)


def test_phase_simulation_advances_with_finite_state() -> None:
    simulation = PhaseOctopusSimulation(_small_config(OctopusV1Config))
    initial_position = simulation.sphere_position().copy()

    simulation.advance(simulation.config.control_dt)

    assert simulation.time == pytest.approx(simulation.config.control_dt)
    assert simulation.is_finite()
    assert not np.array_equal(simulation.sphere_position(), initial_position)


def test_muscle_simulation_accepts_direct_muscle_control() -> None:
    simulation = PhaseOctopusMuscleSimulation(_small_config(OctopusMuscleConfig))
    simulation.policies[0].target_muscle_groups.fill(0.5)

    simulation.advance(simulation.config.control_dt, apply_phase=False)

    assert simulation.time == pytest.approx(simulation.config.control_dt)
    assert simulation.is_finite()


def test_sphere_rests_on_arm_friction_plane() -> None:
    simulation = PhaseOctopusMuscleSimulation(_small_config(OctopusMuscleConfig))
    cfg = simulation.config
    expected_lift = cfg.base_sphere_radius - cfg.base_radius
    sphere_y = simulation.sphere_position()[1]
    arm_base_y = simulation.rods[0].position_collection[1, 0]

    assert sphere_y == pytest.approx(expected_lift)
    assert arm_base_y == pytest.approx(0.0)
    assert sphere_y - cfg.base_sphere_radius == pytest.approx(-cfg.base_radius)
