import numpy as np
from gymnasium.utils.env_checker import check_env

from gym_softrobot.envs.tendon_arm import TendonArmReachEnv
from gym_softrobot.envs.tendon_arm.reward import tendon_arm_reward
from gym_softrobot.envs.tendon_arm.spirob_geometry import spirob_taper_profile
from gym_softrobot.envs.tendon_arm.tendon_forces import (
    TendonActuation,
    _tendon_directions,
)


def make_fast_env(**kwargs):
    env = TendonArmReachEnv(**kwargs)
    env.simulation_steps_per_action = 2
    env.max_episode_steps = 2
    return env


def test_tendon_arm_passes_gymnasium_checker():
    env = make_fast_env()
    check_env(env, skip_render_check=True)
    env.close()


def test_tendon_arm_is_seed_deterministic():
    env = make_fast_env()
    first_observation, first_info = env.reset(seed=4)
    second_observation, second_info = env.reset(seed=4)
    np.testing.assert_array_equal(first_observation, second_observation)
    np.testing.assert_array_equal(
        first_info["target_position"], second_info["target_position"]
    )
    env.close()


def test_sampled_targets_are_in_the_reachable_cylinder():
    env = TendonArmReachEnv()
    env.reset(seed=7)
    for _ in range(100):
        target = env._sample_target()
        assert np.hypot(target[0], target[2]) <= 0.1
        assert -0.26 <= target[1] <= -0.20
        assert np.linalg.norm(target) <= 0.95 * env.base_length
    env.close()


def test_time_limit_truncates_without_terminating():
    env = make_fast_env(target=(0.0, -0.20, 0.0))
    env.max_episode_steps = 1
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "time_limit"
    env.close()


def test_stack_frame_controls_history_and_observation_shape():
    env = make_fast_env(stack_frame=3)
    observation, _ = env.reset(seed=3)
    assert observation.shape == (95,)
    assert env.observation_space.shape == (95,)
    env.close()


def test_tendon_arm_reward_is_negative_distance():
    reward, components = tendon_arm_reward(distance=0.12)
    assert reward == -0.12
    assert components == {"distance": -0.12, "velocity": 0.0, "failure": 0.0}


def test_tendon_arm_reward_penalizes_failure():
    reward, components = tendon_arm_reward(distance=0.12, failure=True)
    assert reward == -50.12
    assert components == {
        "distance": -0.12,
        "velocity": 0.0,
        "failure": -50.0,
    }


def test_tendon_arm_reward_penalizes_tip_speed_near_target():
    reward, components = tendon_arm_reward(distance=0.015, tip_speed=0.1)
    assert np.isclose(components["velocity"], -0.0005)
    assert np.isclose(reward, -0.0155)


def test_tendon_arm_reward_does_not_penalize_speed_far_from_target():
    reward, components = tendon_arm_reward(distance=0.04, tip_speed=1.0)
    assert components["velocity"] == 0.0
    assert reward == -0.04


def test_normalized_action_scales_to_configured_tendon_tensions():
    max_tension = 32.0
    env = make_fast_env(max_tension=max_tension)
    env.reset(seed=2)
    action = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(
        info["tendon_tensions"], action * max_tension
    )
    _, _, _, _, info = env.step(np.zeros(12, dtype=np.float32))
    np.testing.assert_array_equal(info["tendon_tensions"], np.zeros(12))
    _, _, _, _, info = env.step(np.ones(12, dtype=np.float32))
    np.testing.assert_array_equal(info["tendon_tensions"], np.full(12, max_tension))
    env.close()


def test_spirob_profile_matches_target_length_and_tip_radius():
    lengths, radii = spirob_taper_profile(25)
    assert np.isclose(lengths.sum(), 0.30)
    assert np.isclose(radii[0], 0.023)
    assert radii[-1] >= 0.01


def test_default_arm_uses_25_element_spirob_geometry():
    env = TendonArmReachEnv()
    assert env.rod.n_elems == 25
    assert np.isclose(env.rod.rest_lengths.sum(), 0.30)
    assert np.isclose(env.rod.radius[0], 0.023)
    assert env.rod.radius[-1] >= 0.01
    assert env.action_space.shape == (12,)
    np.testing.assert_array_equal(env.action_space.low, np.zeros(12))
    np.testing.assert_array_equal(env.action_space.high, np.ones(12))
    assert env.max_tension == 55.2
    assert env.damping_constant == 1.0
    assert env.simulation_steps_per_action == 1111
    assert np.isclose(
        env.time_step * env.simulation_steps_per_action, 1.0 / 30.0, atol=2e-5
    )
    assert env.max_episode_steps == 180
    assert env.observation_space.shape == (125,)
    env.close()


def test_damping_constant_is_configurable_and_validated():
    env = make_fast_env(damping_constant=0.35)
    assert env.damping_constant == 0.35
    env.close()

    try:
        make_fast_env(damping_constant=-0.1)
    except ValueError as error:
        assert "damping_constant" in str(error)
    else:
        raise AssertionError("negative damping should be rejected")


def test_tendon_actuation_uses_six_hexagonal_offsets_at_75_percent_radius():
    radii = np.linspace(0.023, 0.0108, 25)
    route_nodes = np.array((3, 8, 13, 18))
    actuation = TendonActuation(
        contact_nodes=route_nodes,
        tensions=np.zeros(6),
        rod_radii=radii,
    )
    assert actuation.routing_offsets.shape == (6, 5, 3)
    np.testing.assert_allclose(
        np.linalg.norm(actuation.routing_offsets, axis=2),
        np.broadcast_to(
            0.75
            * np.array(
                (
                    radii[0],
                    *(0.5 * (radii[node - 1] + radii[node]) for node in route_nodes),
                )
            ),
            (6, 5),
        ),
    )


def test_tendon_route_ends_at_the_last_contact():
    positions = np.zeros((3, 6))
    positions[1] = np.linspace(0.0, -0.3, 6)
    directors = np.repeat(np.eye(3)[:, :, None], 5, axis=2)
    contact_nodes = np.array((2, 4))
    offsets = np.zeros((1, 3, 3))

    directions = _tendon_directions(positions, directors, contact_nodes, offsets)

    assert directions.shape == (1, 3, 3)
    np.testing.assert_allclose(directions[0, :2], np.tile((0.0, -1.0, 0.0), (2, 1)))
    np.testing.assert_array_equal(directions[0, 2], np.zeros(3))
