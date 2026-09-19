import gymnasium as gym
import numpy as np
import pytest

import gym_softrobot  # noqa: F401


BENCHMARK_IDS = (
    "ElasticaArmTracking-v0",
    "ElasticaArmReach-v0",
    "ElasticaArmObstacle-v0",
    "ElasticaArmObstacleRandom-v0",
)


@pytest.mark.parametrize("env_id", BENCHMARK_IDS)
def test_elastica_benchmark_reset_step(env_id):
    env = gym.make(env_id, final_time=0.02)
    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    result = env.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
    observation, reward, terminated, truncated, info = result

    assert env.observation_space.contains(observation)
    assert np.isscalar(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "tip_target_distance" in info
    env.close()


def test_random_obstacle_layout_is_seeded():
    layouts = []
    for _ in range(2):
        env = gym.make("ElasticaArmObstacleRandom-v0", final_time=0.02)
        env.reset(seed=42)
        layouts.append(env.unwrapped.obstacle_state.copy())
        env.close()

    np.testing.assert_array_equal(*layouts)
