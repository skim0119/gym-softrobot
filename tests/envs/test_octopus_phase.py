import numpy as np

from gymnasium.utils.env_checker import check_env

from gym_softrobot.envs.octopus.muscle_crawl_env import OctoMuscleCrawlEnv
from gym_softrobot.envs.octopus.crawling_simulation.config import OctopusMuscleConfig


def _small_config() -> OctopusMuscleConfig:
    return OctopusMuscleConfig(
        n_elem=5,
        episode_duration_cycles=0.01,
        control_dt=0.01,
        time_step=0.001,
    )


def test_phase_crawl_feedback_step() -> None:
    env = OctoMuscleCrawlEnv(config=_small_config(), horizon=2)
    observation, info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    assert info["time"] == 0.0

    observation, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["time"] > 0.0
    env.close()


def test_phase_crawl_gymnasium_contract() -> None:
    env = OctoMuscleCrawlEnv(config=_small_config(), horizon=2)
    check_env(env, skip_render_check=True)
    env.close()
