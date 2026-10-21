"""Train PPO on the Spirob tendon-arm Gymnasium environment."""

import importlib.util
from pathlib import Path

import click
import gymnasium as gym

import gym_softrobot  # noqa: F401
from plot_progress import plot_progress


@click.command()
@click.option(
    "--timesteps",
    type=click.IntRange(min=1),
    default=1_000_000,
    show_default=True,
    help="Number of policy steps to train.",
)
@click.option(
    "--n-envs",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of parallel simulation environments.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path(__file__).resolve().parent / "save" / "tendon_arm_tracking",
    show_default=True,
    help="Directory for checkpoints, logs, and normalization statistics.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--max-tension",
    type=click.FloatRange(min=0.01),
    default=55.2,
    show_default=True,
    help="Physical tension (N) represented by normalized action +1.",
)
@click.option(
    "--checkpoint-freq",
    type=click.IntRange(min=1),
    default=10_000,
    show_default=True,
    help="Save a checkpoint every this many environment steps.",
)
def main(
    timesteps: int,
    n_envs: int,
    output: Path,
    seed: int,
    max_tension: float,
    checkpoint_freq: int,
) -> None:
    """Train PPO on the moving-target TendonArmReach-v1 task."""

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
            VecNormalize,
        )
    except ImportError as error:
        raise SystemExit(
            "Install the optional trainer with: uv pip install stable-baselines3"
        ) from error

    output.mkdir(parents=True, exist_ok=True)
    env_kwargs = {"max_tension": max_tension}

    def make_env():
        return gym.make("TendonArmReach-v1", **env_kwargs)

    env_fns = [make_env for _ in range(n_envs)]
    if n_envs == 1:
        vector_env = DummyVecEnv(env_fns)
    else:
        vector_env = SubprocVecEnv(env_fns, start_method="spawn")
    monitored_env = VecMonitor(
        vector_env,
        filename=str(output / "monitor.csv"),
        info_keywords=("distance_to_target",),
    )
    env = VecNormalize(monitored_env, norm_obs=True, norm_reward=True)
    checkpoint = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=output,
        name_prefix="ppo_tendon_arm_tracking",
        save_vecnormalize=True,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        tensorboard_log=(
            output / "tensorboard"
            if importlib.util.find_spec("tensorboard") is not None
            else None
        ),
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        n_steps=2048,
        batch_size=256,
        n_epochs=6,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
    )
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint,
            progress_bar=(
                importlib.util.find_spec("rich") is not None
                and importlib.util.find_spec("tqdm") is not None
            ),
        )
        model.save(output / "final_model")
        env.save(output / "vecnormalize.pkl")
    finally:
        env.close()
    plot_progress(
        output / "monitor.csv",
        output / "training_progress.png",
        n_envs=n_envs,
    )


if __name__ == "__main__":
    main()
