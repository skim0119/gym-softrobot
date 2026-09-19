"""Train PPO on one of the Elastica-RL-control arm benchmarks."""

import argparse
from pathlib import Path

import gymnasium as gym

import gym_softrobot  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id",
        choices=(
            "ElasticaArmTracking-v0",
            "ElasticaArmReach-v0",
            "ElasticaArmObstacle-v0",
            "ElasticaArmObstacleRandom-v0",
        ),
        default="ElasticaArmTracking-v0",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("save/elastica_arm"))
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise SystemExit(
            "Install the optional trainer with: uv sync --group benchmark"
        ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(gym.make(args.env_id))
    env.reset(seed=args.seed)
    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.output_dir,
        name_prefix="ppo",
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.output_dir / "tensorboard",
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint,
            progress_bar=True,
        )
        model.save(args.output_dir / "final_model")
    finally:
        env.close()


if __name__ == "__main__":
    main()
