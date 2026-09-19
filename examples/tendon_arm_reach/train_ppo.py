"""Train PPO on TendonArmReach-v0 using Stable-Baselines3."""

import argparse
import importlib.util
from pathlib import Path

import gymnasium as gym

import gym_softrobot  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("save/tendon_arm_reach"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a coarse, inexpensive simulator for pipeline checks only.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as error:
        raise SystemExit(
            "Install the optional trainer with: uv sync --group benchmark"
        ) from error

    args.output.mkdir(parents=True, exist_ok=True)
    env_kwargs = {}
    if args.fast:
        env_kwargs = {
            "n_elements": 20,
            "simulation_steps_per_action": 5,
            "episode_time": 0.0375,
        }
    def make_env():
        base_env = gym.make("TendonArmReach-v0", **env_kwargs)
        return Monitor(
            base_env,
            filename=str(args.output / "monitor.csv"),
            info_keywords=("distance_to_target",),
        )

    vector_env = DummyVecEnv([make_env])
    env = VecNormalize(vector_env, norm_obs=True, norm_reward=True)
    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=args.output,
        name_prefix="ppo_tendon_arm",
        save_vecnormalize=True,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=(
            args.output / "tensorboard"
            if importlib.util.find_spec("tensorboard") is not None
            else None
        ),
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        n_steps=128 if args.fast else 2048,
        batch_size=64 if args.fast else 256,
        n_epochs=6,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint,
        progress_bar=(
            importlib.util.find_spec("rich") is not None
            and importlib.util.find_spec("tqdm") is not None
        ),
    )
    model.save(args.output / "final_model")
    env.save(args.output / "vecnormalize.pkl")
    env.close()


if __name__ == "__main__":
    main()
