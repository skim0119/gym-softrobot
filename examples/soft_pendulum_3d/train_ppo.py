"""Train a PPO policy for SoftPendulum3D-v0.

Install Stable-Baselines3 separately before running this example.
"""

from pathlib import Path

import gymnasium as gym

import gym_softrobot  # noqa: F401


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as error:
        raise SystemExit(
            "Install Stable-Baselines3 to run this example: "
            "uv pip install stable-baselines3"
        ) from error

    output_dir = Path("save/soft_pendulum_3d")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("SoftPendulum3D-v0")
    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=output_dir,
        name_prefix="ppo",
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=output_dir / "tensorboard",
    )
    model.learn(total_timesteps=1_000_000, callback=checkpoint, progress_bar=True)
    model.save(output_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
