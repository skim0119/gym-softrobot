#!/usr/bin/env python3
"""Replay a fixed HBBO policy with the registered Gymnasium environment.

The CSV stores a fixed ``(8, 9, T)`` action sequence, which is played through
the public ``reset``, ``step``, and ``render`` API of ``OctoMuscleCrawl-v0``.
"""

from __future__ import annotations

from pathlib import Path

import click
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

import gym_softrobot  # noqa: F401 - registers OctoMuscleCrawl-v0

DEFAULT_POLICY = Path(__file__).with_name("policy.csv")


def _load_csv_policy(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"policy CSV not found: {path}")
    flat_actions = np.atleast_2d(
        np.loadtxt(path, delimiter=",", skiprows=1).astype(np.float64)
    )
    if flat_actions.shape[1] != 8 * 9:
        raise ValueError(
            f"Expected policy CSV with shape (T, 72), got {flat_actions.shape}"
        )
    return flat_actions.reshape(-1, 8, 9).transpose(1, 2, 0)


def _save_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise RuntimeError("Gymnasium render() produced no frames")
    figure, axis = plt.subplots(figsize=(8, 5), dpi=100)
    axis.axis("off")
    image = axis.imshow(frames[0])
    writer = FFMpegWriter(fps=fps)
    with writer.saving(figure, str(path), dpi=100):
        for frame in frames:
            image.set_data(frame)
            writer.grab_frame()
    plt.close(figure)


@click.command()
@click.option(
    "--policy",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_POLICY,
    show_default=True,
    help="CSV containing flattened (T, 8, 9) normalized actions.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(__file__).with_name("output"),
    show_default=True,
    help="Directory for the generated video.",
)
@click.option("--fps", type=click.FloatRange(min=0.1), default=30.0, show_default=True)
def main(policy: Path, output: Path, fps: float) -> None:
    """Replay a fixed policy through OctoMuscleCrawl-v0."""
    actions = _load_csv_policy(policy)
    env = gym.make(
        "OctoMuscleCrawl-v0",
        horizon=actions.shape[-1],
        render_mode="rgb_array",
    )

    frames: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = truncated = False
    try:
        _observation, _info = env.reset(seed=0)
        frames.append(env.render())
        for step in range(actions.shape[-1]):
            _observation, reward, terminated, truncated, _info = env.step(
                actions[:, :, step]
            )
            rewards.append(float(reward))
            frames.append(env.render())
            if terminated or truncated:
                break
    finally:
        env.close()

    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "motion.mp4"
    _save_video([frame for frame in frames if frame is not None], video_path, fps)
    print(f"Gymnasium reward sum: {sum(rewards):.8f}")
    print(f"Action sequence shape: {actions.shape}")
    print(f"Terminated={terminated}, truncated={truncated}")
    print(f"motion={video_path}")


if __name__ == "__main__":
    main()
