"""Plot episode-level Stable-Baselines3 Monitor training progress."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log",
        type=Path,
        help="Training output directory or a monitor.csv file",
    )
    parser.add_argument("--output", type=Path, default=Path("training_progress.png"))
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    plot_progress(args.log, args.output, window=args.window)


def plot_progress(log: Path, output: Path, *, window: int = 20) -> None:
    """Plot episode statistics from an SB3 monitor file or output directory."""

    files = sorted(log.rglob("*monitor.csv")) if log.is_dir() else [log]
    if not files:
        raise SystemExit(f"No monitor CSV files found under {log}")

    rows = []
    for path in files:
        with path.open(newline="") as stream:
            reader = csv.DictReader(line for line in stream if not line.startswith("#"))
            rows.extend(reader)
    if not rows:
        raise SystemExit("Monitor logs contain no completed episodes yet")

    rewards = np.asarray([float(row["r"]) for row in rows])
    lengths = np.asarray([float(row["l"]) for row in rows])
    timesteps = np.cumsum(lengths)
    distances = np.asarray(
        [float(row.get("distance_to_target", "nan")) for row in rows]
    )
    window = max(1, min(window, len(rows)))

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    plot_metric(axes[0], timesteps, rewards, window, "Episode return")
    plot_metric(axes[1], timesteps, lengths, window, "Episode length")
    if np.any(np.isfinite(distances)):
        plot_metric(axes[2], timesteps, distances, window, "Final target distance (m)")
        axes[2].axhline(0.02, color="black", linestyle="--", label="settled")
        axes[2].legend()
    else:
        axes[2].text(
            0.5,
            0.5,
            "Final-distance logging is unavailable in this run",
            ha="center",
            transform=axes[2].transAxes,
        )
    axes[2].set_xlabel("Environment timesteps")
    figure.suptitle(f"PPO training progress ({len(rows)} episodes)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"progress plot: {output}")


def plot_metric(
    axis,
    timesteps: np.ndarray,
    values: np.ndarray,
    window: int,
    label: str,
) -> None:
    axis.plot(timesteps, values, marker=".", alpha=0.35, color="tab:blue")
    if window == 1:
        smooth_values = values
        smooth_steps = timesteps
    else:
        kernel = np.ones(window) / window
        smooth_values = np.convolve(values, kernel, mode="valid")
        smooth_steps = timesteps[window - 1 :]
    axis.plot(smooth_steps, smooth_values, color="tab:blue", linewidth=2)
    axis.set_ylabel(label)
    axis.grid(alpha=0.25)


if __name__ == "__main__":
    main()
