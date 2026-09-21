"""Replay the latest PPO checkpoint against a moving figure-eight target."""

from __future__ import annotations

import re
from pathlib import Path

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import gym_softrobot  # noqa: F401
from gym_softrobot.envs.tendon_arm.rendering import add_tapered_rod


CHECKPOINT_DIR = Path(__file__).resolve().parent / "save" / "tendon_arm_tracking"
CHECKPOINT_PATTERN = re.compile(r"ppo_tendon_arm_tracking_(\d+)_steps\.zip$")


def latest_checkpoint(directory: Path) -> tuple[Path, Path, int]:
    checkpoints = []
    for path in directory.glob("ppo_tendon_arm_tracking_*_steps.zip"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise click.ClickException(
            f"No v1 tracking checkpoints found in {directory}. Train v1 with "
            "`python examples/tendon_arm_reach/train_ppo.py`. For the v0 "
            "zero-shot baseline, use track_figure_eight_v0.py."
        )

    step, checkpoint = max(checkpoints)
    normalization = directory / f"ppo_tendon_arm_tracking_vecnormalize_{step}_steps.pkl"
    if not normalization.exists():
        raise click.ClickException(
            f"Normalization statistics for {checkpoint.name} are missing: "
            f"{normalization}"
        )
    return checkpoint, normalization, step


@click.command()
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=CHECKPOINT_DIR,
    show_default=True,
    help="Training output directory; the highest-step checkpoint is selected.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path(__file__).resolve().parent / "figure_eight_tracking_v1.mp4",
    show_default=True,
    help="Output video path (.mp4).",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--period",
    type=click.FloatRange(min=0.1),
    default=8.0,
    show_default=True,
    help="Seconds per figure-eight cycle.",
)
@click.option(
    "--duration",
    type=click.FloatRange(min=0.1),
    default=16.0,
    show_default=True,
    help="Total simulated tracking time in seconds.",
)
@click.option(
    "--x-amplitude",
    type=click.FloatRange(min=0.001),
    default=0.06,
    show_default=True,
    help="Figure-eight half-width in meters.",
)
@click.option(
    "--z-amplitude",
    type=click.FloatRange(min=0.001),
    default=0.04,
    show_default=True,
    help="Figure-eight half-height in meters.",
)
def main(
    checkpoint_dir: Path,
    output: Path,
    seed: int,
    period: float,
    duration: float,
    x_amplitude: float,
    z_amplitude: float,
) -> None:
    """Run deterministic PPO inference and render the tracking episode."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as error:
        raise click.ClickException(
            "Install the optional trainer with: uv pip install stable-baselines3"
        ) from error

    checkpoint, normalization, checkpoint_step = latest_checkpoint(checkpoint_dir)
    base_env = gym.make(
        "TendonArmReach-v1",
        figure_eight_period=period,
        figure_eight_x_amplitude=x_amplitude,
        figure_eight_z_amplitude=z_amplitude,
        figure_eight_center_y=-0.26,
    )
    arm = base_env.unwrapped
    control_dt = arm.time_step * arm.simulation_steps_per_action
    steps = max(1, int(round(duration / control_dt)))
    # Extend this evaluation episode without changing the simulator's control rate.
    arm.max_episode_steps = steps + 1
    vector_env = DummyVecEnv([lambda: base_env])
    vector_env.seed(seed)
    env = VecNormalize.load(normalization, vector_env)
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load(checkpoint, env=env)
        observation = env.reset()
        positions = [arm.rod.position_collection.copy()]
        targets = [arm._target.copy()]
        tensions = [np.zeros(12)]
        radii = arm.rod.radius.copy()
        distances = []

        for _ in range(steps):
            action, _state = model.predict(observation, deterministic=True)
            observation, _reward, done, infos = env.step(action)
            positions.append(arm.rod.position_collection.copy())
            targets.append(infos[0]["target_position"].copy())
            tensions.append(infos[0]["tendon_tensions"].copy())
            distances.append(infos[0]["distance_to_target"])
            if done[0]:
                break

        positions = np.asarray(positions)
        targets = np.asarray(targets)
        tensions = np.asarray(tensions)
        distances = np.asarray(distances)
        save_tracking_video(
            positions,
            targets,
            distances,
            tensions,
            radii,
            output,
            control_dt=control_dt,
            checkpoint_step=checkpoint_step,
            policy_label="v1 tracking",
        )
    finally:
        env.close()

    print("policy: v1 tracking")
    print(f"checkpoint: {checkpoint}")
    print(f"tracking steps: {len(distances)} ({len(distances) * control_dt:.2f} s)")
    print(f"mean tip-target error: {distances.mean():.4f} m")
    print(f"RMS tip-target error: {np.sqrt(np.mean(distances**2)):.4f} m")
    print(f"final tip-target error: {distances[-1]:.4f} m")
    print(f"video: {output}")


def save_tracking_video(
    positions: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
    tensions: np.ndarray,
    radii: np.ndarray,
    output: Path,
    *,
    control_dt: float,
    checkpoint_step: int,
    policy_label: str = "PPO",
) -> None:
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    output.parent.mkdir(parents=True, exist_ok=True)
    if not FFMpegWriter.isAvailable():
        raise click.ClickException("ffmpeg is required to write the MP4 video")

    figure = plt.figure(figsize=(17, 8), constrained_layout=True)
    figure.suptitle(
        f"Figure-eight tracking · {policy_label} · {checkpoint_step:,} steps"
    )
    grid = figure.add_gridspec(2, 3, width_ratios=(1.2, 1.0, 1.15))
    axis = figure.add_subplot(grid[:, 0], projection="3d")
    axis.view_init(elev=18, azim=-55, vertical_axis="y")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title("3D side view")

    extent = (
        max(
            np.max(np.abs(positions[:, 0, :])),
            np.max(np.abs(positions[:, 2, :])),
            np.max(np.abs(targets[:, (0, 2)])),
        )
        + np.max(radii)
        + 0.015
    )
    y_min = min(np.min(positions[:, 1, :]), np.min(targets[:, 1])) - 0.02
    y_max = max(np.max(positions[:, 1, :]), np.max(targets[:, 1])) + 0.02
    axis.set_xlim(-extent, extent)
    axis.set_ylim(y_min, y_max)
    axis.set_zlim(-extent, extent)
    axis.set_box_aspect((2 * extent, y_max - y_min, 2 * extent))
    axis.plot(
        targets[:, 0],
        targets[:, 1],
        targets[:, 2],
        linestyle="--",
        color="tab:red",
        alpha=0.45,
        label="target path",
    )
    (tip_trace,) = axis.plot(
        [], [], [], color="tab:green", linewidth=2, label="tip path"
    )
    target_marker = axis.scatter([], [], [], marker="*", s=110, color="tab:red")
    axis.legend(loc="upper left")
    rod_surface = None

    distance_axis = figure.add_subplot(grid[0, 1])
    times = (np.arange(len(distances)) + 1) * control_dt
    distance_axis.plot(times, distances, color="tab:blue", linewidth=1.5)
    distance_axis.axhline(
        0.02, color="black", linestyle="--", linewidth=1, label="2 cm reference"
    )
    distance_axis.set(xlabel="Time (s)", ylabel="Tip-target error (m)")
    distance_axis.grid(alpha=0.25)
    distance_axis.legend()
    distance_cursor = distance_axis.axvline(0.0, color="tab:orange", linewidth=1)

    top_axis = figure.add_subplot(grid[1, 1])
    top_axis.plot(
        targets[:, 0],
        targets[:, 2],
        linestyle="--",
        color="tab:red",
        alpha=0.55,
        label="target path",
    )
    (tip_top_trace,) = top_axis.plot(
        [], [], color="tab:green", linewidth=2, label="tip path"
    )
    (arm_top_line,) = top_axis.plot(
        [], [], color="tab:blue", linewidth=3, label="arm projection"
    )
    (tip_top_marker,) = top_axis.plot([], [], "o", color="tab:green", markersize=5)
    (target_top_marker,) = top_axis.plot([], [], "*", color="tab:red", markersize=11)
    top_axis.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel="x (m)",
        ylabel="z (m)",
        title="Top view (looking down the arm)",
    )
    top_axis.set_aspect("equal", adjustable="box")
    top_axis.grid(alpha=0.25)
    top_axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8)

    actuation_times = np.arange(len(tensions)) * control_dt
    tendon_colors = plt.get_cmap("tab10").colors[:6]
    full_axis = figure.add_subplot(grid[0, 2])
    half_axis = figure.add_subplot(grid[1, 2], sharex=full_axis)
    for tendon_index, color in enumerate(tendon_colors):
        full_axis.plot(
            actuation_times,
            tensions[:, tendon_index],
            color=color,
            linewidth=1.3,
            label=f"T{tendon_index + 1}",
        )
        half_axis.plot(
            actuation_times,
            tensions[:, tendon_index + 6],
            color=color,
            linewidth=1.3,
            label=f"T{tendon_index + 7}",
        )
    full_axis.set_title("Long routes (T1–T6)")
    half_axis.set_title("Half-length routes (T7–T12)")
    full_axis.set_ylabel("Tension (N)")
    half_axis.set_ylabel("Tension (N)")
    half_axis.set_xlabel("Time (s)")
    for actuation_axis in (full_axis, half_axis):
        actuation_axis.set_xlim(0.0, actuation_times[-1])
        actuation_axis.grid(alpha=0.25)
        actuation_axis.legend(ncol=3, fontsize=7, loc="upper right")
    full_cursor = full_axis.axvline(0.0, color="red", linewidth=1.5)
    half_cursor = half_axis.axvline(0.0, color="red", linewidth=1.5)

    def update(frame: int):
        nonlocal rod_surface
        if rod_surface is not None:
            rod_surface.remove()
        rod_surface = add_tapered_rod(axis, positions[frame], radii)
        tip = positions[frame, :, -1]
        target = targets[frame]
        tip_trace.set_data(positions[: frame + 1, 0, -1], positions[: frame + 1, 1, -1])
        tip_trace.set_3d_properties(positions[: frame + 1, 2, -1])
        target_marker._offsets3d = ([target[0]], [target[1]], [target[2]])
        tip_top_trace.set_data(
            positions[: frame + 1, 0, -1], positions[: frame + 1, 2, -1]
        )
        arm_top_line.set_data(positions[frame, 0, :], positions[frame, 2, :])
        tip_top_marker.set_data((tip[0],), (tip[2],))
        target_top_marker.set_data((target[0],), (target[2],))
        elapsed = frame * control_dt
        distance_cursor.set_xdata((elapsed, elapsed))
        full_cursor.set_xdata((elapsed, elapsed))
        half_cursor.set_xdata((elapsed, elapsed))
        error = np.linalg.norm(target - tip)
        top_axis.set_title(f"Top view at {elapsed:.2f} s · tip error {error:.3f} m")
        return (
            rod_surface,
            tip_trace,
            target_marker,
            distance_cursor,
            tip_top_trace,
            arm_top_line,
            tip_top_marker,
            target_top_marker,
            full_cursor,
            half_cursor,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=len(positions),
        interval=1000 * control_dt,
        blit=False,
    )
    animation.save(output, writer=FFMpegWriter(fps=30, bitrate=2200), dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
