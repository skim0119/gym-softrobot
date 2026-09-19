"""Evaluate and visualize a trained TendonArmReach-v0 PPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import gym_softrobot  # noqa: F401
from gym_softrobot.envs.tendon_arm.rendering import add_tapered_rod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="PPO .zip checkpoint")
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        help="VecNormalize .pkl; defaults to vecnormalize.pkl beside the model",
    )
    parser.add_argument("--output", type=Path, default=Path("policy_rollout"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--max-tension", type=float, default=55.2)
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional .gif or .mp4 animation path",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as error:
        raise SystemExit(
            "Install the optional trainer with: uv pip install stable-baselines3"
        ) from error

    args.output.mkdir(parents=True, exist_ok=True)
    raw_env = gym.make("TendonArmReach-v0", max_tension=args.max_tension)
    raw_env.reset(seed=args.seed)
    vec_env = DummyVecEnv([lambda: raw_env])

    if args.vecnormalize is not None:
        vecnormalize_path = args.vecnormalize
    else:
        checkpoint_stats = args.model.with_name(f"{args.model.stem}_vecnormalize.pkl")
        final_stats = args.model.with_name("vecnormalize.pkl")
        vecnormalize_path = (
            checkpoint_stats if checkpoint_stats.exists() else final_stats
        )
    if not vecnormalize_path.exists():
        raise SystemExit(f"VecNormalize statistics not found: {vecnormalize_path}")
    env = VecNormalize.load(vecnormalize_path, vec_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.model, env=env)

    observation = env.reset()
    arm = raw_env.unwrapped
    positions = [arm.rod.position_collection.copy()]
    radii = arm.rod.radius.copy()
    targets = [arm._target.copy()]
    tensions = []
    distances = []
    rewards = []

    for _ in range(args.steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, infos = env.step(action)
        info = infos[0]
        positions.append(arm.rod.position_collection.copy())
        targets.append(info["target_position"].copy())
        tensions.append(info["tendon_tensions"].copy())
        distances.append(info["distance_to_target"])
        rewards.append(float(reward[0]))
        if done[0]:
            break

    positions_array = np.asarray(positions)
    targets_array = np.asarray(targets)
    tensions_array = np.asarray(tensions)
    distances_array = np.asarray(distances)
    rewards_array = np.asarray(rewards)
    np.savez_compressed(
        args.output / "rollout.npz",
        positions=positions_array,
        radii=radii,
        targets=targets_array,
        tensions=tensions_array,
        distances=distances_array,
        rewards=rewards_array,
    )
    plot_summary(
        positions_array,
        radii,
        targets_array[-1],
        tensions_array,
        distances_array,
        rewards_array,
        args.output / "rollout_summary.png",
    )
    if args.video is not None:
        save_animation(
            positions_array,
            targets_array,
            radii,
            args.video,
            frame_duration=arm.time_step * arm.simulation_steps_per_action,
        )
    env.close()

    print(f"steps: {len(distances_array)}")
    print(f"final distance: {distances_array[-1]:.5f} m")
    print(f"summary: {args.output / 'rollout_summary.png'}")
    print(f"data: {args.output / 'rollout.npz'}")
    if args.video is not None:
        print(f"animation: {args.video}")


def plot_summary(
    positions: np.ndarray,
    radii: np.ndarray,
    target: np.ndarray,
    tensions: np.ndarray,
    distances: np.ndarray,
    rewards: np.ndarray,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_3d.view_init(vertical_axis="y")
    time_axis = np.arange(len(distances))

    stride = max(1, len(positions) // 12)
    for frame in positions[::stride]:
        axis_3d.plot(*frame, color="tab:blue", alpha=0.18)
    add_tapered_rod(axis_3d, positions[-1], radii, color="tab:blue")
    axis_3d.scatter(*target, marker="*", s=140, color="tab:red", label="target")
    axis_3d.set(
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
        title="Arm trajectory",
    )
    bounds = np.vstack((positions[-1].T, target))
    padding = float(np.max(radii)) + 0.01
    lower = bounds.min(axis=0) - padding
    upper = bounds.max(axis=0) + padding
    axis_3d.set(
        xlim=(lower[0], upper[0]),
        ylim=(lower[1], upper[1]),
        zlim=(lower[2], upper[2]),
    )
    axis_3d.set_box_aspect(upper - lower)
    axis_3d.legend()

    distance_axis = figure.add_subplot(grid[0, 1])
    distance_axis.plot(time_axis, distances, color="tab:red", label="distance")
    distance_axis.axhline(0.02, color="black", linestyle="--", label="settled")
    distance_axis.set(
        ylabel="Tip-target distance (m)",
        title=f"Final distance: {distances[-1]:.4f} m",
    )
    distance_axis.grid(alpha=0.25)
    distance_axis.legend()

    action_axis = figure.add_subplot(grid[1, 1])
    for tendon in range(tensions.shape[1]):
        action_axis.plot(tensions[:, tendon], linewidth=1, label=f"T{tendon + 1}")
    action_axis.set(
        xlabel="Policy step",
        ylabel="Tension (N)",
        title=f"Actions; return={rewards.sum():.1f}",
    )
    action_axis.grid(alpha=0.25)
    action_axis.legend(ncol=4, fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_animation(
    positions: np.ndarray,
    targets: np.ndarray,
    radii: np.ndarray,
    output: Path,
    *,
    frame_duration: float,
) -> None:
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(6, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.view_init(vertical_axis="y")
    target_plot = axis.scatter([], [], [], marker="*", s=140, color="tab:red")
    horizontal_extent = max(
        np.max(np.abs(positions[:, 0, :])),
        np.max(np.abs(positions[:, 2, :])),
        np.max(np.abs(targets[:, (0, 2)])),
    ) + np.max(radii) + 0.01
    y_min = (
        min(np.min(positions[:, 1, :]), np.min(targets[:, 1]))
        - np.max(radii)
        - 0.01
    )
    y_max = (
        max(np.max(positions[:, 1, :]), np.max(targets[:, 1]))
        + np.max(radii)
        + 0.01
    )
    axis.set(
        xlim=(-horizontal_extent, horizontal_extent),
        ylim=(y_min, y_max),
        zlim=(-horizontal_extent, horizontal_extent),
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
    )
    axis.set_box_aspect(
        (
            2.0 * horizontal_extent,
            y_max - y_min,
            2.0 * horizontal_extent,
        )
    )
    tube = None

    def update(frame_index: int):
        nonlocal tube
        if tube is not None:
            tube.remove()
        frame = positions[frame_index]
        tube = add_tapered_rod(axis, frame, radii, color="tab:blue")
        target_plot._offsets3d = (
            [targets[frame_index, 0]],
            [targets[frame_index, 1]],
            [targets[frame_index, 2]],
        )
        axis.set_title(f"Simulation time: {frame_index * frame_duration:.2f} s")
        return tube, target_plot

    animation = FuncAnimation(
        figure, update, frames=len(positions), interval=1000 / 20, blit=False
    )
    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=20)
    elif output.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=20, bitrate=1800)
    else:
        raise ValueError("--video must end in .gif or .mp4")
    animation.save(output, writer=writer, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
