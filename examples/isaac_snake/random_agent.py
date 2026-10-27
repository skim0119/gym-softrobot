"""Random agent for the continuum snake inside Isaac Lab 2.3.2.

Launch this file with Isaac Lab's Python so ``AppLauncher`` starts Omniverse
before the task is registered. From an Isaac Lab checkout:

    ./isaaclab.sh -p /path/to/gym-softrobot/examples/isaac_snake/random_agent.py --num_envs 1 --headless --steps 1

``--num_envs`` copies the same PyElastica snake. Each copy steps on the CPU.
PhysX only advances an empty stage.
"""

import argparse


def main() -> None:
    try:
        from isaaclab.app import AppLauncher
    except ImportError as error:
        raise SystemExit(
            "Isaac Lab is not installed in this interpreter. From an Isaac Lab "
            "checkout, pass this file to isaaclab.sh. Example:\n"
            "./isaaclab.sh -p /path/to/gym-softrobot/examples/isaac_snake/"
            "random_agent.py --num_envs 1 --headless --steps 1"
        ) from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default="SoftRobot-ContinuumSnake-Direct-v0",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import parse_env_cfg

    from gym_softrobot.isaac.env import SoftRobotEnvCfg

    @configclass
    class SnakeEnvCfg(SoftRobotEnvCfg):
        task_entry: str = (
            "gym_softrobot.envs.snake.locomotion_task:SnakeLocomotionTask"
        )

    gym.register(
        id=args.task,
        entry_point="gym_softrobot.isaac.env:SoftRobotEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": SnakeEnvCfg},
    )
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    observation, _info = env.reset()
    print(f"reset observation {tuple(observation['policy'].shape)}")
    try:
        for index in range(args.steps):
            actions = (
                2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            observation, reward, terminated, truncated, _info = env.step(actions)
            print(
                f"step {index + 1} reward {reward} "
                f"terminated {terminated} truncated {truncated}"
            )
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
