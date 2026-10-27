"""Train the continuum snake with Isaac Lab 2.3.2 and RSL-RL.

Launch this file with Isaac Lab's Python so ``AppLauncher`` starts Omniverse
before the task is registered. From an Isaac Lab checkout:

    ./isaaclab.sh -p /path/to/gym-softrobot/examples/isaac_snake/train.py --num_envs 1 --headless --max_iterations 5

Each environment is one CPU rod. The first iteration collects 16 steps before
RSL-RL prints a mean reward.
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
            "train.py --num_envs 1 --headless --max_iterations 5"
        ) from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default="SoftRobot-ContinuumSnake-Direct-v0",
    )
    parser.add_argument(
        "--rod_processes",
        action="store_true",
        help="step each env's rod in a worker process",
    )
    parser.add_argument(
        "--rod_workers",
        type=int,
        default=None,
        help="worker processes for --rod_processes; default is the CPU count",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1:
        raise SystemExit("--num_envs must be >= 1")
    if args.rod_workers is not None and args.rod_workers < 1:
        raise SystemExit("--rod_workers must be >= 1")
    if args.max_iterations < 1:
        raise SystemExit("--max_iterations must be >= 1")
    app = AppLauncher(args).app

    import os
    from datetime import datetime

    import gymnasium as gym
    from isaaclab.utils import configclass
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )
    from isaaclab_tasks.utils import parse_env_cfg
    from rsl_rl.runners import OnPolicyRunner

    from gym_softrobot.isaac.env import SoftRobotEnvCfg

    @configclass
    class SnakeEnvCfg(SoftRobotEnvCfg):
        task_entry: str = (
            "gym_softrobot.envs.snake.locomotion_task:SnakeLocomotionTask"
        )

    @configclass
    class SnakePPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 16
        max_iterations = 5
        save_interval = 5
        experiment_name = "snake_locomotion"
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    gym.register(
        id=args.task,
        entry_point="gym_softrobot.isaac.env:SoftRobotEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": SnakeEnvCfg},
    )
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.rod_processes = args.rod_processes
    env_cfg.rod_workers = args.rod_workers
    agent_cfg = SnakePPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(
        "logs",
        "rsl_rl",
        agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    print(f"[INFO] Logging experiment in directory: {os.path.abspath(log_dir)}")
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=False,
        )
    finally:
        env.close()
        app.close()


if __name__ == "__main__":
    main()
