"""Roll out the continuum snake without Isaac Lab.

One step integrates 0.2 s of rod time, about 25,000 integrator steps.
The first rewards are 0 until two gait periods have been recorded.
"""

import argparse

import numpy as np

from gym_softrobot.envs.snake.locomotion_task import SnakeLocomotionTask
from gym_softrobot.sim import RodTaskEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    env = RodTaskEnv(SnakeLocomotionTask())
    observation, info = env.reset(seed=args.seed)
    print(f"reset observation {observation.shape} info {info}")
    try:
        for index in range(args.steps):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[-1] = 1.0
            observation, reward, terminated, truncated, info = env.step(action)
            print(
                f"step {index + 1} reward {reward} "
                f"terminated {terminated} truncated {truncated}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
