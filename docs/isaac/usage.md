# How to run a soft-robot task with Isaac Lab

Use the CPU shell when Isaac Sim is not installed. Use the Isaac Lab script when you want that framework's `AppLauncher`, batched `num_envs`, and USD curves. Both shells call the same task.

## Run one snake step without Isaac Lab

From the repository root, with the project environment active, run:

```bash
uv run python examples/isaac_snake/cpu_rollout.py --steps 1 --seed 0
```

The process prints a reset observation of shape `(756,)` and one line for the step. The reward on that first step is `0.0`. `terminated` and `truncated` are both false. The reward stays `0.0` until the callback history covers two gait periods. One step is 0.2 s of rod time.

To take more steps, pass `--steps`. The episode truncates on step 111.

## Run the same task inside Isaac Lab 2.3.2

Install Isaac Lab using NVIDIA's Isaac Lab instructions for the 2.3.2 release. This repository does not install it for you.

From the Isaac Lab checkout, point `isaaclab.sh` at this file. Replace the repository path with your checkout.

```bash
./isaaclab.sh -p /path/to/gym-softrobot/examples/isaac_snake/random_agent.py --num_envs 1 --headless --steps 1
```

`AppLauncher` starts first. The script then registers `SoftRobot-ContinuumSnake-Direct-v0` and builds it with `parse_env_cfg` and `gym.make`. The printed policy observation has shape `(num_envs, 756)`.

`--num_envs` is the number of PyElastica snakes. Each one steps on the CPU. Start with `1`. Raise it only after a single snake step is acceptable on that machine.

If this file is started with the gym-softrobot interpreter, it exits and prints the `isaaclab.sh` command. It does not try to import Omniverse.

## Add another task

Write an `ElasticaTask` with a `TaskSpec` and the four methods `build`, `apply_action`, `observe`, and `reward`. Run it locally with `RodTaskEnv`. For Isaac Lab, set `SoftRobotEnvCfg.task_entry` to `"module:Class"` and register `gym_softrobot.isaac.env:SoftRobotEnv` after `AppLauncher`. Do not register that id from `gym_softrobot/__init__.py`.
