# Run the continuum snake

We will step the existing continuum snake once, then point the same task at Isaac Lab. At the end of the first command you should see a reset observation of shape `(756,)` and one reward.

The snake is the `ContinuumSnakeEnv` rod. Fifty elements, seven action numbers. The task reward is the center-of-mass speed along +z over the control step that just finished, in meters per second. `ContinuumSnake-v0.step` still returns the original whole-run average. This task does not register a new `ContinuumSnake-v0`. `gym.make("ContinuumSnake-v0")` still builds the original environment.

## Step it on the CPU

From the repository root, run:

```bash
uv run python examples/isaac_snake/cpu_rollout.py --steps 1 --seed 0
```

You should see:

```text
reset observation (756,) info {}
step 1 reward 0.0 terminated False truncated False
```

The printed reward is forward speed for that 0.2 s step. This script commands six zero spline coefficients and wavelength `1.0`, which leaves the muscles off, so the value stays near zero. `terminated` stays false. `truncated` becomes true on step 111, which is the first step at or past 22.02 s.

The code that owns this run is `SnakeLocomotionTask` in `gym_softrobot/envs/snake/locomotion_task.py`. `RodTaskEnv` in `gym_softrobot/sim/gym_env.py` performs the Gymnasium `reset` and `step`.

## Step it in Isaac Lab

Isaac Lab 2.3.2 has to be installed separately. From that checkout, run:

```bash
./isaaclab.sh -p /path/to/gym-softrobot/examples/isaac_snake/random_agent.py --num_envs 1 --headless --steps 1
```

Replace the repository path with your gym-softrobot checkout. The script registers `SoftRobot-ContinuumSnake-Direct-v0` only after `AppLauncher` starts. `parse_env_cfg` builds the config, and `gym.make` builds `SoftRobotEnv`. The policy tensor has shape `(1, 756)` when `--num_envs` is 1.

The random actions are drawn in `[-1, 1]` and clipped to the snake's action bounds inside `SoftRobotEnv`. A finished env is reset before its observation is returned. The `terminated` and `truncated` tensors still refer to the step that ended.

Keep `--num_envs` at 1 until that single step is fast enough. Each extra env is another CPU rod simulation.

## Train it

From the same Isaac Lab checkout:

```bash
./isaaclab.sh -p /path/to/gym-softrobot/examples/isaac_snake/train.py --num_envs 1 --headless --max_iterations 5
```

RSL-RL collects 16 steps before the first mean reward. Five iterations is a smoke test. The log directory is `logs/rsl_rl/snake_locomotion/` under the working directory. Raise `--max_iterations` for a real run, and raise `--num_envs` only after one rod step is fast enough on that machine.

By default every env steps in the Isaac Lab process, one after another. Add `--rod_processes` to step them in worker processes instead. The worker count is the CPU count unless `--rod_workers` sets it, and it never exceeds `--num_envs`. Each worker is another Python process with its own copy of the rods, so envs beyond the worker count share a worker and step in turn.
