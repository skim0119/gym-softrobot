# Isaac Lab and PyElastica

gym-softrobot keeps the Cosserat rod in PyElastica. Isaac Lab 2.3.2 can own the training loop, the batched `num_envs` step, and the USD view of that rod. The two programs meet at one object, `RodBatch`.

A researcher who already calls `gym.make` can run the same task through `RodTaskEnv` with no Isaac Sim install. A researcher who has Isaac Lab launches `AppLauncher`, registers one direct task, and then calls `gym.make` the way Isaac Lab's own scripts do. The next person to add a task writes `build`, `apply_action`, `observe`, and `reward`. They do not write a second environment class for the Isaac path.

## Why the physics stays in PyElastica

Isaac Lab's scene cloner copies a PhysX articulation. A PyElastica rod is not one of those articulations. Stepping `num_envs` snakes means stepping `num_envs` independent rod simulators in one process and stacking their observations into arrays of shape `(num_envs, ...)`. That is slower than a GPU PhysX scene, and it is the parallelism this package can offer for the current engine. `SoftRobotEnv` still subclasses `DirectRLEnv`, so Isaac Lab's reset bookkeeping, episode clock, and `rsl_rl` style wrappers see a normal direct task. PhysX advances an empty stage and acts as the clock. The rod moves inside `_apply_action`.

Rendering does not read the observation vector. Each frame copies centerline positions and per-element radii. `reset` builds a new rod, and the next frame reads that rod. A viewer that kept the old rod object would keep drawing the previous episode.

## What a second physics engine implements

A later engine implements `RodBatch`. `reset` rebuilds the listed worlds and returns their observations. `step` advances every world by one control interval and returns observations, rewards, and a failure flag. `frame` returns copied centerlines and radii. `SoftRobotTask.make_batch` is where the task chooses that engine. `RodTaskEnv` and `SoftRobotEnv` stay as they are.

Isaac Lab's develop branch describes a `PhysicsManager` that other engines can plug into `SimulationContext`. That interface is not part of the 2.3.2 release this package targets. When it is, the manager should wrap a `RodBatch` and `SoftRobotEnv` should stop driving the batch from `_apply_action`. The centerline copy is the data that manager would hand to a scene-data backend.

## What stays out of the core install

Importing `gym_softrobot` does not import `isaaclab` and does not register an Isaac task id. Isaac Lab is not a uv dependency group, because CI runs `uv sync --frozen --all-groups` and cannot lock that package. Tests under `tests/sim` use a fake rod. They do not call `ContinuumSnakeEnv.step`. An entry point that contains `gym_softrobot.isaac` is left out of the default environment list, so an accidental registration is not constructed by the suite.
