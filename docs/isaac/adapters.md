# Isaac Lab adapters

These types live in `gym_softrobot.sim` and `gym_softrobot.isaac`. Importing `gym_softrobot` does not import either package. Import `gym_softrobot.isaac.env` only after `AppLauncher` has started Omniverse.

## `TaskSpec`

Frozen description of one task, valid before any simulator exists.

| Field | Meaning |
| --- | --- |
| `action_space` | One-dimensional `float32` `Box` for a single world. |
| `observation_space` | One-dimensional `float32` `Box` for a single world. |
| `control_dt` | Seconds of simulation time in one policy step. Positive. |
| `horizon` | Policy steps in an episode. Integer, at least 1. |
| `rod_nodes` | Node count of each rod. Each count is an integer of at least 2. |

`episode_length_s` is `horizon * control_dt`. Construction raises `ValueError` when a field breaks those rules.

## `BatchStep`

One transition for `N` worlds.

| Field | Shape |
| --- | --- |
| `obs` | `(N, observation_size)` `float32` |
| `reward` | `(N,)` `float32` |
| `failed` | `(N,)` bool. True means that world failed, including a non-finite rod position. |

## `BatchFrame`

Copied render data. Holding a frame does not keep a rod alive.

| Field | Shape |
| --- | --- |
| `centerlines` | One array per rod, each `(N, 3, K)` `float64`. `K` is that rod's node count. |
| `radii` | One array per rod, each `(N, K - 1)` `float64`. These are per-element radii. |

A one-dimensional array raises `ValueError`.

## `RodBatch`

Protocol for `N` worlds.

| Method | Behavior |
| --- | --- |
| `reset(env_ids, rngs)` | Rebuilds each listed world with the paired generator. Returns observations for those rows, shape `(len(env_ids), observation_size)`. |
| `step(actions)` | Advances every world by `control_dt`. Returns a `BatchStep`. Raises `RuntimeError` if any world was never reset. |
| `frame()` | Returns a `BatchFrame` of copies. |
| `close()` | Drops the worlds. |

`actions` has shape `(N, action_size)`.

## `ElasticaTask`

PyElastica implementation of the batch. Subclasses set `spec` and implement `build`, `apply_action`, `observe`, and `reward`. `failed` defaults to false. The batch also marks a world failed when a rod position is non-finite. `make_batch(num_envs)` returns the batch. `build` receives one `numpy.random.Generator` and is the only place that task may draw random numbers.

`ElasticaWorld` holds one finalized simulator, a timestepper, `dt`, the rods, and the current time. The batch calls `timestepper.step(simulator, time, dt)` once per integrator step. `control_dt / dt` must be an integer within a relative tolerance of `1e-9`, checked at reset.

## `RodTaskEnv`

Single-world `gymnasium.Env` over `make_batch(1)`.

`render_mode` is `None` or `"rgb_array"`. `reset(seed=...)` calls `gymnasium.Env.reset` and passes `self.np_random` into the batch. `step` returns `(observation, reward, terminated, truncated, info)`. `terminated` is the batch failure flag. `truncated` becomes true when the step count reaches `horizon`. The env does not reset itself on that step. `render` returns `None` when `render_mode` is `None`, and a `uint8` RGB array of shape `(height, width, 3)` otherwise. The image is a top-down plot of the centerline.

## `SoftRobotEnv`

`DirectRLEnv` in `gym_softrobot.isaac.env`. `SoftRobotEnvCfg.task_entry` is a string `module:Class`. The constructor loads that class, copies its `TaskSpec` into `action_space`, `observation_space`, `sim.dt`, `decimation` (always 1), and `episode_length_s`, then initializes `DirectRLEnv`.

`reset` zeroes per-environment episode counts and calls `DirectRLEnv.reset`. It does not call `gymnasium.Env.reset`. `DirectRLEnv.reset` seeds through `DirectRLEnv.seed`. Generators for each world come from `SeedSequence` with spawn key `(env_index, episode)`. An automatic reset increments that episode count. A full `reset` does not.

`step` follows `DirectRLEnv`. One policy step calls the batch once. Torch actions are clipped to the `TaskSpec` bounds and converted to `float32` before the batch sees them. Observations are `{"policy": tensor}` on the environment device. A world that hits the horizon or fails is reset before the observation is returned. The returned `terminated` and `truncated` flags still describe the transition that ended. `truncated` is true when `episode_length_buf >= max_episode_length`. With `sim.dt = control_dt` and `decimation = 1`, that length is `horizon`.

USD `BasisCurves` under each env prim store the centerline. Widths are `2 * radius` at nodes. End nodes copy the neighboring element radius. Interior nodes average the two neighboring elements. A curve prim is replaced when the node count changes.
