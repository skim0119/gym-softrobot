# Locomotion and control

## `ContinuumSnake-v0`

- **Task:** propel a continuum snake over an anisotropic frictional plane.
- **Action:** traveling-wave spline coefficients and wavelength.
- **Observation:** rod position, velocity, and director frames.
- **Reward:** average forward velocity.
- **Episode end:** the simulation horizon or an invalid state.

The task is adapted from PyElastica's continuum-snake example.

## `SoftPendulum-v0`

- **Task:** control a deformable pendulum constrained at one end.
- **Action:** a point force applied to the rod.
- **Observation:** pendulum position, velocity, angle, and previous action.
- **Reward:** progress toward the target orientation.
- **Episode end:** the simulation horizon or an invalid state.

This is the smallest environment in the package and a practical starting point
for checking a control pipeline.

## `SoftPendulum3D-v0`

- **Task:** keep a vertical deformable pendulum upright by moving its constrained
  base in the horizontal plane.
- **Action:** two normalized commands in the range $[-1, 1]$ that increment the
  base position along the $x$ and $y$ axes.
- **Observation:** a nine-value vector containing the base position, base
  velocity, previous action, and pendulum tilt angle.
- **Reward:** the negative weighted sum of squared tilt, base displacement, and
  control effort.
- **Episode end:** the simulation horizon or an invalid simulation state.
- **Rendering:** `rgb_array`.

Create the environment through Gymnasium:

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("SoftPendulum3D-v0")
observation, info = env.reset(seed=42)

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

An optional PPO training script is available at
`examples/soft_pendulum_3d/train_ppo.py`. It requires Stable-Baselines3 in
addition to the core package dependencies.
