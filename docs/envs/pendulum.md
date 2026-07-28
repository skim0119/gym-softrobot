# Pendulum — stabilization

| Environment | Actuation | Action | Constraint | Objective |
| --- | --- | --- | --- | --- |
| `SoftPendulum-v0` | Point force applied to the rod | Continuous | One end fixed; planar motion | Reach the target orientation |
| `SoftPendulum3D-v0` | Horizontal translation of the base | Two continuous axes | Base orientation fixed; 3-D motion | Keep the rod upright |

## `SoftPendulum-v0`

This compact environment is a practical starting point for checking a control
pipeline. Its observation contains pendulum position, velocity, angle, and the
previous action.

## `SoftPendulum3D-v0`

The action contains two normalized commands in $[-1, 1]$ that increment the
base position along the horizontal axes. The reward penalizes squared tilt,
base displacement, and control effort.

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("SoftPendulum3D-v0")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```
