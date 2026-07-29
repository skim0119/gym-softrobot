# Pendulum — stabilization

| Environment | Actuation | Action | Constraint | Objective |
| --- | --- | --- | --- | --- |
| `SoftPendulum-v0` | Point force applied to the rod | Continuous | One end fixed; planar motion | Reach the target orientation |
| `SoftPendulum3D-v0` | Horizontal translation of the base | Two continuous axes | Base orientation fixed; 3-D motion | Keep the rod upright |

## `SoftPendulum-v0`

This compact environment is a practical starting point for checking a control
pipeline. Its observation contains pendulum position, velocity, angle, and the
previous action.

The pendulum is a deformable Cosserat rod with one constrained end. Its
distributed mass and bending stiffness create many vibration modes rather than
the single angle of an ideal rigid pendulum. The point-force action injects
linear momentum locally; the resulting lever arm bends the rod and changes its
orientation. Elastic restoring moments, inertia, damping, and the boundary
reaction at the clamp determine the response.

## `SoftPendulum3D-v0`

The action contains two normalized commands in $[-1, 1]$ that increment the
base position along the horizontal axes. The reward penalizes squared tilt,
base displacement, and control effort.

Here the controller does not push the rod directly. Translating the constrained
base accelerates the support, and the rod responds through inertia and elastic
waves transmitted from the boundary. The base director remains fixed, so the
task resembles stabilization by cart motion, but with a flexible distributed
body whose higher modes can be excited by abrupt commands. Motion in two
horizontal directions couples the two bending planes.

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
