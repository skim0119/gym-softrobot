# Soft Arm

The continuum-arm environments share a slender Cosserat-rod body, but differ in
task, actuation, and workspace constraints.

In each case, the arm is a distributed mechanical system rather than a chain of
rigid links. Centerline positions and material frames evolve under inertia,
elastic force and moment resultants, damping, boundary reactions, and applied
loads. The actuation choice determines how a low-dimensional policy couples to
those equations: idealized B-spline torques act directly as a smooth moment
density, whereas routed tendons generate forces and offset moments at discrete
vertebrae. The obstacle tasks additionally solve compliant rod-cylinder
contact.

## Tracking

| Environment | Actuation | Action | Workspace | Target |
| --- | --- | --- | --- | --- |
| `SoftArmTracking-v0` | B-spline bending torques in two material directions | Continuous | 3-D | Fixed by default |
| `ElasticaArmTracking-v0` | Six B-spline controls in each bending direction | Continuous | 3-D | Procedurally moving |

## Reaching

| Environment | Actuation | Action | Workspace | Goal |
| --- | --- | --- | --- | --- |
| `ElasticaArmReach-v0` | Bending and tangent twist torques | Continuous | 3-D | Tip position and orientation |
| `TendonArmReach-v0` | Four full-length and four half-length tendons | Continuous | 3-D | Tip position |

See the [tendon-arm guide](tendon_arm.md) for its physical model, action and
observation definitions, reward terms, source provenance, and training example.

## Obstacle reaching

| Environment | Actuation | Action | Workspace | Obstacles |
| --- | --- | --- | --- | --- |
| `ElasticaArmObstacle-v0` | One-axis B-spline bending | Continuous, underactuated | 2-D | Eight fixed cylinders |
| `ElasticaArmObstacleRandom-v0` | Two-axis B-spline bending | Continuous, underactuated | 3-D | Twelve seeded random cylinders |

The `ElasticaArm*` variants reproduce Cases 1–4 of the Elastica RL benchmark.
See the [benchmark provenance and training notes](elastica_rl.md) for their
mapping to the upstream implementations.

## Usage

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("TendonArmReach-v0")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

```{toctree}
:hidden:

elastica_rl
tendon_arm
```
