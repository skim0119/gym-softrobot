# Octopus Control

Octopus environments use tapered Cosserat rods for arms and, for whole-body
tasks, a rigid central body. The tables below make the variant-defining details
explicit.

## Arm — positioning and manipulation

| Environment | Task | Actuation | Action | Constraint or scene |
| --- | --- | --- | --- | --- |
| `OctoArmSingle-v0` | Position one arm tip | Distributed muscle activation | Continuous | Arm base constrained |
| `OctoArmTwo-v0` | Coordinate two arms | Muscle activation and anchor control | Continuous | Controllable arm anchors |
| `OctoArmPush-v0` | Push a rigid object | Preset arm command | Discrete | Arm base constrained |
| `OctoArmPush-v1` | Push a rigid object | Distributed muscle activation | Continuous | Arm base constrained |
| `OctoArmPullWeight-v0` | Pull an attached weight | Distributed muscle activation | Continuous | Arm base constrained; rigid weight attached |

The two push IDs share a body and task. Their principal difference is the
actuation interface: `v0` selects between discrete commands, while `v1`
directly supplies continuous muscle commands.

## Body — planar control

| Environment | Task | Actuation | Action | Constraint or scene |
| --- | --- | --- | --- | --- |
| `OctoFlat-v0` | Control a full octopus body | Per-arm curvature control | Continuous | Eight-arm planar model |
| `OctoFlatLite-v0` | Run the reduced task | Curvature control | Continuous | One-arm planar model |

`OctoFlatLite-v0` preserves the broad task while reducing the number of arms
and simulation cost.

## Body — crawling and reaching

| Environment | Task | Actuation | Action | Constraint or scene |
| --- | --- | --- | --- | --- |
| `OctoCrawl-v0` | Crawl by coordinating the arms | Arm muscles and anchor control | Continuous | Eight arms with controllable anchors |
| `OctoReach-v0` | Move toward a target | Coordinated arm muscles | Continuous | Eight arms; central rigid body fixed |

## Usage

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("OctoArmSingle-v0")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

These simulations are comparatively expensive. Start with a reduced model or
short horizon while validating a training pipeline.
