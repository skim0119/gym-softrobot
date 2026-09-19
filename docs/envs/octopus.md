# CyberOctopus

CyberOctopus groups the octopus-arm and whole-body environments. They use
tapered Cosserat rods for arms and, for whole-body tasks, a rigid central body.
The tables below make the variant-defining details explicit.

## Shared mechanics

Each soft arm is a tapered Cosserat rod, so its centerline and cross-section
frames resolve extension, shear, bending, and twist along the body. Tapering
changes mass and bending stiffness with arclength: proximal sections resist
curvature more strongly, while distal sections respond more readily. The
whole-body variants attach several arms to a rigid central body, exchanging
forces and torques through their connections.

Muscle commands prescribe distributed activation or preferred curvature along
an arm. They generate internal moments rather than teleporting the tip, so
motion depends on the competition between active torque, elastic resistance,
inertia, damping, and any external contact. Controllable anchors add or remove
kinematic constraints; switching an anchor changes which reaction forces can be
used for manipulation or locomotion.

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

In the single- and two-arm tasks, a constrained base converts distributed
muscle moment into arm deformation and tip motion. The push variants add
soft-arm/rigid-object interaction: compliant contact transfers momentum only
when the bodies touch, so object motion depends on approach geometry, arm
stiffness, and impact dynamics. The pull task instead transmits load through an
attachment, coupling the arm's elastic deformation to the inertia of the
weight.

## Body — planar control

| Environment | Task | Actuation | Action | Constraint or scene |
| --- | --- | --- | --- | --- |
| `OctoFlat-v0` | Control a full octopus body | Per-arm curvature control | Continuous | Eight-arm planar model |
| `OctoFlatLite-v0` | Run the reduced task | Curvature control | Continuous | One-arm planar model |

`OctoFlatLite-v0` preserves the broad task while reducing the number of arms
and simulation cost.

The planar construction suppresses out-of-plane motion but retains distributed
arm elasticity and the rigid-body coupling at the center. With eight arms,
simultaneous curvature commands can create cancelling or reinforcing body
forces and moments; the lite variant isolates the mechanics of one arm.

## Body — crawling and reaching

```{caution}
`OctoCrawl-v0` is retained for older experiments and PyMARL-style
decentralized controllers. Its action, observation, physics, and reward
contracts differ from the migrated crawlers. New work should use
`OctoPhaseCrawl-v0` for abstract phase actuation or `OctoMuscleCrawl-v0` for
explicit TM/LM/OM muscle control.
```

| Environment | Task | Actuation | Action | Constraint or scene |
| --- | --- | --- | --- | --- |
| `OctoCrawl-v0` | Legacy crawl task | Arm muscles and anchor control | Continuous | Eight arms with controllable anchors; retained for compatibility |
| `OctoPhaseCrawl-v0` | Feedback-controlled crawling | Abstract stiffness, extension, suction, and bend | Continuous | Eight arms; one action per control interval |
| `OctoMuscleCrawl-v0` | Feedback-controlled crawling | TM/LM/OM muscle groups and suction | Continuous | Eight arms; one action per control interval |
| `OctoReach-v0` | Move toward a target | Coordinated arm muscles | Continuous | Eight arms; central rigid body fixed |

Crawling requires an asymmetric interaction cycle. Anchored arms provide
reaction forces while other arms recover, and changing the anchor pattern
converts otherwise reciprocal deformation into translation. Reaching uses the
combined elastic loads from several arms, making coordination important because
unbalanced activation can rotate or deform the assembly instead of advancing
it toward the target.

```{seealso}
The [muscle-crawler guide](octopus_muscle.md) documents `OctoMuscleCrawl-v0`:
Cosserat arms and suction, the nine-channel muscle action, the body
observation, and the crawl reward.
```

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

```{warning}
CyberOctopus simulations are comparatively expensive. Validate a training
pipeline on a reduced model or a short horizon before full-length rollouts.
```

```{toctree}
:hidden:

octopus_muscle
```
