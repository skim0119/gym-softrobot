# Octopus environments

The octopus family uses tapered Cosserat rods as arms and, where needed, a rigid
central body. Choose a task based on how many arms must be coordinated and
whether the objective is manipulation or locomotion.

## Arm manipulation

### `OctoArmSingle-v0`

- **Task:** move a single arm tip while the base remains constrained.
- **Action:** muscle activations distributed along the arm.
- **Observation:** sampled arm geometry, velocity, and previous control state.
- **Reward:** progress of the arm tip toward the task objective, with stability
  penalties.

### `OctoArmTwo-v0`

- **Task:** coordinate two muscle-driven arms.
- **Action:** activation profiles for both arms and their controllable anchor
  points.
- **Observation:** per-arm state and shared geometric state.
- **Reward:** coordinated progress toward the target configuration.

### `OctoArmPush-v0` and `OctoArmPush-v1`

- **Task:** use a soft arm to push a rigid body.
- **Action:** discrete commands in `v0`; continuous muscle commands in `v1`.
- **Observation:** arm state, object state, and relative geometry.
- **Reward:** object displacement in the desired direction.

### `OctoArmPullWeight-v0`

- **Task:** pull an attached rigid weight with one arm.
- **Action:** continuous muscle activation.
- **Observation:** arm and weight motion.
- **Reward:** progress of the weight toward the pull objective.

## Whole-body tasks

### `OctoFlat-v0` and `OctoFlatLite-v0`

- **Task:** control a planar octopus body through its arms.
- **Action:** per-arm muscle activations.
- **Observation:** arm states together with central-body position and velocity.
- **Reward:** body displacement and control progress.
- **Variant:** `OctoFlatLite-v0` uses one arm and is cheaper to simulate.

### `OctoCrawl-v0`

- **Task:** propel the central body using coordinated arms and controllable
  anchors.
- **Action:** muscle activations and anchor control.
- **Observation:** shared body state and per-arm state.
- **Reward:** forward crawling progress.

### `OctoReach-v0`

- **Task:** move the octopus toward a target.
- **Action:** coordinated muscle activation across the arms.
- **Observation:** arm state, body state, and target-relative position.
- **Reward:** reduction in distance to the target.

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

These simulations are comparatively expensive. Use a short horizon and a
reduced environment while validating a training pipeline.
