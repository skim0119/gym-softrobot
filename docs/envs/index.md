# Environments

gym-softrobot groups its environments by the soft-body system and control
objective. All public IDs are versioned so experiments can record the exact task
definition.

## Environment catalog

| Family | Environment | Objective | Control |
| --- | --- | --- | --- |
| Octopus | `OctoArmSingle-v0` | Move the tip of one soft arm | Distributed muscle activation |
| Octopus | `OctoArmTwo-v0` | Coordinate two arms | Muscle activation with controllable anchors |
| Octopus | `OctoArmPush-v0` | Push an object | Discrete arm commands |
| Octopus | `OctoArmPush-v1` | Push an object | Continuous arm commands |
| Octopus | `OctoArmPullWeight-v0` | Pull an attached weight | Continuous arm commands |
| Octopus | `OctoFlat-v0` | Control an eight-arm planar body | Per-arm muscle activation |
| Octopus | `OctoFlatLite-v0` | Run a reduced planar task | Single-arm activation |
| Octopus | `OctoCrawl-v0` | Move the body by coordinated crawling | Arm muscles and controllable anchors |
| Octopus | `OctoReach-v0` | Move the body toward a target | Coordinated arm muscles |
| Soft arm | `SoftArmTracking-v0` | Track a moving target with the arm tip | Muscle-torque profiles |
| Locomotion | `ContinuumSnake-v0` | Generate forward motion | Traveling-wave coefficients |
| Control | `SoftPendulum-v0` | Stabilize a deformable pendulum | Applied point force |
| Control | `SoftPendulum3D-v0` | Stabilize a vertical deformable pendulum | Two-axis base motion |

## Choosing an environment

- Start with `SoftPendulum-v0` for a compact control problem.
- Use `SoftPendulum3D-v0` for pendulum stabilization with coupled motion in
  three dimensions.
- Use `OctoArmSingle-v0` before moving to multi-arm octopus tasks.
- Use `ContinuumSnake-v0` for friction-driven locomotion.
- Use `SoftArmTracking-v0` for trajectory tracking.

The octopus tasks are the most computationally intensive. Their reduced
variants are useful for integration tests and early policy experiments.

```{toctree}
:hidden:

octopus
soft_arm
misc
wrappers
```
