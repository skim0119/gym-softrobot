# Environments

Environments are grouped by **theme**: a simulated body together with its task.
IDs within a theme are variants distinguished by actuation, action space,
dimensionality, constraints, or scene configuration.

```{seealso}
Theme pages compare those variants in detail. Use this table only to pick a
family, then open the theme page before training.
```

A practical order is pendulum (pipeline checks), arm (continuum reaching or
tracking), snake (traveling-wave locomotion), then CyberOctopus (distributed
muscle and whole-body coordination). CyberOctopus simulations are the most
expensive.

## Choose a starting point

| Family | Start here | What it is good for | Action and feedback | Rendering | Relative cost |
| --- | --- | --- | --- | --- | --- |
| Pendulum | `SoftPendulum-v0` | Fast control-pipeline checks and stabilization | Low-dimensional continuous action; observation returned every step | `rgb_array` | Low |
| Soft arm | `TendonArmReach-v0`, `TendonArmReach-v1` | Fixed-target reaching (v0) or moving-target tracking (v1) | Continuous tendon or torque control; feedback through tip/body state | `rgb_array` | Medium |
| Snake | `ContinuumSnake-v0` | Locomotion through anisotropic contact | Continuous traveling-wave parameters; feedback through rod state | `rgb_array`, `human` | Medium–high |
| CyberOctopus | `OctoMuscleCrawl-v0` | Distributed muscles, suction, contact, and whole-body crawling | `(8, 9)` muscle/suction command each control step; six-value body observation | `rgb_array` | High |

Every environment follows the Gymnasium `reset`/`step` contract. The action is
chosen by the controller or RL policy after receiving the latest observation;
the environment itself advances the physical simulation and returns the next
observation, reward, termination flags, and diagnostic `info`.

## Complete environment catalog

| Theme | Environments | Main variant details |
| --- | --- | --- |
| [Arm](arm.md) | `SoftArmTracking-v0`, `ElasticaArmTracking-v0`, `ElasticaArmReach-v0`, `TendonArmReach-v0`, `TendonArmReach-v1`, `ElasticaArmObstacle-v0`, `ElasticaArmObstacleRandom-v0` | Tracking, reaching, or obstacle reaching; torque or tendon actuation; 2-D or 3-D |
| [CyberOctopus](octopus.md) | `OctoArmSingle-v0`, `OctoArmTwo-v0`, `OctoArmPush-v0`, `OctoArmPush-v1`, `OctoArmPullWeight-v0`, `OctoFlat-v0`, `OctoFlatLite-v0`, `OctoCrawl-v0`, `OctoPhaseCrawl-v0`, `OctoMuscleCrawl-v0`, `OctoReach-v0` | Arm or whole-body task; abstract phase or explicit muscle actuation; fixed or controllable constraints |
| [Snake](snake.md) | `ContinuumSnake-v0` | Traveling-wave actuation with anisotropic ground contact |
| [Pendulum](pendulum.md) | `SoftPendulum-v0`, `SoftPendulum3D-v0` | Point-force or moving-base actuation; planar or 3-D motion |

See each theme page for the detailed comparison. All public IDs are versioned
so experiments can record the exact task definition.
