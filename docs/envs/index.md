# Environments

Environments are grouped by **theme**: a simulated body together with its task.
IDs within a theme are variants distinguished by actuation, action space,
dimensionality, constraints, or scene configuration.

| Theme | Environments | Main variant details |
| --- | --- | --- |
| [Arm](arm.md) | `SoftArmTracking-v0`, `ElasticaArmTracking-v0`, `ElasticaArmReach-v0`, [`TendonArmReach-v0`](tendon_arm.md), `ElasticaArmObstacle-v0`, `ElasticaArmObstacleRandom-v0` | Tracking, reaching, or obstacle reaching; torque or tendon actuation; 2-D or 3-D |
| [Octopus](octopus.md) | `OctoArmSingle-v0`, `OctoArmTwo-v0`, `OctoArmPush-v0`, `OctoArmPush-v1`, `OctoArmPullWeight-v0`, `OctoFlat-v0`, `OctoFlatLite-v0`, `OctoCrawl-v0`, `OctoReach-v0` | Arm or whole-body task; discrete or continuous actuation; fixed or controllable constraints |
| [Snake](snake.md) | `ContinuumSnake-v0` | Traveling-wave actuation with anisotropic ground contact |
| [Pendulum](pendulum.md) | `SoftPendulum-v0`, `SoftPendulum3D-v0` | Point-force or moving-base actuation; planar or 3-D motion |

See each theme page for the detailed comparison. All public IDs are versioned
so experiments can record the exact task definition.

```{toctree}
:maxdepth: 2
:caption: Environment themes
:hidden:

arm
octopus
snake
pendulum
```
