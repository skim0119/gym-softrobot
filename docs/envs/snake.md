# Snake — locomotion

## `ContinuumSnake-v0`

| Detail | Description |
| --- | --- |
| Task | Generate forward locomotion |
| Actuation | Traveling-wave muscle torque |
| Action | Continuous spline coefficients and wavelength |
| Constraint | Contact with an anisotropic-friction plane |
| Workspace | Planar locomotion of a three-dimensional rod model |

The reward measures average forward velocity. An episode ends at the simulation
horizon or when the simulation becomes invalid. The task is adapted from
PyElastica's continuum-snake example.
