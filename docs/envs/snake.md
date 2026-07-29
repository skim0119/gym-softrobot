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

## Physics

The snake is a three-dimensional Cosserat rod whose centerline remains close to
a horizontal support plane. A traveling internal torque wave bends the body
laterally. The action changes the spatial waveform through spline coefficients
and its wavelength, so the policy controls body curvature rather than applying
a net propulsive force.

Locomotion emerges from contact. The plane supplies a normal reaction that
prevents penetration and a direction-dependent friction force opposing slip.
Tangential and lateral friction coefficients differ, breaking the symmetry
that would otherwise make a reciprocal bending cycle produce no net motion.
The rod must continually trade elastic bending energy, inertia, contact
damping, and frictional dissipation. Forward progress is therefore an emergent
result of the gait and substrate interaction rather than a prescribed velocity.
