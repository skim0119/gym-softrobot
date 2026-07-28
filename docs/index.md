---
description: Soft-robot control environments powered by PyElastica.
---

# gym-softrobot

Soft robots generate motion through continuous deformation instead of a small
set of rigid joints. This gives them compliance and adaptability, but creates a
coupled design problem: their actuators are commonly distributed along the
body, and changing the actuation layout also changes the structure's dynamics.
Controllers must therefore cope with nonlinear continuum mechanics while the
appropriate actuator design may itself still be unknown.

gym-softrobot provides Gymnasium environments for exploring that problem with
high-fidelity PyElastica simulations. The environments span direct push-pull
forcing, tendon-driven actuation, distributed muscle-like activation, and
idealized force or torque inputs. Researchers can use these different levels of
abstraction to study control and actuator co-design, policy transfer,
curriculum learning from simple to physically constrained tasks, robustness,
and reinforcement learning for distributed soft-robot control.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
getting_started
```

```{toctree}
:maxdepth: 2
:caption: Environments

envs/index
```

```{toctree}
:maxdepth: 1
:caption: User guide

envs/wrappers
```

```{toctree}
:maxdepth: 1
:caption: Development

GitHub <https://github.com/skim0119/gym-softrobot>
Issue tracker <https://github.com/skim0119/gym-softrobot/issues>
```
