# Octopus-muscle crawler

This page documents the current `OctoMuscleCrawl-v0` environment. The older
`OctoCrawl-v0` remains registered as a legacy compatibility environment and
is documented separately on the [CyberOctopus overview](octopus.md).

`OctoMuscleCrawl-v0` is a feedback-ready crawling task for an eight-arm
octopus with independently commanded muscle groups. It exposes the
phase-physics muscle simulator as a standard Gymnasium environment: each
`step` applies one normalized command per muscle channel and arm, advances one
control interval, and returns the new observation and reward. The environment
does not choose the next action; an RL policy or feedback controller does that
from the observation returned by `step`.

This environment sits at a more detailed actuation level than
`OctoCrawl-v0`. Instead of a mixed muscle-and-anchor interface, the policy
commands transverse, longitudinal, and oblique muscle groups together with
two suction bands on each arm. Ground friction, sucker attachment, and
elastic arm–body coupling then determine whether those activations produce
net translation.

```{note}
`OctoPhaseCrawl-v0` uses the same body and suction layout with a lower-level
phase action (stiffness, extension, suction, bend). This page is the explicit
muscle-group interface. Open-loop phase-Gaussian parameters remain in
the internal crawling simulation modules for a later whole-episode wrapper; they are not the
Gymnasium action.
```

## A first feedback loop

The action is selected after observing the current state. This is the intended
RL interface and is different from replaying a precomputed gait:

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("OctoMuscleCrawl-v0")
observation, info = env.reset(seed=1)

terminated = truncated = False
while not (terminated or truncated):
    action = controller(observation)  # shape: (8, 9), values in [-1, 1]
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

`controller` may be an RL policy, a hand-written controller, or a random
policy for a smoke test. The environment advances the mechanics by one
`control_dt` interval per call, so the controller can react to the returned
body displacement and velocity at every step.

## Physical model

The robot has a rigid central body modeled as a sphere of radius 0.09 m and
eight tapered Cosserat-rod arms. By default each arm has length 0.45 m, base
radius 0.02 m, and 15 elements. Arms are spaced evenly around the body in the
horizontal plane. Gravity acts in the negative $y$ direction, and the ground
is the plane $y = -R_{\mathrm{arm}}$.

Each arm is tethered to the sphere, so distributed muscle forces transmit
loads into the rigid body. Rod–plane contact uses anisotropic kinetic
friction; the sphere uses the same ground plane with Hertz contact and
damping. Optional rod–rod contact is disabled by default.

Muscle actuation is applied through COOMM `BatchMuscle` groups on each arm:

| Group | Count per arm | Mechanical role |
| --- | ---: | --- |
| Transverse muscle (TM) | 1 | Circumferential contraction |
| Longitudinal muscle (LM0–LM3) | 4 | Axial contraction at $90^\circ$ offsets |
| Oblique muscle (OM+, OM−) | 2 | Helical contraction of opposite handedness |

The four longitudinal groups sit at angles $0$, $\pi/2$, $\pi$, and
$3\pi/2$. Each oblique channel comprises two fibers that share an activation
and wind with rotation number $\pm 6$. Muscle force is scaled by the
force–length weight

```{math}
w(\ell)=\max\bigl(0,\,1-5(\ell-1)^{2}\bigr),
```

so activation is most effective near rest length and vanishes for large
stretch or shortening.

Suction is not a muscle group. Base and middle sucker bands occupy element
ranges $[1,3]$ and $[6,8]$. When a suction channel is active, those nodes
are pulled toward the ground plane by a spring–damper, providing temporary
anchors. Crawling therefore still depends on an asymmetric attach–deform–
release cycle: suction and friction supply reaction forces while muscle
activation changes arm shape.

The integrator is Position Verlet. The default simulation step is
$3\times 10^{-4}$ s. Each environment step advances one control interval of
$1/60$ s.

## Action space

The action is a `Box(-1, 1, shape=(8, 9), dtype=float32)`. Rows are arms.
Columns are, in order:

| Index | Channel |
| ---: | --- |
| 0 | TM |
| 1–4 | LM0, LM1, LM2, LM3 |
| 5 | Base suction |
| 6 | Middle suction |
| 7 | OM+ |
| 8 | OM− |

Each entry $a_{i,j}$ is mapped to a nonnegative activation

```{math}
u_{i,j}=\mathrm{clip}\bigl((a_{i,j}+1)/2,\,0,\,1\bigr).
```

Thus `-1` is off and `1` is full activation. The environment writes these
values directly into the muscle and suction targets for the next control
interval; it does not apply the open-loop phase-Gaussian schedule.

## Observation space

The observation is a six-value `float32` vector for the central body:

| Values | Description |
| ---: | --- |
| 3 | Displacement of the sphere from its pose at `reset` |
| 3 | Sphere translational velocity |

Arm shape, muscle activation, and suction state are not included. A policy
that needs those signals must reconstruct them from its own action history.

## Reward and episode

Forward progress is measured along $-z$, the default crawl heading implied
by the arm layout. The current Gymnasium muscle-crawl environment uses the
following per-step reward:

```{math}
r=\frac{\Delta z_{\text{forward}}}{L}
-\lambda\frac{|x|}{L},
```

where $L$ is the arm rest length and $\lambda=0.5$ penalizes lateral
drift. A non-finite simulation state sets `terminated=True` and replaces the
reward with a large failure penalty. Reaching the time limit sets
`truncated=True`.

The default horizon is the number of control intervals in five locomotion
cycles of period $T_L=2.4$ s, or 12 s of simulated time (720 steps at
60 Hz). `info` reports `time`, `forward_progress`, `lateral_deviation`, and
`strain_energy`.

## Rendering

The environment supports an RGB-array renderer for diagnostics and video
generation:

```python
env = gym.make("OctoMuscleCrawl-v0", render_mode="rgb_array")
observation, info = env.reset(seed=1)
frame = env.render()  # uint8 RGB NumPy array
```

Call `render()` after `reset()` and after any `step()` whose state you want to
record. The current environment does not provide a human-window renderer;
video writers and plotting tools can consume the returned frames.

## Configuration and reduced-cost experiments

The registered environment uses `OctopusMuscleConfig` defaults: 15 elements
per arm, a `3e-4` s physics step, a `1/60` s control interval, and a five-cycle
episode. For development and tests, pass a smaller configuration through
`gym.make` and shorten the horizon:

```python
from gym_softrobot.envs.octopus.crawling_simulation.config import (
    OctopusMuscleConfig,
)

config = OctopusMuscleConfig(
    n_elem=5,
    control_dt=0.01,
    time_step=0.001,
    episode_duration_cycles=0.01,
)
env = gym.make("OctoMuscleCrawl-v0", config=config, horizon=2)
```

The full model is intentionally detailed and can be expensive. Use a reduced
configuration to validate an RL pipeline before starting long experiments.

## Fixed-policy video example

The environment can also produce a reproducible demonstration from a
precomputed action sequence. This example is intentionally open-loop; it is
not a replacement for the feedback loop above:

```bash
uv run python examples/octopus_crawling/replay_policy.py
```

`examples/octopus_crawling/policy.csv` stores normalized actions as flattened
`(T, 72)` rows, corresponding to an action tensor of logical shape
`(8, 9, T)`. The script still interacts with the environment only through
`gym.make`, `reset`, `step`, and `render`.

## Usage

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("OctoMuscleCrawl-v0")
observation, info = env.reset(seed=1)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

```{seealso}
For the broader Octopus family and legacy compatibility environments, see the
[CyberOctopus overview](octopus.md).
```
