# Octopus-muscle crawler

This page documents the current `OctoMuscleCrawl-v0` environment. The older
`OctoCrawl-v0` remains registered as a legacy compatibility environment and
is documented separately on the [Octopus overview](octopus.md).

`OctoMuscleCrawl-v0` is a closed-loop crawling task for an eight-arm octopus
with independently commanded muscle groups. It exposes the phase-physics
muscle simulator as a standard Gymnasium environment: each `step` applies one
normalized command per muscle channel and arm, then advances one control
interval.

This environment sits at a more detailed actuation level than
`OctoCrawl-v0`. Instead of a mixed muscle-and-anchor interface, the policy
commands transverse, longitudinal, and oblique muscle groups together with
two suction bands on each arm. Ground friction, sucker attachment, and
elastic arm–body coupling then determine whether those activations produce
net translation.

The reusable `phase_physics` layer still contains the open-loop
phase-Gaussian parameters used by the earlier full-rollout bandit interface.
Those parameters are not the Gymnasium action; they remain available for a
later whole-episode wrapper.

## Physical model

The robot has a rigid central body modeled as a sphere of radius 0.09 m and
eight tapered Cosserat-rod arms. By default each arm has length 0.45 m, base
radius 0.02 m, and 15 elements. Arms are spaced evenly around the body in the
horizontal plane. Gravity acts in the negative $y$ direction, and the ground
is the plane $y = -R_{\mathrm{arm}}$.

Each arm is tethered to the sphere, so distributed muscle forces transmit
loads into the rigid body. Rod–plane contact uses anisotropic kinetic
friction; the sphere has its own plane contact and damping. Optional
rod–rod contact is disabled by default.

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
by the arm layout. After each control interval the reward is

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

The default model is computationally expensive. For smoke tests, construct
`OctoMuscleCrawlEnv` with a reduced `OctopusMuscleConfig` (fewer elements,
a larger time step, and a short horizon) rather than sampling full-length
rollouts.
