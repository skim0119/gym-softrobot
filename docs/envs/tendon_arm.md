# Tendon-driven continuum arm

`TendonArmReach-v0` is a three-dimensional reaching task for a tendon-driven
continuum arm. It ports the core environment design from
[`gabotuzl/rl-cr-robot`](https://github.com/gabotuzl/rl-cr-robot) to Gymnasium
and current PyElastica. The port retains the reference project's eight-tendon
layout, target distribution, observation structure, action scaling, and shaped
reward while using seeded Gymnasium randomness and explicit
`terminated`/`truncated` episode semantics.

This environment complements the idealized torque-controlled soft-arm tasks.
Tendon forces are routed through discrete vertebra locations, so actions act on
the rod through a physically interpretable transmission rather than as direct
bending torques.

## Physical model

The arm is a 0.25 m Cosserat rod fixed at its base. Its default model uses 100
elements and advances PyElastica with a Position Verlet integrator at a
`3e-5` s time step. Linear damping suppresses unresolved high-frequency motion.
Gravity is disabled in the reference task.

Unlike a rigid serial manipulator, the simulated arm has distributed
translational and rotational degrees of freedom. The Cosserat formulation
resolves centerline stretching and shear together with bending and twist of the
material frames. Internal force and moment resultants are computed from these
strains using the rod's elastic constitutive law. Fixing the first node and
director frame creates a clamped base while leaving the remainder of the body
free to deform in three dimensions.

Eight tendons are divided into two antagonistic groups:

- four tendons run through six vertebrae over approximately 98% of the arm;
- four tendons run through six vertebrae over approximately 50% of the arm.

Within each group, tendons are placed at 90-degree intervals around the rod.
The long and short groups use routing radii of 15 mm and 8 mm, respectively.
Forces follow the tendon segments between vertebrae, and their offsets from the
centerline also produce torques on the rod.

At an intermediate vertebra, a tendon changes direction. The two adjacent
tension vectors therefore produce the nodal force

```{math}
\mathbf{f}_{i,j}
= T_i\left(\hat{\mathbf{t}}_{i,j}^{+}
- \hat{\mathbf{t}}_{i,j}^{-}\right),
```

where $T_i$ is the tendon tension and the unit vectors point along the
outgoing and incoming tendon segments. Because the tendon passes through an
offset $\mathbf{r}_{i,j}$ in the local cross-section, it also applies

```{math}
\boldsymbol{\tau}_{i,j}
= \mathbf{r}_{i,j}\times\mathbf{f}_{i,j}.
```

Opposing tendons can therefore generate bending in either transverse direction.
Co-activating tendons increases internal loading and stiffness-like resistance
without requiring the same net bending moment, while the shorter group
concentrates its control authority in the proximal half of the arm.

## Action space

The action is an eight-value `Box(-1, 1, shape=(8,), dtype=float32)`. The first
four values command the full-length tendons and the remaining four command the
half-length tendons. Each normalized command is mapped linearly to a
nonnegative tension:

```{math}
T_i = \frac{a_i + 1}{2} T_{\max},
```

where the default maximum tension is 8 N. Thus `-1` releases a tendon and `1`
applies its maximum tension.

## Observation space

The observation is a 105-value `float32` vector:

| Values | Description |
| ---: | --- |
| 15 | Five-step arm-tip position history |
| 3 | Target position |
| 3 | Target position relative to the current tip |
| 3 | Current tip velocity |
| 1 | Tip speed |
| 10 | Speeds at ten sampled rod nodes |
| 30 | Positions of those ten nodes |
| 40 | Five-step tendon-tension history |

The history terms expose short-time motion and control changes without requiring
the policy to reconstruct them from a single instantaneous state.

## Target and episode

Unless a fixed target is supplied to the constructor, `reset(seed=...)` samples
a reachable target near the undeformed tip. A target can also be selected for a
single episode:

```python
observation, info = env.reset(
    seed=1,
    options={"target": [0.20, 0.02, -0.03]},
)
```

Each policy action advances 800 simulation substeps by default. A 6 s episode
therefore contains 250 policy steps. Reaching the time limit sets
`truncated=True`. A non-finite simulation state or a control collapse that
moves the tip-target distance beyond the arm length sets `terminated=True`.
The reason is reported as `info["termination_reason"]`.

## Reward

The shaped reward follows the final reward structure in the source project. It
combines:

- a continuous piecewise distance objective;
- progress relative to the best distance achieved in the episode;
- a penalty for tip displacement;
- a tip-speed penalty near the target;
- a tendon-command-rate penalty close to the goal; and
- a bonus for remaining nearly stationary within 2 cm of the target.

Individual terms are available in `info["reward_components"]`, which is useful
for diagnosing policy behavior.

## Usage

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("TendonArmReach-v0")
observation, info = env.reset(seed=1)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Training example

The
[example guide](https://github.com/skim0119/gym-softrobot/tree/main/examples/tendon_arm_reach)
provides a fast environment check and an optional Stable-Baselines3 PPO
baseline:

```bash
uv run --no-sync python examples/tendon_arm_reach/check_env.py --fast
uv run --no-sync python examples/tendon_arm_reach/train_ppo.py \
  --fast --timesteps 4096
```

The upstream project notes that PPO from scratch can converge to degenerate
solutions and uses PID demonstrations, behavioral cloning, and PPO
fine-tuning. The included PPO script is a minimal integration baseline rather
than a reproduction of an upstream trained policy.
