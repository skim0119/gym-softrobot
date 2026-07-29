# Elastica RL benchmark suite

The four `ElasticaArm*` environments port the benchmark tasks accompanying
Naughton et al., *Elastica: A Compliant Mechanics Environment for Soft Robotic
Control*, IEEE Robotics and Automation Letters 6(2), 2021,
[doi:10.1109/LRA.2021.3063698](https://doi.org/10.1109/LRA.2021.3063698).
Their source is the
[GazzolaLab/Elastica-RL-control](https://github.com/GazzolaLab/Elastica-RL-control)
supplementary repository.

The upstream code targeted OpenAI Gym, TensorFlow 1, and the original
Stable-Baselines package. These ports use Gymnasium, current PyElastica, and an
optional Stable-Baselines3 PPO example. Seeded randomness is provided through
Gymnasium rather than the global NumPy generator.

## Shared mechanics

All four cases represent the arm as a shearable, extensible Cosserat rod. Each
cross-section carries a position and an orientation, allowing the solver to
capture axial stretch, transverse shear, bending about two material axes, and
twist about the centerline. A clamped boundary condition fixes the base, while
elastic restoring forces, inertia, and numerical damping determine the
distributed transient response.

Actuation is expressed as a torque density along the rod. Policy actions are
control values for a B-spline, which converts a low-dimensional action into a
smooth spatial torque profile. Normal and binormal profiles bend the arm about
its two material axes; a tangent profile twists it. This is an idealized
distributed actuator model: it exposes where torque should be applied without
assuming a particular tendon, pneumatic chamber, or motor transmission.

## Case mapping

### Case 1 — moving-target tracking

`ElasticaArmTracking-v0` follows the upstream
[`Case1/set_environment.py`](https://github.com/GazzolaLab/Elastica-RL-control/blob/main/Case1/set_environment.py).
The arm tracks a procedurally generated target trajectory in three dimensions.
The action contains six B-spline control values in each of the material normal
and binormal bending directions.

The target is kinematic and does not exchange contact forces with the arm. The
control problem is dominated by elastic lag, inertia, and vibration: the policy
must anticipate target motion while avoiding torque profiles that excite large
transient oscillations.

### Case 2 — position and orientation reaching

`ElasticaArmReach-v0` follows
[`Case2/set_environment.py`](https://github.com/GazzolaLab/Elastica-RL-control/blob/main/Case2/set_environment.py).
The action additionally controls tangent torque, allowing the arm to twist.
The reward combines squared tip-to-target distance with the quaternion
orientation error used by the original benchmark.

Matching position alone leaves the tip material frame unconstrained. Tangent
torque supplies torsional strain so the policy can rotate that frame. The
quaternion term measures orientation independently of the sign ambiguity
between $\mathbf{q}$ and $-\mathbf{q}$, while the distance term still
requires the bending deformation needed to place the tip.

### Case 3 — structured obstacles

`ElasticaArmObstacle-v0` follows the main-text variant under
[`Case3`](https://github.com/GazzolaLab/Elastica-RL-control/tree/main/Case3).
It is deliberately underactuated: two B-spline controls bend the arm in one
plane. Eight fixed cylindrical obstacles form a sequence of offset openings,
and PyElastica contact forces make collision avoidance part of the dynamics.

The cylinders are rigid and clamped. Rod-cylinder contact uses a compliant
penalty law: interpenetration produces a restoring normal force and relative
normal motion produces damping. Because only one bending direction is
actuated, the policy cannot simply move out of plane; it must exploit the arm's
distributed curvature and elastic interaction with the obstacles.

### Case 4 — unstructured obstacles

`ElasticaArmObstacleRandom-v0` follows
[`Case4/set_environment.py`](https://github.com/GazzolaLab/Elastica-RL-control/blob/main/Case4/set_environment.py).
Two control points in each bending direction steer the arm through twelve
randomly positioned and oriented cylinders. Calling `reset(seed=...)`
reconstructs the same obstacle nest.

This case restores both bending directions but retains only two spline controls
per direction, so the arm remains strongly underactuated. Oblique contacts
couple bending directions and can redirect or trap the arm. Seeded obstacle
geometry makes it possible to distinguish memorization of a particular nest
from policies that generalize across contact configurations.

## Training

Stable-Baselines3 remains optional. For example:

```bash
uv pip install stable-baselines3
uv run python examples/elastica_arm_tracking/train_ppo.py \
  --env-id ElasticaArmObstacle-v0 \
  --total-timesteps 1000000 \
  --seed 0
```

The full simulations are computationally expensive. Use a small
`--total-timesteps` value only as an integration check, not as a meaningful
training run.
