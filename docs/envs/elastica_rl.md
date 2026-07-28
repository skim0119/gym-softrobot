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

## Case mapping

### Case 1 — moving-target tracking

`ElasticaArmTracking-v0` follows the upstream
[`Case1/set_environment.py`](https://github.com/GazzolaLab/Elastica-RL-control/blob/main/Case1/set_environment.py).
The arm tracks a procedurally generated target trajectory in three dimensions.
The action contains six B-spline control values in each of the material normal
and binormal bending directions.

### Case 2 — position and orientation reaching

`ElasticaArmReach-v0` follows
[`Case2/set_environment.py`](https://github.com/GazzolaLab/Elastica-RL-control/blob/main/Case2/set_environment.py).
The action additionally controls tangent torque, allowing the arm to twist.
The reward combines squared tip-to-target distance with the quaternion
orientation error used by the original benchmark.

### Case 3 — structured obstacles

`ElasticaArmObstacle-v0` follows the main-text variant under
[`Case3`](https://github.com/GazzolaLab/Elastica-RL-control/tree/main/Case3).
It is deliberately underactuated: two B-spline controls bend the arm in one
plane. Eight fixed cylindrical obstacles form a sequence of offset openings,
and PyElastica contact forces make collision avoidance part of the dynamics.

### Case 4 — unstructured obstacles

`ElasticaArmObstacleRandom-v0` follows
[`Case4/set_environment.py`](https://github.com/GazzolaLab/Elastica-RL-control/blob/main/Case4/set_environment.py).
Two control points in each bending direction steer the arm through twelve
randomly positioned and oriented cylinders. Calling `reset(seed=...)`
reconstructs the same obstacle nest.

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
