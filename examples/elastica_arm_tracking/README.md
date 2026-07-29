# Elastica RL benchmarks

This example trains any of the four benchmark cases ported from
[Elastica-RL-control](https://github.com/GazzolaLab/Elastica-RL-control) to
Gymnasium and current PyElastica:

| Environment | Upstream case | Task |
| --- | --- | --- |
| `ElasticaArmTracking-v0` | Case 1 | Track a randomly moving 3-D target |
| `ElasticaArmReach-v0` | Case 2 | Reach a target and match orientation |
| `ElasticaArmObstacle-v0` | Case 3 | Planar reaching through fixed obstacles |
| `ElasticaArmObstacleRandom-v0` | Case 4 | 3-D reaching through a random obstacle nest |

Stable-Baselines3 is an optional dependency:

```bash
uv pip install stable-baselines3
uv run python examples/elastica_arm_tracking/train_ppo.py \
  --env-id ElasticaArmReach-v0 --total-timesteps 1000000 --seed 0
```

Use a smaller value such as `--total-timesteps 1000` for an integration check.
Checkpoints and TensorBoard logs are written to `save/elastica_arm` by default.

These environments are ports, not byte-for-byte copies. They retain the
published task geometry, B-spline muscle-torque controls, distance/orientation
objectives, and obstacle contacts while using Gymnasium termination semantics,
current PyElastica APIs, and `self.np_random` for reproducibility.
