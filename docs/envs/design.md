# Environment API

gym-softrobot environments follow the Gymnasium API. Each task combines a
PyElastica simulation with an action space, observation space, reward, and
episode-ending conditions.

## Observations

Observations vary by task and may include:

- rod position, orientation, and velocity;
- curvature, shear, stretch, or their rates;
- previous actions;
- target position or velocity;
- shared state for multi-agent tasks.

Consult an environment's `observation_space` at runtime for its exact shape,
bounds, and data type.

## Actions

Actions control quantities such as muscle activation, internal curvature,
torque, or applied force. Exact bounds and dimensions are available from
`action_space`.

```python
action = env.action_space.sample()
```

## Rewards and episode endings

Rewards are task-specific and can combine progress, target distance, control
cost, elastic energy, and simulation-stability terms.

`terminated` reports a task terminal state, such as reaching a target or an
invalid simulation state. `truncated` reports a time limit. Reward definitions
and numerical behavior may change while an environment remains experimental.

For background on the underlying rod model, see
[CosseratRods.org](https://www.cosseratrods.org/).
