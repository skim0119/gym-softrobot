# Soft-arm environment

## `SoftArmTracking-v0`

- **Task:** follow a moving three-dimensional target with the arm tip.
- **Action:** control points defining muscle-torque profiles along the arm.
- **Observation:** target-relative tip position, sampled arm state, and previous
  activation.
- **Reward:** tracking accuracy, measured from the distance between the tip and
  target.
- **Episode end:** the configured horizon or an invalid simulation state.

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("SoftArmTracking-v0")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```
