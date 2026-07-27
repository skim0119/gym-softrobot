# Soft-arm environment

`SoftArmTracking-v0` controls a muscle-actuated arm whose tip follows a target.
The observation describes the arm and target, while the action controls muscle
activation profiles.

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

The historical `SoftArmTrackingEnv-v0` name is not registered.
