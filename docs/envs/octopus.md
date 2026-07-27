# Octopus environments

The octopus environments cover individual-arm control, multi-arm locomotion,
reaching, pushing, and crawling.

| Environment ID | Task |
| --- | --- |
| `OctoArmSingle-v0` | Control one soft arm |
| `OctoArmTwo-v0` | Coordinate two arms |
| `OctoArmPush-v0` | Push with a discrete action |
| `OctoArmPush-v1` | Push with continuous actions |
| `OctoArmPullWeight-v0` | Pull an attached weight |
| `OctoFlat-v0` | Control an eight-arm planar octopus |
| `OctoFlatLite-v0` | Reduced planar configuration |
| `OctoCrawl-v0` | Crawl using coordinated arms |
| `OctoReach-v0` | Reach a target |

These environments are computationally intensive. Start with a short rollout
and inspect `action_space` and `observation_space` before configuring a learning
algorithm.

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("OctoArmSingle-v0")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```
