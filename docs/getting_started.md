# Getting Started

```{tip}
`SoftPendulum-v0` is the cheapest environment for a first `reset`/`step` loop.
Use a CyberOctopus ID only after that loop works.
```

## Run an environment

Importing `gym_softrobot` registers its environment IDs with Gymnasium.

```python
import gymnasium as gym
import gym_softrobot

env = gym.make("SoftPendulum-v0")
observation, info = env.reset(seed=42)

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

See [Environments](envs/index.md) for the available tasks and their control
interfaces.

## Rendering

Choose a render mode when constructing an environment:

```python
env = gym.make("OctoArmSingle-v0", render_mode="human")
observation, info = env.reset()
env.render()
env.close()
```

```{note}
Available modes differ by environment and are listed in
`env.metadata["render_modes"]`. Interactive and POV-Ray rendering may require
graphics or system packages beyond the core Python installation.
```

## Debug commands

```bash
# Show registered gym-softrobot environments.
python -m gym_softrobot.debug.registry

# Run a short rollout.
python -m gym_softrobot.debug.make --env SoftPendulum-v0

# Run a short rendered rollout.
python -m gym_softrobot.debug.render --env OctoArmSingle-v0
```
