# Getting started

## Installation

gym-softrobot supports Python 3.12 and newer.

```bash
pip install gym-softrobot
```

For development, clone the repository and create the complete environment with
`uv`:

```bash
uv sync --all-groups
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

Available modes differ by environment. Interactive and POV-Ray rendering may
require graphics or system packages beyond the core Python installation.

## Debug commands

```bash
# Show registered gym-softrobot environments.
python -m gym_softrobot.debug.registry

# Run a short rollout.
python -m gym_softrobot.debug.make --env SoftPendulum-v0

# Run a short rendered rollout.
python -m gym_softrobot.debug.render --env OctoArmSingle-v0
```
