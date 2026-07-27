<div align="center">

<h1> Soft-Robot Control Environment (gym-softrobot) </h1>
  <a href="https://github.com/skim0119/gym-softrobot/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/MIT-License-Green"></a>
  <a href="https://github.com/skim0119/gym-softrobot"><img src="https://img.shields.io/github/release/skim0119/gym-softrobot.svg?style=flat"></a>
    <img src="https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg?style=flat">
  <a href='https://gym-softrobot.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/gym-softrobot/badge/?version=latest' alt='Documentation Status' />
</a>
</div>

Gymnasium environments for soft-robot control, powered by
[PyElastica](https://github.com/GazzolaLab/PyElastica). The package includes
bio-inspired soft-slender-robot simulations.


## Installation

gym-softrobot requires Python 3.12 or newer.

```bash
pip install gym-softrobot
```

## Quick start

```python
import gymnasium as gym
import gym_softrobot  # Registers the environments.

env = gym.make("SoftPendulum-v0")
observation, info = env.reset(seed=42)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

List the registered environments:

```bash
python -m gym_softrobot.debug.registry
```

See the [documentation](https://gym-softrobot.readthedocs.io/) for environment
IDs, API conventions, and rendering notes.

## Development

```bash
uv sync --all-groups
uv run pre-commit install
```

## Citation

```bibtex
@misc{gym_softrobot,
  author = {Chia-Hsien Shih and Seung Hyun Kim and Mattia Gazzola},
  title = {Soft Robotics Environment for Gymnasium},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/skim0119/gym-softrobot}}
}
```

## Author

![GitHub Contributors Image][badge-Contributors-image]

<!-- -->
[badge-CI]: https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg
[badge-Contributors-image]: https://contrib.rocks/image?repo=skim0119/gym-softrobot
