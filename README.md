<div align="center">
<h1> Soft-robot Control Environment (gym-softrobot) </h1>
</div>

The environment is designed to leverage reinforcement learning methods into soft-robotics control, inspired from slender-body living creatures.
The code is built on [PyElastica](https://github.com/GazzolaLab/PyElastica), an open-source physics simulation for slender structure.
We intend this package to be easy-to-install and fully compatible to [OpenAI Gym](https://github.com/openai/gym).

Requirements:
- Python 3.8+
- OpenAI Gym
- PyElastica 0.2+
- Matplotlib (optional for display rendering and plotting)

Please use this bibtex to cite in your publications:

```
@misc{gym_softrobot,
  author = {Chia-Hsien Shih, Seung Hyun Kim, Mattia Gazzola},
  title = {Soft Robotics Environment for OpenAI Gym},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/skim0119/gym-softrobot}},
}
```

## Installation

```
pip install gym-softrobot
```

## Reinforcement Learning Example

We tested the environment using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for centralized control.
More advanced algorithms are still under development.

## Environment Design

## Included Environments

### Octopus[Multi-arm control]

- `octo-flat` [2D]
- `octo-reach`
- `octo-swim`
- `octo-flat`

## Contribution

We are currently developing the package internally.
