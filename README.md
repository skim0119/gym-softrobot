<div align="center">
<h1> Soft-Robot Control Environment (gym-softrobot) </h1>
  <img src="https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg">
</div>


The environment is designed to leverage reinforcement learning methods into soft-robotics control, inspired from slender-body living creatures.
The code is built on [PyElastica](https://github.com/GazzolaLab/PyElastica), an open-source physics simulation for slender structure.
We intend this package to be easy-to-install and fully compatible to [OpenAI Gym](https://github.com/openai/gym).

Requirements:
- Python 3.8+
- OpenAI Gym 0.21.0
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

```bash
pip install gym-softrobot
```

To test the installation, you can run couple steps of the environment as the following.
```py
import gym 
import gym_softrobot
env = gym.make('OctoFlat-v0', policy_mode='centralized')

# env is created, now we can use it: 
for episode in range(2): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample() 
        observation, reward, done, info = env.step(action)
        print(f"{episode=:2} |{step=:2}, {reward=}, {done=}")
        if done:
            break
```

## Reinforcement Learning Example

We tested the environment using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for centralized control.
More advanced algorithms are still under development.

## Environment Design

## Included Environments

### Octopus[Multi-arm control]

- `OctoFlat-v0`
- `OctoReach-v0`
- `OctoSwim-v0`
- `OctoHunt-v0`

### Snake

- 'ContinuumSnake-v0'

### Simple Control

## Contribution

We are currently developing the package internally.

[badge-CI]: https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg
