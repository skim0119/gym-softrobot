<div align="center">

<h1> Soft-Robot Control Environment (gym-softrobot) </h1>
  <a href="https://github.com/skim0119/gym-softrobot/blob/main/LICENSE"><img src="https://img.shields.io/apm/l/atomic-design-ui.svg?style=flat"></a>
  <a href="https://github.com/skim0119/gym-softrobot"><img src="https://img.shields.io/github/release/skim0119/gym-softrobot.svg?style=flat"></a>
  <img src="https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg?style=flat">
</div>

The environment is designed to leverage wide-range of reinforcement learning methods into soft-robotics control.
Our inspiration is from slender-body living creatures, such as octopus or snake.
The code is based on [PyElastica](https://github.com/GazzolaLab/PyElastica), an open-source physics simulation for highly-deformable slender structure.
We intend this package to be easy-to-install and fully compatible to [OpenAI Gym](https://github.com/openai/gym) and other available RL algorithms.

> The package is under development, in Alpha phase. Detail roadmap for Q2-2022 will be available.

## Installation

```bash
pip install gym-softrobot
```

To test the installation, we provide few debugging scripts.
The environment can be tested using the following command.

```bash
python -m gym_softrobot.debug.make     # Make environment and run 10 steps
python -m gym_softrobot.debug.registry # Print gym-softrobot environment
```

We use [POVray](https://wiki.povray.org/content/HowTo:Install_POV) python wrapper [Vapory](https://github.com/Zulko/vapory) to visualize the motion in 3D.
POVray is not a requirement to run the environment, but it is necessary to use `env.render()` function as typical gym environment.

If you would like to test `POVray` with `gym-softrobot`, use

```bash
python -m gym_softrobot.debug.render  # Render 10 frames using vapory
```

Requirements:
- Python 3.8+
- OpenAI Gym 0.21.0
- PyElastica 0.2+
- Matplotlib (optional for display rendering and plotting)	
- POVray (optional for 3D rendering)


## Reinforcement Learning Example

We tested the environment using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for centralized control.
More advanced algorithms are still under development.

If you have your own algorithm that you would like to test with our environment, you are welcome to reach out to us.

## Citation

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

## Environment Documentation

The description of each environment is available in [documentation](docs/design.md).

## Contribution

We are currently developing the package internally.
We plan to deploy the package to open-development in Q2-2022.

## Author

![GitHub Contributors Image][badge-Contributors-image]

<!-- -->
[badge-CI]: https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg
[badge-Contributors-image]: https://contrib.rocks/image?repo=skim0119/gym-softrobot
