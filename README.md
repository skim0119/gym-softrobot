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

## About

Soft robots replace conventional rigid links and localized joints with
continuously deformable bodies. Their compliance can make them safe, adaptable,
and capable of rich motion, but it also makes control and actuator design
difficult: deformation is nonlinear, actuation is often distributed throughout
the body, and an actuator changes the mechanics of the structure that it is
trying to control. Useful control studies therefore need both realistic
continuum dynamics and freedom to ask *how* the robot should be actuated.

gym-softrobot provides Gymnasium environments for this problem, using
[PyElastica](https://github.com/GazzolaLab/PyElastica) for high-fidelity
Cosserat-rod simulation. Rather than committing every task to a single hardware
design, the environments cover several levels of actuator abstraction:

- direct push-pull forces for simple control experiments;
- tendon-driven actuation for continuum arms;
- distributed, muscle-like activation for bio-inspired robots; and
- idealized force and torque controls for studying hypothetical actuator
  placement and authority.

This range makes it possible to separate control questions from actuator-design
questions, then progressively introduce physical constraints. The environments
can be used to investigate policy transfer between actuation models, curriculum
learning from simpler to higher-fidelity tasks, sim-to-real-oriented robustness,
co-design, and reinforcement-learning methods for distributed control.


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

## Reinforcement-learning examples

The four `ElasticaArm*` environments are modern Gymnasium/PyElastica ports of
the tracking, orientation-reaching, structured-obstacle, and
unstructured-obstacle benchmarks from
[Elastica-RL-control](https://github.com/GazzolaLab/Elastica-RL-control).
See the [benchmark documentation](docs/envs/elastica_rl.md) for the mapping to
upstream Cases 1–4 and the [PPO example](examples/elastica_arm_tracking/README.md)
for Stable-Baselines3 training.

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

@article{Naughton2021,
  author = {Naughton, Noel and Sun, Jiarui and Tekinalp, Arman and
            Parthasarathy, Tejaswin and Chowdhary, Girish and Gazzola, Mattia},
  title = {Elastica: A Compliant Mechanics Environment for Soft Robotic Control},
  journal = {IEEE Robotics and Automation Letters},
  year = {2021},
  volume = {6},
  number = {2},
  pages = {3389--3396},
  doi = {10.1109/LRA.2021.3063698}
}
```

## Author

![GitHub Contributors Image][badge-Contributors-image]

<!-- -->
[badge-CI]: https://github.com/skim0119/gym-softrobot/actions/workflows/main.yml/badge.svg
[badge-Contributors-image]: https://contrib.rocks/image?repo=skim0119/gym-softrobot
