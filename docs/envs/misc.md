# Miscellaneous Environments

## Soft Pendulum [Transfer learning, Soft Arm Control]

- `SoftPendulum-v0`

## Snake

- `ContinuumSnake-v0` [Alpha]

This environment is inspired from the [Continuum Snake case](https://github.com/GazzolaLab/PyElastica/tree/master/examples/ContinuumSnakeCase) in PyElastica.
The goal is to control the snake to achieve fastest velocity. 
Unlike the original example, where the control is defined by the amplitude and phase-shift of the sinusoidal activation, our environment challenges the player to give an action every `dt` time-steps.
