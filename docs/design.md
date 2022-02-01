<div align="center">
<h1> Environment Documentation </h1>
</div>

Our environment contains set of controllable slender-bodies to achieve set of tasks.
The theory and details of the physics simulation is documented in [CosseratRods.org](https://cosseratrods.org).

### State

The state information may or may-not reflect the entirety of the system: partially observable.
The state space is composed with the following parameters:

- __Position__ (relative) and __director__ at each element (spatial discretization in simulation)
    - Could be supplemented with __velocity/acceleration__ and __angular velocity/acceleration__
- __6-modes of strains__ (normal/binormal shear, normal/binormal curvature, stretch, twist)
- __Absolute position__
- __Target location__ and its velocity (if applicable)
- Index of the agent (multi-agent case)
- Previous action


### Action 

- __Internal curvature__ resembling tendon-driven actuation or muscle actuation.
    - The number of DoFs (degrees of freedoms) along the arm depends on the length of the arm, and may vary depending on the environment and its version. Actuation functions equals to the interpolation of provided DoFs.
- Internal torque/force for direct activation

### Reward

> The reward function is not yet finalized. Different version may contain different reward function.

The reward is defined as the composition of the following quantities:
- __Forward Reward__: typically used in locomotion case
    - Velocity of the body
    - Position difference between control-steps
- __Survive Reward__: given for stability purpose and to prevant wild policy
    - Nan panelty: unstable or unexpected behavior
    - Cross-over panelty: panelty for multiple arm crossing eachother (in 2D)
- __Control Panelty__: minimum-actuation solution
    - Square-average of the action
- __Energy__: minimum-energy solution
    - Total bending energy
    - Total shear energy
- __Time Limit__:
    - Small constant panelty at each steps
    - Large constant panelty for not achieving goal.
- __Miscellaneous__:
    - Target reaching/grabbing reward
    - Remaining distance to the target


## Registered Environments

### Octopus[Multi-arm control]

- `OctoArmSingle-v0` [Alpha]

This environment is the testcase for simplest one-arm control.

- `OctoFlat-v0` [Alpha]

The goal of this environment is to control 8 arms of the octopus attached to the body, and move towards the targeted location.

- `OctoReach-v0` [Working in Process]
- `OctoSwim-v0` [Working in Process]
- `OctoHunt-v0` [Working in Process]

### Snake

- `ContinuumSnake-v0` [Alpha]

This environment is inspired from the [Continuum Snake case](https://github.com/GazzolaLab/PyElastica/tree/master/examples/ContinuumSnakeCase) in PyElastica.
The goal is to control the snake to achieve fastest velocity. 
Unlike the original example, where the control is defined by the amplitude and phase-shift of the sinusoidal activation, our environment challenges the player to give an action every `dt` time-steps.


