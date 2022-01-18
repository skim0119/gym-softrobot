from gym.envs.registration import register

""" Octopus Environment """
register(
    id='OctoFlat-v0',
    entry_point='gym_softrobot.envs.octopus:FlatEnv',
)

#register(
#    id='OctoReach-v0',
#    entry_point='gym_softrobot.envs.octopus:ReachEnv',
#)

#register(
#    id='OctoSwim-v0',
#    entry_point='gym_softrobot.envs.octopus:SwimEnv',
#)

#register(
#    id='OctoHunt-v0',
#    entry_point='gym_softrobot.envs.octopus:HuntEnv',
#)

register(
    id='OctoArmSingle-v0',
    entry_point='gym_softrobot.envs.octopus:ArmSingleEnv',
)

""" Snake Environment """
register(
    id='ContinuumSnake-v0',
    entry_point='gym_softrobot.envs.snake:ContinuumSnakeEnv',
)

""" Simple Control Environment """
#register(
#    id='InertialPull-v0',
#    entry_point='gym_softrobot.envs.simple_control:InertialPullEnv',
#)
