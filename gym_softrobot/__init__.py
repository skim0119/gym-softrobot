from gym.envs.registration import register

from gym_softrobot.config import *

""" Octopus Environment """
register(
    id='OctoFlat-v0',
    entry_point='gym_softrobot.envs.octopus:FlatEnv',
)

register(
    id='OctoFlatLite-v0',
    entry_point='gym_softrobot.envs.octopus:FlatEnv',
    kwargs=dict(n_arm=1, n_action=8),
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

register(
    id='OctoArmPush-v0',
    entry_point='gym_softrobot.envs.octopus:ArmPushEnv',
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

#""" Soft Arm Environment """
register(
    id='SoftArmTracking-v0',
    entry_point='gym_softrobot.envs.soft_arm:SoftArmTrackingEnv',
)
#register(
#    id='SoftArmTracking-v1',
#    entry_point='gym_softrobot.envs.soft_arm:SoftArmTrackingEnv',
#    game_mode=2
#)


""" Global Configuration Parameters """
RENDERER_CONFIG = RendererType.POVRAY
