from gymnasium.envs.registration import register

from gym_softrobot.config import RendererType

""" Octopus Environment """
register(
    id="OctoFlat-v0",
    entry_point="gym_softrobot.envs.octopus:FlatEnv",
)

register(
    id="OctoFlatLite-v0",
    entry_point="gym_softrobot.envs.octopus:FlatEnv",
    kwargs=dict(n_arm=1, n_action=8),
)

register(
    id="OctoCrawl-v0",
    entry_point="gym_softrobot.envs.octopus:CrawlEnv",
)

register(
    id="OctoReach-v0",
    entry_point="gym_softrobot.envs.octopus:ReachEnv",
)

register(
    id="OctoArmSingle-v0",
    entry_point="gym_softrobot.envs.octopus:ArmSingleEnv",
)

register(
    id="OctoArmTwo-v0",
    entry_point="gym_softrobot.envs.octopus:ArmTwoEnv",
)

register(
    id="OctoArmPush-v0",
    entry_point="gym_softrobot.envs.octopus:ArmPushEnv",
)

register(
    id="OctoArmPush-v1",
    entry_point="gym_softrobot.envs.octopus:ArmPushEnv",
    kwargs=dict(mode="continuous"),
)

register(
    id="OctoArmPullWeight-v0",
    entry_point="gym_softrobot.envs.octopus:ArmPullWeightEnv",
    kwargs=dict(mode="continuous"),
)

""" Snake Environment """
register(
    id="ContinuumSnake-v0",
    entry_point="gym_softrobot.envs.snake:ContinuumSnakeEnv",
)

""" Simple Control Environment """

# """ Soft Arm Environment """
register(
    id="SoftArmTracking-v0",
    entry_point="gym_softrobot.envs.soft_arm:SoftArmTrackingEnv",
)
# register(
#    id='SoftArmTracking-v1',
#    entry_point='gym_softrobot.envs.soft_arm:SoftArmTrackingEnv',
#    kwargs=dict(game_mode=2)
# )

""" Soft Pendulum Environment """
register(
    id="SoftPendulum-v0", entry_point="gym_softrobot.envs.soft_pendulum:SoftPendulumEnv"
)

""" Global Configuration Parameters """
RENDERER_CONFIG = RendererType.POVRAY
