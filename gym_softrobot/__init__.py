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
    id="OctoPhaseCrawl-v0",
    entry_point="gym_softrobot.envs.octopus:OctoPhaseCrawlEnv",
)

register(
    id="OctoMuscleCrawl-v0",
    entry_point="gym_softrobot.envs.octopus:OctoMuscleCrawlEnv",
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


register(
    id="TendonArmReach-v0",
    entry_point="gym_softrobot.envs.tendon_arm:TendonArmReachEnv",
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
register(
    id="ElasticaArmTracking-v0",
    entry_point="gym_softrobot.envs.soft_arm:SoftArmTrackingEnv",
    kwargs={"game_mode": 2, "number_of_control_points": 6},
)
register(
    id="ElasticaArmReach-v0",
    entry_point="gym_softrobot.envs.soft_arm:ElasticaArmReachEnv",
)
register(
    id="ElasticaArmObstacle-v0",
    entry_point="gym_softrobot.envs.soft_arm:ElasticaArmStructuredObstacleEnv",
)
register(
    id="ElasticaArmObstacleRandom-v0",
    entry_point="gym_softrobot.envs.soft_arm:ElasticaArmRandomObstacleEnv",
)

""" Soft Pendulum Environment """
register(
    id="SoftPendulum-v0", entry_point="gym_softrobot.envs.soft_pendulum:SoftPendulumEnv"
)
register(
    id="SoftPendulum3D-v0",
    entry_point="gym_softrobot.envs.soft_pendulum_3d:SoftPendulum3DEnv",
)

""" Global Configuration Parameters """
RENDERER_CONFIG = RendererType.POVRAY
