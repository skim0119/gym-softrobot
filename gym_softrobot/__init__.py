from gym.envs.registration import register

register(
    id='OctoFlat-v0',
    entry_point='gym_softrobot.envs.octopus:FlatEnv',
)

register(
    id='OctoReach-v0',
    entry_point='gym_softrobot.envs.octopus:ReachEnv',
)

register(
    id='OctoSwim-v0',
    entry_point='gym_softrobot.envs.octopus:SwimEnv',
)

register(
    id='OctoHunt-v0',
    entry_point='gym_softrobot.envs.octopus:HuntEnv',
)
