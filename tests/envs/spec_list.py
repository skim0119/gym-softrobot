from gym import envs, logger
import os

spec_list = [
    spec for spec in sorted(envs.registry.all(), key=lambda x:x.id)
    if "gym_softrobot" in spec.entry_point
]

