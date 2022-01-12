import sys
sys.path.append("..")

import gym
import gym_softrobot
from gym import envs

spec_list = [
    spec for spec in sorted(envs.registry.all(), key=lambda x:x.id)
    if "gym_softrobot" in spec.entry_point
]

print(spec_list)
