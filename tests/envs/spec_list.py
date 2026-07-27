from gymnasium import envs

import gym_softrobot  # noqa: F401

spec_list = [
    spec
    for spec in sorted(envs.registry.values(), key=lambda item: item.id)
    if isinstance(spec.entry_point, str) and "gym_softrobot" in spec.entry_point
]
