import gym
import gym_softrobot
from gym import envs

def main():
    spec_list = [
        spec for spec in sorted(envs.registry.all(), key=lambda x:x.id)
        if "gym_softrobot" in spec.entry_point
    ]
    print(spec_list)

if __name__=="__main__":
    main()
