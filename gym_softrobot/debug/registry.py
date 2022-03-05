import gym
import gym_softrobot
from gym import envs

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main():
    spec_list = [
        spec for spec in sorted(envs.registry.all(), key=lambda x:x.id)
        if "gym_softrobot" in spec.entry_point
    ]
    for spec in spec_list:
        print(f"{bcolors.BOLD}{spec.entry_point}{bcolors.ENDC}")
        print(spec)

if __name__=="__main__":
    main()
