import gym 
import gym_softrobot

import argparse

def main():
    parser = argparse.ArgumentParser(description='Make registered environment and test run.')
    parser.add_argument('--env', type=str, default='OctoFlat-v0')
    args = parser.parse_args()

    # env is created, now we can use it: 
    env = gym.make(args.env)

    for episode in range(10): 
        observation = env.reset()
        for step in range(50):
            action = env.action_space.sample() 
            observation, reward, done, info = env.step(action)
            print(f"{episode=:2} |{step=:2}, {reward=}, {done=}")
            if done:
                break

if __name__=="__main__":
    main()
