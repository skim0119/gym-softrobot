import sys
sys.path.append("..")

import gym 
import gym_softrobot
env = gym.make('OctoFlat-v0', policy_mode="decentralized")

# env is created, now we can use it: 
for episode in range(10): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample() 
        observation, reward, done, info = env.step(action)
        print(f"{episode=:2} |{step=:2}, {reward=}, {done=}")
        if done:
            break

