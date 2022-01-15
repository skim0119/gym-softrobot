import sys
sys.path.append("..")

import gym 
import gym_softrobot
env = gym.make('OctoFlat-v0')

# env is created, now we can use it: 
observation = env.reset()
for step in range(50):
    env.render()
    action = env.action_space.sample() 
    observation, reward, done, info = env.step(action)
    print(f"{episode=:2} |{step=:2}, {reward=}, {done=}")
    if done:
        break

        

