import sys
sys.path.append("..")

import gym 
import gym_softrobot
env = gym.make('OctoFlat-v0', recording_fps=30)

# env is created, now we can use it: 
observation = env.reset()
for step in range(50):
    env.render()
    action = env.action_space.sample()  / 10
    observation, reward, done, info = env.step(action)
    print(f"{step=:2}| {reward=}, {done=}")
    if done:
        break

        

