import sys
sys.path.append("..")

import gym 
import gym_softrobot
env = gym.make('OctoFlat-v0')

# env is created, now we can use it: 
observation = env.reset()
print(env.rigid_rod.radius)
print(env.rigid_rod.position_collection[:,0])
print(env.rigid_rod.director_collection[2,:,0])
print(env.rigid_rod.length)
sys.exit()
for step in range(50):
    env.render()
    action = env.action_space.sample() 
    observation, reward, done, info = env.step(action)
    print(f"{episode=:2} |{step=:2}, {reward=}, {done=}")
    if done:
        break

        

