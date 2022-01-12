import gym 
import gym_softrobot
env = gym.make('OctoFlat-v0')

# env is created, now we can use it: 
for episode in range(10): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)

