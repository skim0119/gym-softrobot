import gym 
import gym_softrobot

def main():
    # env is created, now we can use it: 
    env = gym.make('OctoFlat-v0', policy_mode="centralized")

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
