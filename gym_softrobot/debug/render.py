import gym 
import gym_softrobot

from gym_softrobot.config import RendererType

#gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB

import argparse

def main():
    parser = argparse.ArgumentParser(description='Make registered environment and test run.')
    parser.add_argument('--env', type=str, default='OctoArmSingle-v0')
    args = parser.parse_args()

    env = gym.make(args.env, recording_fps=30)

    observation = env.reset()
    for step in range(10):
        env.render() # rendering
        input('')
        action = env.action_space.sample() 
        observation, reward, done, info = env.step(action)
        print(f"{step=:2}| {reward=}, {done=}")
        if done:
            break

if __name__=="__main__":
    main()
        

