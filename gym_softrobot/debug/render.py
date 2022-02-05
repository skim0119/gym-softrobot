import gym 
import gym_softrobot

from gym_softrobot.config import RendererType

#gym_softrobot.RENDERER_CONFIG = RendererType.MATPLOTLIB

def main():
    env = gym.make('OctoArmSingle-v0', recording_fps=30)

    observation = env.reset()
    for step in range(10):
        env.render() # rendering
        action = env.action_space.sample() 
        observation, reward, done, info = env.step(action)
        print(f"{step=:2}| {reward=}, {done=}")
        if done:
            break

if __name__=="__main__":
    main()
        

