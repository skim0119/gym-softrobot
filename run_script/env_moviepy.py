import sys
sys.path.append("..")

import gym
import gym_softrobot

from moviepy.editor import VideoClip

fps = 30
env = gym.make('OctoFlat-v0', recording_fps=fps)

# env is created, now we can use it: 
observation = env.reset()
def make_frame(step):
    frame = env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"{step=:2}| {reward=}, {done=}")
    if done:
        return np.zeros_like(frame)
    return frame


clip = VideoClip(make_frame, duration=2)
clip.write_videofile("save.mp4", fps=fps)
