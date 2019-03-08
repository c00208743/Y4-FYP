import gym
from gym import wrappers
from gym import envs
import numpy as np
import matplotlib.pyplot as plt

import os

##env = gym.make("BreakoutNoFrameskip-v4")
##plt.imshow(env.render('rgb_array'))
##plt.grid(False)
##print("observation space:", env.observation_space)
##print("action_space: ", env.action_space)

env = gym.make("MsPacmanNoFrameskip-v4")
monitor_dir = os.getcwd()

record_video = True
should_record = lambda i: record_video
env = wrappers.Monitor(env, monitor_dir, video_callable = should_record, force=True)

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)

record_video = False
env.close()

os.chdir(monitor_dir)
