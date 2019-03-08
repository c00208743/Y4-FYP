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

env = gym.make("BreakoutDeterministic-v4")

frame = env.reset()

env.render()

is_done = False
while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    env.render()
