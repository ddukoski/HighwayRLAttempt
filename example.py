import gymnasium as gym
import highway_env
import numpy as np

env = gym.make("highway-v0",render_mode="rgb_array")

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        env.reset()

env.close()
