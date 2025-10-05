import gymnasium as gym
import highway_env
import numpy as np
import environments.environment_utils as eu

env = eu.make_env(eu.EnvRegistry.u_turn)

env.reset()

while(True):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        env.reset()

env.close()
