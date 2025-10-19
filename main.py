import os
import gymnasium as gym
import highway_env
import numpy as np
from agents.agents import ContinuousAgents
import environments.environment_utils as eu
import pygame
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback


if __name__ == "__main__":

    envstr: str = eu.EnvRegistry.intersection

    model_class = PPO 
    model_name = f"{model_class.__name__}_{envstr}"
    model_path = f"{model_name}.zip"

    train_env = eu.make_env(envstr, render_mode=None)
    eval_env = eu.make_env(envstr, render_mode=None)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    if os.path.exists(model_path):
        print(f"Found existing model: {model_path}, loading...")
        model = model_class.load(model_path)
        model.set_env(train_env)
    else:
        print(f"No model found. Training new model: {model_path}")
        model = ContinuousAgents.PPO(train_env, tensorboard_log="./logs/")
        total_timesteps = int(3e4)
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(model_name)
        print(f"Model successfully saved to: {model_path}")

    render_env = eu.make_env(envstr, render_mode="human")
    obs, info = render_env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = render_env.step(action)
        render_env.render()
        if terminated or truncated:
            obs, info = render_env.reset()
