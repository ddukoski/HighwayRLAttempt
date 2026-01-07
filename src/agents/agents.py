from __future__ import annotations
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

class BaseHighwayAgents:
    """
    Continuous agents for highway-env.
    Tuned for:
        - Highway
        - Parking: SAC + HER (goal-conditioned)
    """

    @staticmethod
    def make_dqn_agent(env: gym.Env, tensorboard_log=None, **kwargs) -> DQN:
        """
        Creates a DQN agent for discrete action environments (e.g., highway).
        kwargs can include policy_kwargs, learning_rate, buffer_size, gamma, etc.
        """
        return DQN(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            tensorboard_log=tensorboard_log,
            **kwargs
        )

    @staticmethod
    def make_her_sac_agent(env: gym.Env, tensorboard_log=None, n_sampled_goal=4, goal_selection_strategy='future', **kwargs) -> SAC:
        """
        Creates a HER+SAC agent for continuous action environments (e.g., parking).
        kwargs can include policy_kwargs, learning_rate, buffer_size, gamma, etc.
        """
        her_kwargs = dict(n_sampled_goal=n_sampled_goal, goal_selection_strategy=goal_selection_strategy)
        return SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            verbose=0,
            tensorboard_log=tensorboard_log,
            **kwargs
        )
