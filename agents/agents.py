from stable_baselines3 import PPO, A2C
from stable_baselines3.ddpg import DDPG 

class ContinuousAgents:
    @staticmethod
    def PPO(env, tensorboard_log=None):
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.9,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log=tensorboard_log,
        )

    @staticmethod
    def DDPG(env, tensorboard_log=None):
        return DDPG(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            action_noise=None,
            tensorboard_log=tensorboard_log,
        )

    @staticmethod
    def A2C(env, tensorboard_log=None):
        return A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            tensorboard_log=tensorboard_log,
        )
