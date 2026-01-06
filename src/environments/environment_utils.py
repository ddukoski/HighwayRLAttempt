import gymnasium as gym
import highway_env
import numpy as np

from gymnasium import spaces
from typing import Dict, Any

class EnvRegistry:
    highway = "highway-v0"
    highway_fast = "highway-fast-v0"
    merge = "merge-v0"
    intersection = "intersection-v0"
    intersection_continuous = "intersection-v1"
    intersection_multi = "intersection-multi-agent-v0"
    lane_keeping = "lane-keeping-v0"
    parking = "parking-v0"
    racetrack = "racetrack-v0"
    racetrack_large = "racetrack-large-v0"
    roundabout = "roundabout-v0"
    two_way = "two-way-v0"
    u_turn = "u-turn-v0"
    exit = "exit-v0"

    @classmethod
    def all(cls) -> list[str]:
        """Return a list of all registered environment names."""
        return [v for k, v in cls.__dict__.items() if not k.startswith("__") and not callable(v)]

def make_env(env_id: str, config: Dict[str, Any] = None, render_mode: str = None) -> gym.Env:
    base_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h", "heading"],
            "normalize": True,
        },
        "reward_speed_range": [10, 20],
        "simulation_frequency": 10,
        "policy_frequency": 1,
        "initial_speed": 10,
        "action": {
            "type": "ContinuousAction",
        },
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 800,
        "screen_height": 600,
    }

    if config is not None:
        base_config.update(config)

    if env_id in [EnvRegistry.highway, EnvRegistry.highway_fast, EnvRegistry.merge]:
        base_config["action"] = {"type": "DiscreteMetaAction"}

    
    
    
    if env_id == EnvRegistry.parking:
        base_config["observation"] = {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }

    env = gym.make(env_id, render_mode=render_mode, config=base_config)

    if env_id == EnvRegistry.parking:
        if not isinstance(env.observation_space, spaces.Dict):
            class GoalDictWrapper(gym.Wrapper):
                def __init__(self, env):
                    super().__init__(env)
                    
                    sample_obs, _ = env.reset()
                    obs_arr = np.array(sample_obs)
                    obs_shape = obs_arr.shape
                    obs_dtype = obs_arr.dtype if hasattr(obs_arr, 'dtype') else np.float32
                    self.observation_space = spaces.Dict({
                        'observation': spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=obs_dtype),
                        'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=obs_dtype),
                        'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=obs_dtype),
                    })

                def _to_goal_dict(self, observation):
                    obs_clean = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
                    obs_clean = obs_clean.astype(self.observation_space['observation'].dtype)
                    
                    desired = None
                    if hasattr(self.env.unwrapped, 'goal'):
                        desired = getattr(self.env.unwrapped, 'goal')
                    elif hasattr(self.env.unwrapped, 'desired_goal'):
                        desired = getattr(self.env.unwrapped, 'desired_goal')
                    if desired is None:
                        desired = np.zeros(obs_clean.shape, dtype=self.observation_space['desired_goal'].dtype)
                    
                    achieved = obs_clean.copy()
                    return {'observation': obs_clean, 'achieved_goal': achieved, 'desired_goal': desired}

                def reset(self, **kwargs):
                    obs, info = self.env.reset(**kwargs)
                    return self._to_goal_dict(obs), info

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    return self._to_goal_dict(obs), reward, terminated, truncated, info

            env = GoalDictWrapper(env)

    return env

