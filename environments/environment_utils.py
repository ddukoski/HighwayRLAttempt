import gymnasium as gym
import highway_env
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


def make_env(env_id: str, config: Dict[str, Any] = None, render_mode: str = 'human') -> gym.Env:
    """
    Create a Gym environment by ID.

    Args:
        env_id: Full Gym environment ID (e.g., "highway-v0")
        config: Optional dict to override default configuration
        render_mode: "human", "rgb_array", etc.

    Returns:
        A gym environment instance.
    """

    config = {
        "screen_width": 800,
        "screen_height": 600,
    }

    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode, config=config)
    else:
        env = gym.make(env_id)

    return env
