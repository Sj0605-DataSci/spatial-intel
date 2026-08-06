"""spatial-intel: a synthetic benchmark + Gym env for spatial reasoning in image models."""
from .grid import GridSpec
from .reward import RewardConfig, score_attempt
from .env import SpatialMoveEnv
from .dataset import generate_dataset, make_sample, instruction_text

__all__ = [
    "GridSpec", "RewardConfig", "score_attempt", "SpatialMoveEnv",
    "generate_dataset", "make_sample", "instruction_text",
]
