"""Reward / scoring for a spatial move attempt.

Given the *commanded* move (start position + direction + steps) we know the
intended target cell. An attempt is scored by reading the object position out
of the produced image and comparing it to that target.

Components
----------
- success:          attempt lands exactly on the target block.
- placement reward: shaped, in [0, 1], decays with cell distance to target.
- direction bonus:  did the object move in (roughly) the commanded direction?

The total reward is in [0, 1] so it composes nicely with RL baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .grid import GridSpec, DIRECTIONS, normalize_direction, detect_object_topleft


@dataclass
class RewardConfig:
    success_bonus: float = 1.0      # reward when exactly on target
    placement_weight: float = 0.7   # weight of the shaped distance term
    direction_weight: float = 0.3   # weight of the "moved the right way" term
    detection_penalty: float = 0.0  # reward when the object can't be found


@dataclass
class ScoreResult:
    reward: float
    success: bool
    detected_topleft: Optional[Tuple[int, int]]
    target_topleft: Tuple[int, int]
    cell_distance: float
    direction_correct: bool
    confidence: float

    def as_dict(self) -> dict:
        return {
            "reward": round(self.reward, 4),
            "success": self.success,
            "detected_topleft": self.detected_topleft,
            "target_topleft": self.target_topleft,
            "cell_distance": round(self.cell_distance, 3),
            "direction_correct": self.direction_correct,
            "confidence": round(self.confidence, 5),
        }


def _cosine_dir(vec: Tuple[float, float], direction: str) -> float:
    dc, dr = DIRECTIONS[normalize_direction(direction)]
    v = np.array(vec, dtype=float)
    d = np.array([dc, dr], dtype=float)
    if np.linalg.norm(v) < 1e-6 or np.linalg.norm(d) < 1e-6:
        return 0.0
    return float(np.dot(v, d) / (np.linalg.norm(v) * np.linalg.norm(d)))


def score_attempt(
    spec: GridSpec,
    attempt_img: np.ndarray,
    *,
    start_topleft: Tuple[int, int],
    target_topleft: Tuple[int, int],
    direction: str,
    cfg: RewardConfig = RewardConfig(),
) -> ScoreResult:
    detected, conf = detect_object_topleft(attempt_img, spec)

    if detected is None:
        return ScoreResult(
            reward=cfg.detection_penalty, success=False,
            detected_topleft=None, target_topleft=target_topleft,
            cell_distance=float("inf"), direction_correct=False, confidence=conf,
        )

    dist = float(np.hypot(detected[0] - target_topleft[0],
                          detected[1] - target_topleft[1]))
    success = detected == tuple(target_topleft)

    # shaped placement term: 1 at target, decaying to 0 across the board diagonal
    diag = float(np.hypot(spec.n, spec.n)) or 1.0
    placement = max(0.0, 1.0 - dist / diag)

    # direction term: did we move the right way vs the start?
    move_vec = (detected[0] - start_topleft[0], detected[1] - start_topleft[1])
    cos = _cosine_dir(move_vec, direction)
    direction_correct = cos > 0.5
    direction_term = max(0.0, cos)

    if success:
        reward = cfg.success_bonus
    else:
        reward = (cfg.placement_weight * placement
                  + cfg.direction_weight * direction_term)

    return ScoreResult(
        reward=float(reward), success=success,
        detected_topleft=detected, target_topleft=tuple(target_topleft),
        cell_distance=dist, direction_correct=direction_correct, confidence=conf,
    )
