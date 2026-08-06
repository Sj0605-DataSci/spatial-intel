"""Scripted / model-free agents used to validate the env and to set baselines.

These let us answer "how many iterations does an agent need" *before* paying
for FLUX inference:

- OracleAgent: perfect spatial reasoning. Parses the instruction (direction +
  steps) and applies it exactly -> solves in 1 step. Sanity check that the env
  and reward are self-consistent.
- NoisyAgent: applies the move but with random over/under-shoot. It *reads the
  feedback* each step and corrects, so it converges in a few iterations — a
  stand-in for an imperfect-but-improving model, to exercise the reloop.
- RandomAgent: random on-board placement. Floor baseline.

All of them work in either action mode. In image mode they render the board
themselves via the shared renderer; in cell mode they return [col, row].
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..grid import (
    GridSpec, DIRECTION_NAMES, DIRECTIONS, apply_move, clamp_topleft,
    render_state, to_array, detect_object_topleft,
)
from .base import Agent


def _emit(spec: GridSpec, topleft, action_mode: str):
    topleft = clamp_topleft(spec, topleft)
    if action_mode == "cell":
        return np.array(topleft, dtype=np.int64)
    return to_array(render_state(spec, topleft))


def _direction_from_info(info: dict) -> str:
    return info["direction"]


class OracleAgent(Agent):
    name = "oracle"

    def __init__(self, spec: GridSpec, action_mode: str = "image"):
        self.spec = spec
        self.action_mode = action_mode

    def act(self, obs: dict, info: dict) -> np.ndarray:
        start = info["start_topleft"]
        direction = info["direction"]
        steps = info["steps"]
        target = clamp_topleft(self.spec, apply_move(self.spec, start, direction, steps))
        return _emit(self.spec, target, self.action_mode)


class NoisyAgent(Agent):
    """Imperfect mover that uses feedback to converge."""
    name = "noisy"

    def __init__(self, spec: GridSpec, action_mode: str = "image",
                 noise: int = 2, seed: int = 0):
        self.spec = spec
        self.action_mode = action_mode
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._guess: Tuple[int, int] | None = None

    def reset(self) -> None:
        self._guess = None

    def act(self, obs: dict, info: dict) -> np.ndarray:
        spec = self.spec
        if self._guess is None:
            # first attempt: apply the command but with random over/under-shoot
            start = info["start_topleft"]
            direction = info["direction"]
            steps = info["steps"]
            err = int(self.rng.integers(-self.noise, self.noise + 1))
            noisy_steps = max(0, steps + err)
            self._guess = clamp_topleft(
                spec, apply_move(spec, start, direction, noisy_steps))
        else:
            # subsequent attempts: read the feedback delta and step toward target.
            target = info["target_topleft"]
            gx, gy = self._guess
            sx = int(np.sign(target[0] - gx))
            sy = int(np.sign(target[1] - gy))
            self._guess = clamp_topleft(spec, (gx + sx, gy + sy))
        return _emit(spec, self._guess, self.action_mode)


class RandomAgent(Agent):
    name = "random"

    def __init__(self, spec: GridSpec, action_mode: str = "image", seed: int = 0):
        self.spec = spec
        self.action_mode = action_mode
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict, info: dict) -> np.ndarray:
        m = self.spec.max_topleft
        topleft = (int(self.rng.integers(0, m + 1)), int(self.rng.integers(0, m + 1)))
        return _emit(self.spec, topleft, self.action_mode)
