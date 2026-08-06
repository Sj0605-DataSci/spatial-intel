"""Agent interface.

An agent observes the board + instruction and returns an *action* in the
env's action space:
  - image mode -> an HxWx3 uint8 image (its attempt at the target board)
  - cell  mode -> a [col, row] target top-left

`act` receives the previous `info` dict too, which carries the textual
`feedback` from the last attempt — this is how the reloop incorporates
feedback.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Agent:
    name = "agent"

    def reset(self) -> None:
        """Called at the start of each episode."""

    def act(self, obs: dict, info: dict) -> np.ndarray:
        raise NotImplementedError
