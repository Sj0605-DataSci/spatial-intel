"""A Gymnasium environment for the spatial-move task.

Episode
-------
- reset(): sample a (start, direction, steps) command. The observation is the
  rendered *input* board (triangle at the start) plus the encoded instruction.
- step(action): the agent submits its attempt for where the triangle should be.

Two action modes (set via `action_mode`):
- "image": action is an HxWx3 uint8 image (what FLUX produces). The env reads
  the object position out of it and scores against the target.
- "cell" : action is the predicted target top-left `[col, row]` (MultiDiscrete).
  Useful for fast, model-free baselines and for validating the env / reward.

The episode is a *feedback reloop*: on a wrong attempt the env keeps the same
command, annotates the observation with a hint arrow pointing from the last
detected position toward the target, and puts a textual critique in `info`.
The number of steps to `success` is the headline metric: "how many iterations
does the agent need". done when success or `max_steps` reached.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "gymnasium is required: uv pip install gymnasium"
    ) from e

from .grid import (
    GridSpec, DIRECTION_NAMES, DIRECTION_ID, apply_move, clamp_topleft,
    max_steps_for, render_state, to_array, detect_object_topleft, DIRECTIONS,
    normalize_direction,
)
from .dataset import instruction_text
from .reward import RewardConfig, score_attempt


class SpatialMoveEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, spec: Optional[GridSpec] = None, *,
                 action_mode: str = "image", max_steps: int = 6,
                 center_start: bool = False,
                 reward_cfg: Optional[RewardConfig] = None,
                 seed: Optional[int] = None):
        super().__init__()
        assert action_mode in ("image", "cell")
        self.spec = spec or GridSpec()
        self.action_mode = action_mode
        self.max_steps = max_steps
        self.center_start = center_start
        self.reward_cfg = reward_cfg or RewardConfig()
        self._rng = np.random.default_rng(seed)

        h, w = self.spec.img_h, self.spec.img_w
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
            "direction": spaces.Discrete(len(DIRECTION_NAMES)),
            "steps": spaces.Discrete(self.spec.n + 1),
        })
        if action_mode == "image":
            self.action_space = spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)
        else:  # cell
            m = self.spec.max_topleft + 1
            self.action_space = spaces.MultiDiscrete([m, m])

        # episode state
        self.start: Tuple[int, int] = (0, 0)
        self.direction: str = "right"
        self.steps_cmd: int = 1
        self.target: Tuple[int, int] = (0, 0)
        self.t: int = 0
        self.last_detected: Optional[Tuple[int, int]] = None

    # -- sampling --------------------------------------------------------
    def _sample_command(self):
        spec = self.spec
        center = (spec.max_topleft // 2, spec.max_topleft // 2)
        for _ in range(200):
            start = center if self.center_start else (
                int(self._rng.integers(0, spec.max_topleft + 1)),
                int(self._rng.integers(0, spec.max_topleft + 1)),
            )
            direction = DIRECTION_NAMES[int(self._rng.integers(len(DIRECTION_NAMES)))]
            max_s = max_steps_for(spec, start, direction)
            if max_s < 1:
                continue
            steps = int(self._rng.integers(1, max_s + 1))
            target = clamp_topleft(spec, apply_move(spec, start, direction, steps))
            return start, direction, steps, target
        start = (spec.max_topleft // 2, spec.max_topleft // 2)
        return start, "right", 1, clamp_topleft(spec, apply_move(spec, start, "right", 1))

    # -- observation -----------------------------------------------------
    def _obs(self, *, feedback: bool = False) -> dict:
        arrow = None
        if feedback and self.last_detected is not None:
            arrow = (self.last_detected, self.target)
        img = render_state(self.spec, self.start, feedback_arrow=arrow,
                           caption=self.instruction)
        return {
            "image": to_array(img),
            "direction": DIRECTION_ID[self.direction],
            "steps": self.steps_cmd,
        }

    @property
    def instruction(self) -> str:
        return instruction_text(self.direction, self.steps_cmd)

    def _info(self, score=None, feedback: Optional[str] = None) -> dict:
        info = {
            "instruction": self.instruction,
            "start_topleft": self.start,
            "target_topleft": self.target,
            "direction": self.direction,
            "steps": self.steps_cmd,
            "t": self.t,
        }
        if score is not None:
            info["score"] = score.as_dict()
        if feedback is not None:
            info["feedback"] = feedback
        return info

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.start, self.direction, self.steps_cmd, self.target = self._sample_command()
        self.t = 0
        self.last_detected = None
        return self._obs(), self._info()

    def _coerce_attempt_image(self, action) -> np.ndarray:
        if self.action_mode == "image":
            return np.asarray(action, dtype=np.uint8)
        # cell mode: render the claimed target so we can reuse the same scorer
        topleft = clamp_topleft(self.spec, (int(action[0]), int(action[1])))
        return to_array(render_state(self.spec, topleft))

    def step(self, action):
        self.t += 1
        attempt = self._coerce_attempt_image(action)
        score = score_attempt(
            self.spec, attempt,
            start_topleft=self.start, target_topleft=self.target,
            direction=self.direction, cfg=self.reward_cfg,
        )
        self.last_detected = score.detected_topleft

        feedback = self._make_feedback(score)
        terminated = bool(score.success)
        truncated = bool(self.t >= self.max_steps and not terminated)
        obs = self._obs(feedback=not terminated)
        return obs, float(score.reward), terminated, truncated, self._info(score, feedback)

    def _make_feedback(self, score) -> str:
        if score.success:
            return "Correct — the triangle is on the target block."
        if score.detected_topleft is None:
            return ("Could not locate the red triangle in your image. "
                    "Render a single solid red triangle aligned to one grid block.")
        dx = self.target[0] - score.detected_topleft[0]
        dy = self.target[1] - score.detected_topleft[1]
        parts = []
        if dy < 0:
            parts.append(f"move it UP by {-dy} cell(s)")
        elif dy > 0:
            parts.append(f"move it DOWN by {dy} cell(s)")
        if dx < 0:
            parts.append(f"move it LEFT by {-dx} cell(s)")
        elif dx > 0:
            parts.append(f"move it RIGHT by {dx} cell(s)")
        where = (f"It is at block {score.detected_topleft}, target is "
                 f"{self.target}.")
        return where + " Adjust: " + ", ".join(parts) + "." if parts else where

    def render(self):
        return self._obs()["image"]
