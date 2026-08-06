"""Gymnasium env for "Santa's Gift Journey" (game_ref RDDL game), for DIAMOND.

Object positions were VERIFIED against `scene.png` directly (color-blob
detection + visual grid-overlay inspection), not transcribed literally from
`instance.rddl` — the reference scene image does not place objects at their
literal RDDL (x,y) coordinates (only Santa and the stable happen to match).
Three inconsistencies were found in the source art and resolved as follows
(see plan doc, "scene.png ground-truth reconciliation" section, for detail):
  - Two boxes are both labeled "P3" (one green at (4,2), one blue at (4,4)) —
    kept as two separate presents (present3, present3b): 5 presents total.
  - Only 3 house buildings are drawn (badges A/B/C at (4,6)/(6,6)/(8,6)) even
    though the RDDL names 4 (house_a..house_d) — house_d dropped; houses can
    each receive multiple presents (win = all 5 presents delivered, not a
    1-present-per-house match).
  - Win condition is therefore `delivery_count == 5`, not the RDDL's 4.

Santa moves one cell per step (4 discrete actions); stepping onto a
present/house/stable auto-triggers the corresponding collect/deliver/rest
transition — there are no separate pickup/dropoff actions, matching the
RDDL's `match-santa_class-*` cpfs which fire purely off position after a move.

Rendering draws Santa's red "S" badge onto a copy of the reference scene.png
by moving the sprite (detect+cutpaste, see `methods.cutpaste`) and clears
collected-present / delivered-house cells. Both the sprite's source hole and
any destination cell are filled from a local same-size crop of the PRISTINE
scene one row below (still snowy floor, same lighting band) rather than a
single fixed remote patch, so cells that hold decorations (trees, snowballs)
still get a locally-consistent fill instead of a mismatched-tone seam.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError("gymnasium is required: uv pip install gymnasium") from e

from PIL import Image

from .geometry import SceneGeom, MOVES, apply_move
from .methods import _find_s

SCENE_PATH = Path(__file__).resolve().parents[2] / "game_ref" / "New Folder With Items" / "scene.png"

ACTIONS = ["left", "right", "up", "down"]  # Discrete(4) index -> move name

# Object cells, 1-indexed (cx, cy) — VERIFIED against scene.png (see module
# docstring), not the literal instance.rddl non-fluents.
SANTA_START = (1, 1)
PRESENTS = {  # name -> cell
    "present1": (2, 2),
    "present2": (2, 3),
    "present3": (4, 2),
    "present3b": (4, 4),
    "present4": (7, 3),
}
HOUSES = {  # each can receive multiple presents (only 3 buildings exist)
    "house_a": (4, 7),
    "house_b": (6, 7),
    "house_c": (8, 7),
}
STABLE = {"reinder_stable": (1, 8)}

WIN_DELIVERY_COUNT = len(PRESENTS)  # 5
MAX_HORIZON = 40  # generous vs instance.rddl's 25, since paths are longer now

# Cell (3,3) is verified by direct visual inspection to be plain, undecorated
# snow — no tree/snowball/present/house anywhere near it. Earlier heuristics
# (sample one row below the hole) sometimes landed on ANOTHER decorated cell
# (e.g. present1 at (2,2) when clearing Santa's start (1,1)), leaving visible
# fragments — a fixed, verified-clean source cell avoids that entirely.
CLEAN_FILL_CELL = (3, 3)
FILL_PAD = 10  # px pad so decorative art that overflows a cell's own bbox
               # (ribbons, sprite rings) still gets fully cleared/covered.
               # NOTE: 35 was tried and made things WORSE (pulled in/created
               # ghost fragments from neighboring cells) — keep this at 10.


def _padded_bbox(bbox):
    bl, bt, br, bb = (int(v) for v in bbox)
    return (bl - FILL_PAD, bt - FILL_PAD, br + FILL_PAD, bb + FILL_PAD)


def _bg_fill(img: np.ndarray, padded_bbox, geom: SceneGeom):
    """Same-size patch of clean snow, sampled from the fixed verified-clean
    `CLEAN_FILL_CELL`, re-centered each call so the patch always matches the
    hole's exact size. `padded_bbox` must already include the `FILL_PAD`
    margin (see `_padded_bbox`) so callers clear/fill the identical region."""
    bl, bt, br, bb = padded_bbox
    h, w = bb - bt, br - bl
    fcx, fcy = geom.center(*CLEAN_FILL_CELL)
    fl, ft = int(fcx - w / 2), int(fcy - h / 2)
    fh_max, fw_max = img.shape[0], img.shape[1]
    ft = max(0, min(ft, fh_max - h))
    fl = max(0, min(fl, fw_max - w))
    return img[ft:ft + h, fl:fl + w].copy()


class SantaEnv(gym.Env):
    """Discrete(4) grid env for Santa's Gift Journey. See module docstring
    for how reward/termination diverge from the literal RDDL (5 presents,
    3 multi-delivery houses, win at delivery_count==5)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *, max_steps: int = MAX_HORIZON, seed: Optional[int] = None):
        super().__init__()
        self.geom = SceneGeom()
        self._scene = Image.open(SCENE_PATH).convert("RGB")
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        h, w = self._scene.size[1], self._scene.size[0]
        self.observation_space = spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)
        self._rng = np.random.default_rng(seed)

        self.santa_cell = SANTA_START
        self.carrying_present_id = 0  # 0 = none, 1 = carrying one
        self.is_collected = {k: False for k in PRESENTS}
        self.is_delivered_count = {k: 0 for k in HOUSES}
        self.delivery_count = 0
        self.stable_visited = False
        self.t = 0

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.santa_cell = (options or {}).get("santa_cell", SANTA_START)
        self.carrying_present_id = 0
        self.is_collected = {k: False for k in PRESENTS}
        self.is_delivered_count = {k: 0 for k in HOUSES}
        self.delivery_count = 0
        self.stable_visited = False
        self.t = 0
        return self._render(), self._info()

    def step(self, action: int):
        self.t += 1
        move = ACTIONS[int(action)]
        new_cell = apply_move(self.santa_cell, move, self.geom)
        self.santa_cell = new_cell

        reward_delta = 0.0
        # Match(SantaClass, Present): pick up if standing on an uncollected present
        for name, cell in PRESENTS.items():
            if cell == self.santa_cell and not self.is_collected[name] and self.carrying_present_id == 0:
                self.is_collected[name] = True
                self.carrying_present_id = 1
                break
        # Match(SantaClass, House): deliver if carrying and standing on a house
        # (houses can receive more than one present — only 3 buildings exist
        # for 5 presents, see module docstring).
        for name, cell in HOUSES.items():
            if cell == self.santa_cell and self.carrying_present_id > 0:
                self.is_delivered_count[name] += 1
                self.carrying_present_id = 0
                self.delivery_count += 1
                reward_delta += 1.0
                break
        # Match(SantaClass, Stable)
        for name, cell in STABLE.items():
            if cell == self.santa_cell and not self.stable_visited:
                self.stable_visited = True
                reward_delta += 1.0
                break

        terminated = self.delivery_count == WIN_DELIVERY_COUNT
        truncated = self.t >= self.max_steps and not terminated
        return self._render(), reward_delta, terminated, truncated, self._info()

    def render(self):
        return self._render()

    # -- internals -----------------------------------------------------------
    def _info(self) -> dict:
        return dict(
            santa_cell=self.santa_cell,
            carrying_present_id=self.carrying_present_id,
            is_collected=dict(self.is_collected),
            is_delivered_count=dict(self.is_delivered_count),
            delivery_count=self.delivery_count,
            stable_visited=self.stable_visited,
            t=self.t,
        )

    def _render(self) -> np.ndarray:
        """Clear collected-present cells first, then paste the S sprite last
        so it always draws on top (Santa may be standing on a present cell
        the same frame it gets collected). Houses are never visually cleared
        (they can receive multiple presents, and stay as normal buildings)."""
        img = np.array(self._scene)
        geom = self.geom
        out = img.copy()

        for name, cell in PRESENTS.items():
            if self.is_collected[name]:
                pl, pt, pr, pb = _padded_bbox(geom.cell_bbox(*cell))
                out[pt:pb, pl:pr] = _bg_fill(img, (pl, pt, pr, pb), geom)

        bbox = _find_s(img, geom, SANTA_START)
        if bbox is not None and self.santa_cell != SANTA_START:
            bl, bt, br, bb = [int(v) for v in bbox]
            sprite = img[bt:bb, bl:br].copy()
            # clear the ORIGINAL S location (padded, since the sprite's ring/
            # shadow overflows its own bbox) before pasting at the
            # destination, else the old badge is left behind (ghost duplicate).
            spl, spt, spr, spb = _padded_bbox((bl, bt, br, bb))
            out[spt:spb, spl:spr] = _bg_fill(img, (spl, spt, spr, spb), geom)
            dx = int((self.santa_cell[0] - SANTA_START[0]) * geom.cell)
            dy = int((self.santa_cell[1] - SANTA_START[1]) * geom.cell)
            # clear the DESTINATION cell's own contents (tree/present/etc,
            # possibly overflowing its cell bbox too) before pasting S on top.
            dpl, dpt, dpr, dpb = _padded_bbox(geom.cell_bbox(*self.santa_cell))
            out[dpt:dpb, dpl:dpr] = _bg_fill(img, (dpl, dpt, dpr, dpb), geom)
            R, G, B = sprite[..., 0].astype(int), sprite[..., 1].astype(int), sprite[..., 2].astype(int)
            m = ((R > 150) & (R - G > 60) & (R - B > 60)) | ((R > 200) & (G > 200) & (B > 200))
            bt2, bb2, bl2, br2 = bt + dy, bb + dy, bl + dx, br + dx
            reg = out[bt2:bb2, bl2:br2]
            reg[m] = sprite[m]
            out[bt2:bb2, bl2:br2] = reg

        return out
