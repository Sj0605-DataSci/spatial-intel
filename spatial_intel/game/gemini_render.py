"""Gemini (Vertex AI) image-editing renderer for Santa's Gift Journey frames.

First attempt (direction-word prompt only, no coordinate anchor) was
unreliable: even a plain valid single-cell move landed several cells off
(see plan doc, "Tested and rejected: Gemini as frame renderer"). This version
adds two fixes plus a safety net:
  1. A faint grid-line + column/row NUMBER overlay drawn onto the frame before
     sending it, so Gemini has an explicit visual coordinate reference instead
     of having to infer cell boundaries from the art.
  2. The prompt states the EXACT before/after (col,row) cell, not just a
     direction word.
  3. Verify + retry: after each generation, the deterministic pixel detector
     (`methods.detect`-style logic, reused via `_find_s`) checks whether S
     actually landed on the expected cell in Gemini's output. If not, retry
     up to `max_retries` times; if still wrong, `image` is returned anyway but
     `verified=False` so callers can fall back to the cutpaste renderer.

Credentials: a Vertex AI service-account JSON at `game_ref/gemini.json`
(gitignored — never commit). `project_id` is read from the JSON itself so we
don't duplicate it in source.
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .geometry import SceneGeom

CREDENTIALS_PATH = Path(__file__).resolve().parents[2] / "game_ref" / "gemini.json"
MODEL_ID = "gemini-3.1-flash-image"

NO_OVERLAY_INSTRUCTION = (
    " IMPORTANT: the magenta grid lines and 'col,row' number labels in the input image are "
    "ONLY a coordinate reference for you — they are NOT part of the actual game scene. Your "
    "output image must NOT contain any magenta lines or coordinate text at all; render the "
    "clean game scene exactly as it would look without that reference overlay."
)

BLOCKED_SUFFIX = (
    " The red circular 'S' badge (the player) does NOT move — keep it in EXACTLY the same grid "
    "cell it is already in. Keep the grid, all houses, presents, the stable, the HUD bar, "
    "decorations, and every other detail exactly the same. Nothing in the image should change."
    + NO_OVERLAY_INSTRUCTION
)

EDIT_SUFFIX = (
    " Keep the grid, all houses, presents, the stable, the HUD bar, decorations, "
    "and every other detail exactly the same. Only the S badge's position should change."
    + NO_OVERLAY_INSTRUCTION
)


def _grid_overlay(frame: Image.Image, geom: SceneGeom) -> Image.Image:
    """Draw faint magenta grid lines + column/row numbers onto a copy of
    `frame`, giving Gemini an explicit coordinate reference (same overlay
    style used for manual debugging, see runs/diamond_dev/scene_grid_overlay.png)."""
    out = frame.copy()
    d = ImageDraw.Draw(out)
    for c in range(1, geom.n + 1):
        for r in range(1, geom.n + 1):
            l, t, rr, b = geom.cell_bbox(c, r)
            d.rectangle([l, t, rr, b], outline=(255, 0, 255), width=1)
            d.text((l + 2, t + 2), f"{c},{r}", fill=(255, 0, 255))
    return out


def _detect_s_cell(img: np.ndarray, geom: SceneGeom):
    """Same biggest-red-blob detector as world_model_bench.py's detect_s_cell."""
    from scipy import ndimage
    R, G, B = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lbl == biggest)
    return geom.pixel_to_cell(xs.mean(), ys.mean())


@dataclass
class GeminiRenderResult:
    image: Optional[Image.Image]
    time_s: float
    verified: bool = False
    attempts: int = 0
    error: Optional[str] = None


def _load_client():
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Vertex credentials not found at {CREDENTIALS_PATH}. "
            "Place the service-account JSON there (gitignored)."
        )
    creds = json.loads(CREDENTIALS_PATH.read_text())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)
    os.environ["GOOGLE_CLOUD_PROJECT"] = creds["project_id"]
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    from google import genai
    return genai.Client()


class GeminiRenderer:
    """Lazy-loaded Vertex client; one instance reused across calls."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._client = None
        self.geom = SceneGeom()

    def load(self):
        if self._client is None:
            self._client = _load_client()

    def _prompt(self, before_cell, after_cell, move: str, blocked: bool) -> str:
        bc = f"({before_cell[0]},{before_cell[1]})"
        ac = f"({after_cell[0]},{after_cell[1]})"
        base = (
            f"The image has a faint magenta grid overlay with each cell labeled 'col,row' "
            f"(1-indexed, column increases right, row increases down). The red circular 'S' "
            f"badge (the player) is currently in cell {bc}."
        )
        if blocked:
            return base + f" The player tries to move {move.upper()}, but is blocked by the " \
                           f"grid edge." + BLOCKED_SUFFIX
        return base + f" Move the S badge from cell {bc} to cell {ac} (one step {move.upper()})." \
                       + EDIT_SUFFIX

    def render_move(self, frame: Image.Image, move: str, *, before_cell, after_cell,
                     blocked: bool = False, max_retries: int = 2) -> GeminiRenderResult:
        """Generate the next frame for `move`, with an explicit coordinate
        anchor (grid overlay + before/after cell numbers in the prompt), then
        verify against the deterministic detector and retry on mismatch."""
        self.load()
        t0 = time.time()
        overlaid = _grid_overlay(frame, self.geom)
        prompt = self._prompt(before_cell, after_cell, move, blocked)
        expected_cell = before_cell if blocked else after_cell

        last_img, last_err = None, None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id, contents=[overlaid, prompt])
                img = None
                for part in resp.candidates[0].content.parts:
                    if part.inline_data:
                        img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                        break
                if img is None:
                    last_err = "no image part in response"
                    continue
                last_img = img
                detected = _detect_s_cell(np.array(img), self.geom)
                if detected == tuple(expected_cell):
                    return GeminiRenderResult(image=img, time_s=time.time() - t0,
                                               verified=True, attempts=attempt)
                last_err = f"verify failed: detected={detected} expected={tuple(expected_cell)}"
            except Exception as e:
                last_err = str(e)

        # exhausted retries: return the last image (if any) unverified, so the
        # caller can decide to fall back to the deterministic cutpaste render.
        return GeminiRenderResult(image=last_img, time_s=time.time() - t0,
                                   verified=False, attempts=max_retries, error=last_err)
