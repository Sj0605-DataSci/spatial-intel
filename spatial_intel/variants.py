"""Object-appearance variants (color x shape) for the generalization study.

The base task uses a RED TRIANGLE. To test attribute generalization we render
the SAME grid/placement with different appearances and detect them back:

    color: red | blue
    shape: triangle | square

Training is on red-triangle only; evaluation swaps appearance while keeping the
coordinates fixed, so a generalized model should still place correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from .grid import (GridSpec, _font, _triangle_points, to_array)

COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220)}
SHAPES = ("triangle", "square")


@dataclass(frozen=True)
class Appearance:
    color: str = "red"
    shape: str = "triangle"

    @property
    def name(self) -> str:
        return f"{self.color}_{self.shape}"

    @property
    def rgb(self):
        return COLORS[self.color]


def _square_points(spec: GridSpec, topleft):
    ox, oy = topleft
    x0, y0 = spec.cell_origin_px(ox, oy)
    side = spec.object_size * spec.cell_px
    inset = max(4, side // 12)
    return [x0 + inset, y0 + inset, x0 + side - inset, y0 + side - inset]


def render_variant(spec: GridSpec, topleft: Tuple[int, int], app: Appearance,
                   *, show_cell_coords: bool = True) -> Image.Image:
    """Render the board + grid + axes with the object drawn per `app`."""
    img = Image.new("RGB", (spec.img_w, spec.img_h), spec.bg)
    d = ImageDraw.Draw(img)
    f_axis = _font(max(12, spec.cell_px // 4))
    f_cell = _font(max(9, spec.cell_px // 6))
    x_end = spec.margin + spec.n * spec.cell_px
    y_end = spec.margin + spec.n * spec.cell_px

    for i in range(spec.n + 1):
        x = spec.margin + i * spec.cell_px
        y = spec.margin + i * spec.cell_px
        d.line([(x, spec.margin), (x, y_end)], fill=spec.line, width=1)
        d.line([(spec.margin, y), (x_end, y)], fill=spec.line, width=1)
    for c in range(spec.n):
        cx = spec.margin + c * spec.cell_px + spec.cell_px // 2
        d.text((cx, spec.margin // 2), str(c), fill=spec.axis, anchor="mm", font=f_axis)
    for r in range(spec.n):
        cy = spec.margin + r * spec.cell_px + spec.cell_px // 2
        d.text((spec.margin // 2, cy), str(r), fill=spec.axis, anchor="mm", font=f_axis)
    if show_cell_coords:
        for c in range(spec.n):
            for r in range(spec.n):
                ox, oy = spec.cell_origin_px(c, r)
                d.text((ox + 3, oy + 2), f"{c},{r}", fill=(195, 195, 205), font=f_cell)

    if app.shape == "triangle":
        d.polygon(_triangle_points(spec, topleft), fill=app.rgb)
    else:
        d.rectangle(_square_points(spec, topleft), fill=app.rgb)
    return img


def detect_variant_topleft(arr: np.ndarray, spec: GridSpec, app: Appearance):
    """Detect the object of the given color, return (topleft, confidence).

    Color-aware: red = high R, low G/B; blue = high B, low R/G. Shape doesn't
    change the centroid mapping (both shapes are centred in the block)."""
    a = arr.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if app.color == "red":
        mask = (r > 120) & (r - g > 50) & (r - b > 50)
    else:  # blue
        mask = (b > 120) & (b - r > 50) & (b - g > 40)
    n_fired = int(mask.sum())
    board_px = spec.n * spec.cell_px
    if n_fired < 8:
        return None, 0.0
    ys, xs = np.nonzero(mask)
    fcol, frow = spec.px_to_cell_frac(xs.mean(), ys.mean())
    half = spec.object_size / 2.0
    ox = int(np.clip(round(fcol - half), 0, spec.max_topleft))
    oy = int(np.clip(round(frow - half), 0, spec.max_topleft))
    return (ox, oy), n_fired / float(board_px * board_px)
