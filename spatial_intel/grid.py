"""Grid world, rendering and spatial geometry for the spatial-intelligence task.

Coordinate convention
----------------------
The board is an ``N x N`` grid of cells indexed ``(col, row)`` where

    col (x) increases to the RIGHT
    row (y) increases DOWNWARD   (standard image convention)

so "up"/"top" means *row decreases* and "down" means *row increases*.

The movable object is a solid triangle that occupies a square block of
``object_size x object_size`` cells. Its position is the *top-left* cell of
that block, ``(ox, oy)``. For the default 8x8 board with ``object_size=2`` the
centre block is top-left ``(3, 3)`` covering cells (3,3),(4,3),(3,4),(4,4) —
exactly the example in the task brief.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------
# Each direction maps to a (d_col, d_row) unit step in *cell* units.
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "right": (1, 0),
    "left": (-1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "up_right": (1, -1),    # a.k.a. "top right"
    "up_left": (-1, -1),    # a.k.a. "top left"
    "down_right": (1, 1),
    "down_left": (-1, 1),
}
DIRECTION_NAMES = list(DIRECTIONS.keys())
DIRECTION_ID = {name: i for i, name in enumerate(DIRECTION_NAMES)}

# Friendly aliases the way a human (or a prompt) might phrase them.
DIRECTION_ALIASES = {
    "top": "up", "bottom": "down", "north": "up", "south": "down",
    "east": "right", "west": "left",
    "top right": "up_right", "top-right": "up_right", "topright": "up_right",
    "top left": "up_left", "top-left": "up_left", "topleft": "up_left",
    "bottom right": "down_right", "bottom-right": "down_right",
    "bottom left": "down_left", "bottom-left": "down_left",
    "north east": "up_right", "north west": "up_left",
    "south east": "down_right", "south west": "down_left",
}


def normalize_direction(name: str) -> str:
    """Map an arbitrary phrasing to a canonical direction key."""
    key = name.strip().lower().replace("_", " ")
    if key in DIRECTIONS:
        return key
    key2 = name.strip().lower()
    if key2 in DIRECTIONS:
        return key2
    if key in DIRECTION_ALIASES:
        return DIRECTION_ALIASES[key]
    if key2 in DIRECTION_ALIASES:
        return DIRECTION_ALIASES[key2]
    raise ValueError(f"Unknown direction: {name!r}")


# ---------------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    n: int = 8                  # cells per side
    object_size: int = 2        # object spans object_size x object_size cells
    cell_px: int = 64           # pixels per cell
    margin: int = 44            # left/top margin (pixels) reserved for axis labels
    bg: Tuple[int, int, int] = (255, 255, 255)
    line: Tuple[int, int, int] = (200, 200, 200)
    axis: Tuple[int, int, int] = (40, 40, 40)
    obj_color: Tuple[int, int, int] = (220, 30, 30)   # vivid red triangle

    @property
    def img_w(self) -> int:
        return self.margin + self.n * self.cell_px + 8

    @property
    def img_h(self) -> int:
        return self.margin + self.n * self.cell_px + 8

    @property
    def max_topleft(self) -> int:
        """Largest valid top-left index so the object stays on the board."""
        return self.n - self.object_size

    def in_bounds(self, topleft: Tuple[int, int]) -> bool:
        ox, oy = topleft
        return 0 <= ox <= self.max_topleft and 0 <= oy <= self.max_topleft

    # -- pixel <-> cell mapping ------------------------------------------
    def cell_origin_px(self, col: int, row: int) -> Tuple[int, int]:
        """Top-left pixel of cell (col, row)."""
        return (self.margin + col * self.cell_px, self.margin + row * self.cell_px)

    def px_to_cell_frac(self, x: float, y: float) -> Tuple[float, float]:
        """Map a pixel coordinate to fractional cell coordinates."""
        return ((x - self.margin) / self.cell_px, (y - self.margin) / self.cell_px)

    def cells_covered(self, topleft: Tuple[int, int]):
        ox, oy = topleft
        return {(ox + dx, oy + dy)
                for dx in range(self.object_size)
                for dy in range(self.object_size)}


# ---------------------------------------------------------------------------
# Move geometry & targets
# ---------------------------------------------------------------------------
def apply_move(spec: GridSpec, topleft: Tuple[int, int], direction: str,
               steps: int) -> Tuple[int, int]:
    """Move *unclamped*. Returns the raw target top-left (may be off board)."""
    d = normalize_direction(direction)
    dc, dr = DIRECTIONS[d]
    return (topleft[0] + dc * steps, topleft[1] + dr * steps)


def clamp_topleft(spec: GridSpec, topleft: Tuple[int, int]) -> Tuple[int, int]:
    m = spec.max_topleft
    return (int(np.clip(topleft[0], 0, m)), int(np.clip(topleft[1], 0, m)))


def max_steps_for(spec: GridSpec, topleft: Tuple[int, int], direction: str) -> int:
    """How many cells the object can travel in `direction` before leaving board."""
    d = normalize_direction(direction)
    dc, dr = DIRECTIONS[d]
    m = spec.max_topleft
    lim = []
    if dc > 0:
        lim.append(m - topleft[0])
    elif dc < 0:
        lim.append(topleft[0])
    if dr > 0:
        lim.append(m - topleft[1])
    elif dr < 0:
        lim.append(topleft[1])
    return max(0, min(lim)) if lim else 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _triangle_points(spec: GridSpec, topleft: Tuple[int, int]):
    """Vertices of an upward triangle inscribed in the object's bbox."""
    ox, oy = topleft
    x0, y0 = spec.cell_origin_px(ox, oy)
    side = spec.object_size * spec.cell_px
    inset = max(4, side // 12)
    x1, y1 = x0 + side, y0 + side
    apex = ((x0 + x1) / 2, y0 + inset)
    bl = (x0 + inset, y1 - inset)
    br = (x1 - inset, y1 - inset)
    return [apex, bl, br]


def render_state(spec: GridSpec, topleft: Tuple[int, int],
                 *, show_cell_coords: bool = True,
                 feedback_arrow: Tuple[Tuple[int, int], Tuple[int, int]] | None = None,
                 caption: str | None = None) -> Image.Image:
    """Render the board with grid, axis/coordinate overlay and the triangle.

    feedback_arrow: optional (from_topleft, to_topleft) drawn as a hint arrow
                    (used by the env to give visual feedback between attempts).
    caption: optional text drawn along the bottom.
    """
    img = Image.new("RGB", (spec.img_w, spec.img_h), spec.bg)
    d = ImageDraw.Draw(img)
    f_axis = _font(max(12, spec.cell_px // 4))
    f_cell = _font(max(9, spec.cell_px // 6))

    x_end = spec.margin + spec.n * spec.cell_px
    y_end = spec.margin + spec.n * spec.cell_px

    # grid lines
    for i in range(spec.n + 1):
        x = spec.margin + i * spec.cell_px
        y = spec.margin + i * spec.cell_px
        d.line([(x, spec.margin), (x, y_end)], fill=spec.line, width=1)
        d.line([(spec.margin, y), (x_end, y)], fill=spec.line, width=1)

    # axis labels (column indices along top, row indices along left)
    for c in range(spec.n):
        cx = spec.margin + c * spec.cell_px + spec.cell_px // 2
        d.text((cx, spec.margin // 2), str(c), fill=spec.axis,
               anchor="mm", font=f_axis)
    for r in range(spec.n):
        cy = spec.margin + r * spec.cell_px + spec.cell_px // 2
        d.text((spec.margin // 2, cy), str(r), fill=spec.axis,
               anchor="mm", font=f_axis)

    # faint per-cell (col,row) coordinates
    if show_cell_coords:
        for c in range(spec.n):
            for r in range(spec.n):
                ox, oy = spec.cell_origin_px(c, r)
                d.text((ox + 3, oy + 2), f"{c},{r}", fill=(195, 195, 205),
                       font=f_cell)

    # feedback arrow (drawn before the triangle so the object stays on top)
    if feedback_arrow is not None:
        (fc, fr), (tc, tr) = feedback_arrow
        half = spec.object_size / 2.0
        p0 = spec.cell_origin_px(0, 0)
        sx = spec.margin + (fc + half) * spec.cell_px
        sy = spec.margin + (fr + half) * spec.cell_px
        ex = spec.margin + (tc + half) * spec.cell_px
        ey = spec.margin + (tr + half) * spec.cell_px
        d.line([(sx, sy), (ex, ey)], fill=(20, 120, 220), width=3)
        # arrow head
        ang = np.arctan2(ey - sy, ex - sx)
        for da in (-0.4, 0.4):
            hx = ex - 14 * np.cos(ang + da)
            hy = ey - 14 * np.sin(ang + da)
            d.line([(ex, ey), (hx, hy)], fill=(20, 120, 220), width=3)

    # object
    d.polygon(_triangle_points(spec, topleft), fill=spec.obj_color)

    if caption:
        d.text((spec.margin, y_end + 2), caption, fill=spec.axis, font=f_cell)
    return img


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def from_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(arr, dtype=np.uint8), "RGB")


# ---------------------------------------------------------------------------
# Object detection (read the object position back out of a rendered/generated image)
# ---------------------------------------------------------------------------
def detect_object_topleft(arr: np.ndarray, spec: GridSpec):
    """Estimate the object's top-left cell from an image.

    Works on our own renders *and* is tolerant enough for model-generated
    images: it looks for the "reddest" blob (high R, low G/B) and maps its
    centroid back to cell coordinates. Returns (topleft, confidence) where
    confidence is the fraction of board pixels that fired the mask.
    """
    a = arr.astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 120) & (r - g > 50) & (r - b > 50)
    n_fired = int(mask.sum())
    board_px = spec.n * spec.cell_px
    if n_fired < 8:
        return None, 0.0
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    fcol, frow = spec.px_to_cell_frac(cx, cy)
    # centroid sits at the object's centre -> back out the top-left
    half = spec.object_size / 2.0
    ox = int(round(fcol - half))
    oy = int(round(frow - half))
    ox = int(np.clip(ox, 0, spec.max_topleft))
    oy = int(np.clip(oy, 0, spec.max_topleft))
    conf = n_fired / float(board_px * board_px)
    return (ox, oy), conf
