"""Grid <-> pixel geometry for the Santa's Gift Journey scene image.

Solved from the reference scene.png (1024x1024) using known RDDL object cells
(Santa at (1,1), stable at (1,8)): the 8x8 play grid has cell size ~87.5px with
cell (1,1) CENTER at pixel (227, 227). Cells are 1-indexed (x=col 1..8 left->right,
y=row 1..8 top->bottom), matching the RDDL/instance convention.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneGeom:
    origin_x: float = 227.0     # pixel center of cell (1,1)
    origin_y: float = 227.0
    cell: float = 87.5          # pixel size of one cell
    n: int = 8

    def center(self, cx: int, cy: int):
        """Pixel center of cell (cx, cy), 1-indexed."""
        return (self.origin_x + (cx - 1) * self.cell,
                self.origin_y + (cy - 1) * self.cell)

    def cell_bbox(self, cx: int, cy: int):
        """(left, top, right, bottom) pixel box of cell (cx, cy)."""
        px, py = self.center(cx, cy)
        h = self.cell / 2
        return (px - h, py - h, px + h, py + h)

    def pixel_to_cell(self, px: float, py: float):
        """Nearest cell (cx, cy) for a pixel; clamped to the board."""
        cx = round((px - self.origin_x) / self.cell) + 1
        cy = round((py - self.origin_y) / self.cell) + 1
        cx = max(1, min(self.n, cx))
        cy = max(1, min(self.n, cy))
        return (cx, cy)

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 1 <= cx <= self.n and 1 <= cy <= self.n


# the four grid moves, in (dx, dy) cell steps
MOVES = {
    "left":  (-1, 0),
    "right": (1, 0),
    "up":    (0, -1),
    "down":  (0, 1),
}


def apply_move(cell, move: str, geom: SceneGeom):
    """Return the new cell after a move, or the same cell if it would leave the board."""
    dx, dy = MOVES[move]
    nx, ny = cell[0] + dx, cell[1] + dy
    if geom.in_bounds(nx, ny):
        return (nx, ny)
    return cell
