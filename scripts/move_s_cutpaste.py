"""Baseline next-frame generator: move S by cut-paste (exact, instant, no diffusion).

This is the GROUND-TRUTH next frame — the world model must reproduce this. Detect
the S badge sprite in its current cell, copy it into the target cell, and fill the
vacated cell with local background (median of surrounding grid pixels).

    python scripts/move_s_cutpaste.py --move right --out runs/move_demo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from spatial_intel.game.geometry import SceneGeom, apply_move

SCENE = "game_ref/New Folder With Items/scene.png"


def find_s_badge(img: np.ndarray, geom: SceneGeom, s_cell):
    """Return a tight bbox of the red S badge within its cell."""
    l, t, r, b = [int(v) for v in geom.cell_bbox(*s_cell)]
    crop = img[t:b, l:r]
    R, G, B = crop[..., 0].astype(int), crop[..., 1].astype(int), crop[..., 2].astype(int)
    red = (R > 180) & (R - G > 90) & (R - B > 90)
    ys, xs = np.where(red)
    if len(xs) == 0:
        return None
    pad = 6
    return (l + xs.min() - pad, t + ys.min() - pad, l + xs.max() + pad, t + ys.max() + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", choices=["left", "right", "up", "down"], default="right")
    ap.add_argument("--s-cell", type=int, nargs=2, default=[1, 1])
    ap.add_argument("--out", default="runs/move_demo")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    geom = SceneGeom()
    img = np.array(Image.open(SCENE).convert("RGB"))
    s_cell = tuple(args.s_cell)
    new_cell = apply_move(s_cell, args.move, geom)
    print(f"move '{args.move}': S {s_cell} -> {new_cell}")
    if new_cell == s_cell:
        print("(blocked by board edge)")

    bbox = find_s_badge(img, geom, s_cell)
    if bbox is None:
        print("could not find S badge!"); return
    bl, bt, br, bb = [int(v) for v in bbox]
    sprite = img[bt:bb, bl:br].copy()
    print(f"S sprite bbox: {bbox} ({br-bl}x{bb-bt}px)")

    result = img.copy()
    # 1) fill old S cell with background. Snow-only cells (no sprite) are the
    #    fill source; (7,1)/(8,1) top-right are empty snow. Sample a patch of the
    #    SAME size at a clean snow cell to fill the hole seamlessly.
    fcx, fcy = (7, 1)                                       # verified empty snow cell
    fpx, fpy = geom.center(fcx, fcy)
    ph, pw = bb - bt, br - bl
    fl2, ft2 = int(fpx - pw / 2), int(fpy - ph / 2)
    fill = img[ft2:ft2 + ph, fl2:fl2 + pw].copy()
    result[bt:bb, bl:br] = fill

    # 2) paste sprite at the new cell (shift by the cell delta in pixels)
    dx = int((new_cell[0] - s_cell[0]) * geom.cell)
    dy = int((new_cell[1] - s_cell[1]) * geom.cell)
    nbt, nbb, nbl, nbr = bt + dy, bb + dy, bl + dx, br + dx
    # only paste the reddish (badge) pixels so we don't overwrite background with the sprite's bg
    R, G, B = sprite[..., 0].astype(int), sprite[..., 1].astype(int), sprite[..., 2].astype(int)
    mask = ((R > 150) & (R - G > 60) & (R - B > 60)) | ((R > 200) & (G > 200) & (B > 200))  # red + white
    region = result[nbt:nbb, nbl:nbr]
    region[mask] = sprite[mask]
    result[nbt:nbb, nbl:nbr] = region

    Image.fromarray(result).save(out / f"next_{args.move}.png")
    Image.fromarray(img).save(out / "frame.png")
    print(f"saved {out}/next_{args.move}.png (ground-truth next frame)")


if __name__ == "__main__":
    main()
