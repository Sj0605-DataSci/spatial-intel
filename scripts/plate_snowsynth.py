"""Stage 2 (snow-synth): erase the sprite by SYNTHESIZING snow texture from the
color distribution of geometry-picked empty cells. No spatial region is ever
copied, so no house/present/tree can leak in.

  1. score every cell by 'plainness' (low saturation + low edge energy)
  2. pool pixels from the N plainest cells  -> the snow distribution
  3. synthesize a snow tile: local mean color + matched gaussian speckle,
     low-freq shading from the pooled snow  -> fill the badge footprint (feathered)

    python scripts/plate_snowsynth.py \
        --frame runs/diamond_dev/playthrough_demo/step00_start.png \
        --sprite-rgba runs/sprite_santa/sprite_rgba.png \
        --sprite-cell 1 1 --out runs/plate_santa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy import ndimage

ORIGIN = (227, 227); CELL = 87.5; N = 8


def cell_box(col, row):
    cx = ORIGIN[0] + (col - 1) * CELL
    cy = ORIGIN[1] + (row - 1) * CELL
    h = CELL / 2
    return int(cx - h), int(cy - h), int(cx + h), int(cy + h)


def plainness(arr, box):
    x0, y0, x1, y1 = box
    p = arr[y0:y1, x0:x1].astype(float)
    if p.size == 0:
        return 1e9
    sat = (p.max(-1) - p.min(-1)).mean()
    gx = np.abs(np.diff(p.mean(-1), 1)).mean()
    gy = np.abs(np.diff(p.mean(-1), 0)).mean()
    return sat + 6 * (gx + gy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--sprite-rgba", default="runs/sprite_santa/sprite_rgba.png")
    ap.add_argument("--sprite-cell", type=int, nargs=2, required=True, metavar=("COL", "ROW"))
    ap.add_argument("--n-donors", type=int, default=5, help="plainest cells to pool")
    ap.add_argument("--out", default="runs/plate_santa")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.frame).convert("RGB")
    arr = np.asarray(img).copy()
    sc, sr = args.sprite_cell

    # rank cells, pool pixels from the plainest N (never the sprite cell)
    ranked = sorted(
        ((plainness(arr, cell_box(c, r)), c, r)
         for c in range(1, N + 1) for r in range(1, N + 1) if (c, r) != (sc, sr)))
    donors = ranked[:args.n_donors]
    print("snow donor cells:", [(c, r) for _, c, r in donors])
    pool = np.concatenate([
        arr[cell_box(c, r)[1]:cell_box(c, r)[3], cell_box(c, r)[0]:cell_box(c, r)[2]]
           .reshape(-1, 3) for _, c, r in donors]).astype(float)
    mean, std = pool.mean(0), pool.std(0)
    print(f"snow mean RGB {mean.round(1)}  std {std.round(1)}")

    # sprite footprint (dilated alpha) centered in the sprite cell
    sbox = cell_box(sc, sr)
    th, tw = sbox[3] - sbox[1], sbox[2] - sbox[0]
    alpha = np.asarray(Image.open(args.sprite_rgba).convert("RGBA"))[..., 3] > 10
    fp = np.zeros((th, tw), bool)
    ah, aw = alpha.shape
    oy, ox = max(0, (th - ah) // 2), max(0, (tw - aw) // 2)
    fp[oy:oy + min(ah, th), ox:ox + min(aw, tw)] = alpha[:min(ah, th), :min(aw, tw)]
    fp = ndimage.binary_dilation(fp, iterations=8)
    soft = np.clip(ndimage.gaussian_filter(fp.astype(float), sigma=3), 0, 1)[..., None]

    # synthesize snow: local mean of the CELL (so shading matches its neighbourhood)
    # + matched fine speckle. Start from the current cell's own low-freq shading so
    # the fill blends with lighting, then overwrite only structure with snow stats.
    region = arr[sbox[1]:sbox[3], sbox[0]:sbox[2]].astype(float)
    base = ndimage.gaussian_filter(region, sigma=(10, 10, 0))          # local shading
    base = 0.5 * base + 0.5 * mean                                     # pull toward snow color
    rng = np.random.default_rng(0)
    speckle = rng.normal(0, std * 0.6, (th, tw, 3))                    # matched grain
    tile = np.clip(base + speckle, 0, 255)

    blended = soft * tile + (1 - soft) * region
    arr[sbox[1]:sbox[3], sbox[0]:sbox[2]] = blended.round().astype(np.uint8)

    Image.fromarray(arr).save(out / "plate.png")
    print(f"-> {out}/plate.png  (badge erased with synthesized snow, no region copy)")


if __name__ == "__main__":
    main()
