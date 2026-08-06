"""Stage 2 (real-copy): erase the sprite using the SAME cell from a real frame
where the sprite has moved away. Pixel-perfect, native, zero synthesis/hallucination.

In the playthrough, step00 has Santa at (1,1); step01 has Santa moved to (2,1),
so cell (1,1) in step01 is genuinely empty snow. Copy that real empty cell over
the badge in step00 -> a perfect empty plate.

    python scripts/plate_realcopy.py \
        --with-sprite runs/diamond_dev/playthrough_demo/step00_start.png \
        --without-sprite runs/diamond_dev/playthrough_demo/step01_right_2-1.png \
        --cell 1 1 --out runs/plate_santa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy import ndimage

ORIGIN = (227, 227); CELL = 87.5


def cell_box(col, row, pad=6):
    cx = ORIGIN[0] + (col - 1) * CELL
    cy = ORIGIN[1] + (row - 1) * CELL
    h = CELL / 2 + pad
    return int(cx - h), int(cy - h), int(cx + h), int(cy + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-sprite", required=True, help="frame WHERE the sprite is (to erase)")
    ap.add_argument("--without-sprite", required=True, help="frame where sprite has moved away")
    ap.add_argument("--cell", type=int, nargs=2, required=True, metavar=("COL", "ROW"))
    ap.add_argument("--out", default="runs/plate_santa")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    withb = np.asarray(Image.open(args.with_sprite).convert("RGB")).copy()
    without = np.asarray(Image.open(args.without_sprite).convert("RGB"))

    box = cell_box(*args.cell)
    x0, y0, x1, y1 = box
    donor = without[y0:y1, x0:x1].astype(float)      # real empty cell
    region = withb[y0:y1, x0:x1].astype(float)
    h, w = region.shape[:2]

    # feather the paste at the cell border so any tiny lighting/snow drift blends
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(float)
    soft = np.clip(edge / 8.0, 0, 1)[..., None]      # 0 at border -> 1 inside
    blended = soft * donor + (1 - soft) * region
    withb[y0:y1, x0:x1] = blended.round().astype(np.uint8)

    Image.fromarray(withb).save(out / "plate.png")
    print(f"copied REAL empty cell {tuple(args.cell)} from {Path(args.without_sprite).name}")
    print(f"-> {out}/plate.png  (pixel-perfect empty board)")


if __name__ == "__main__":
    main()
