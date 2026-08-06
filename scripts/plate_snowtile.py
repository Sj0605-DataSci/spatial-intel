"""Stage 2 (SAM snow-tile): erase the sprite by tiling REAL snow texture into its hole.

  Qwen2.5-VL -> bbox of an empty snowy area
  SAM2       -> precise pure-snow mask
  -> sample snow pixels from that mask, tile them over the sprite's footprint
     (feathered) so the fill is native snow texture — no object, no grid seam.

    python scripts/plate_snowtile.py \
        --frame runs/diamond_dev/playthrough_demo/step00_start.png \
        --sprite-rgba runs/sprite_santa/sprite_rgba.png \
        --sprite-cell 1 1 --out runs/plate_santa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.rl.policy import _load_hf_token_from_env_file
from extract_sprite import vlm_bbox, sam2_mask   # reuse Stage-1 helpers

ORIGIN = (227, 227); CELL = 87.5


def cell_box(col, row):
    cx = ORIGIN[0] + (col - 1) * CELL
    cy = ORIGIN[1] + (row - 1) * CELL
    h = CELL / 2
    return int(cx - h), int(cy - h), int(cx + h), int(cy + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--sprite-rgba", default="runs/sprite_santa/sprite_rgba.png")
    ap.add_argument("--sprite-cell", type=int, nargs=2, required=True, metavar=("COL", "ROW"))
    ap.add_argument("--snow-query", default="a large empty patch of plain white snow "
                    "with no objects, presents, trees or houses on it")
    ap.add_argument("--out", default="runs/plate_santa")
    args = ap.parse_args()

    _load_hf_token_from_env_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.frame).convert("RGB")
    arr = np.asarray(img).copy()

    print("Qwen2.5-VL: locating a clean snow region...")
    bbox, reply = vlm_bbox(args.frame, args.snow_query, device)
    print(f"  snow bbox = {bbox}")
    print("SAM2: segmenting pure snow...")
    snow_mask = sam2_mask(args.frame, bbox, device)
    snow_px = arr[snow_mask]                       # (Nsnow, 3) real snow pixels
    print(f"  snow region: {len(snow_px)} px")

    # sprite footprint (dilated alpha), centered in the sprite cell
    from scipy import ndimage
    sbox = cell_box(*args.sprite_cell)
    th, tw = sbox[3] - sbox[1], sbox[2] - sbox[0]
    alpha = np.asarray(Image.open(args.sprite_rgba).convert("RGBA"))[..., 3] > 10
    fp = np.zeros((th, tw), bool)
    ah, aw = alpha.shape
    oy, ox = max(0, (th - ah) // 2), max(0, (tw - aw) // 2)
    fp[oy:oy + min(ah, th), ox:ox + min(aw, tw)] = alpha[:min(ah, th), :min(aw, tw)]
    fp = ndimage.binary_dilation(fp, iterations=8)
    soft = np.clip(ndimage.gaussian_filter(fp.astype(float), sigma=3), 0, 1)[..., None]

    # tile a snow SWATCH from a contiguous crop of the snow mask (keeps texture, not noise)
    sy, sx = np.where(snow_mask)
    y0, y1, x0, x1 = sy.min(), sy.max() + 1, sx.min(), sx.max() + 1
    swatch = arr[y0:y1, x0:x1]
    tile = np.asarray(Image.fromarray(swatch).resize((tw, th))).astype(float)
    # subtle per-pixel jitter from the snow pixel distribution so it's not a flat copy
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, snow_px.std(0) * 0.15, (th, tw, 3))
    tile = np.clip(tile + jitter, 0, 255)

    region = arr[sbox[1]:sbox[3], sbox[0]:sbox[2]].astype(float)
    blended = soft * tile + (1 - soft) * region
    arr[sbox[1]:sbox[3], sbox[0]:sbox[2]] = blended.round().astype(np.uint8)

    Image.fromarray(arr).save(out / "plate.png")
    Image.fromarray((snow_mask * 255).astype(np.uint8)).save(out / "snow_mask.png")
    print(f"-> {out}/plate.png  (badge erased with native snow texture)")
    print(f"-> {out}/snow_mask.png (the SAM snow region used)")


if __name__ == "__main__":
    main()
