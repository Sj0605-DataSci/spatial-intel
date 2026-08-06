"""Stage 2 (patch-copy): erase the sprite by copying a CLEAN empty-cell patch
over its cell. No diffusion, no hallucination — native board pixels only.

The board is a regular grid (SceneGeom). Santa's cell gets overwritten with the
pixels of an empty snowy donor cell (auto-picked as the cell whose patch has the
lowest saturation / edge energy = plainest snow, or a --donor you specify).

    python scripts/plate_patchcopy.py \
        --frame runs/diamond_dev/playthrough_demo/step00_start.png \
        --sprite-cell 1 1 --out runs/plate_santa
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

# scene geometry (our clean 8x8 santa board, 1024px)
ORIGIN = (227, 227); CELL = 87.5; N = 8


def cell_box(col, row):
    """1-indexed cell -> pixel box (x0,y0,x1,y1), centered on the grid cell."""
    cx = ORIGIN[0] + (col - 1) * CELL
    cy = ORIGIN[1] + (row - 1) * CELL
    h = CELL / 2
    return int(cx - h), int(cy - h), int(cx + h), int(cy + h)


def patch_plainness(arr, box):
    """Lower = plainer (empty snow). Combines color saturation + edge energy."""
    x0, y0, x1, y1 = box
    p = arr[y0:y1, x0:x1].astype(float)
    if p.size == 0:
        return 1e9
    sat = (p.max(-1) - p.min(-1)).mean()            # colored objects raise this
    gx = np.abs(np.diff(p.mean(-1), axis=1)).mean()  # edges/detail
    gy = np.abs(np.diff(p.mean(-1), axis=0)).mean()
    return sat + 6 * (gx + gy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--sprite-cell", type=int, nargs=2, required=True, metavar=("COL", "ROW"))
    ap.add_argument("--donor", type=int, nargs=2, default=None, metavar=("COL", "ROW"),
                    help="explicit empty donor cell; default = auto-pick plainest")
    ap.add_argument("--out", default="runs/plate_santa")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    img = Image.open(args.frame).convert("RGB")
    arr = np.asarray(img).copy()
    sc, sr = args.sprite_cell

    if args.donor:
        dc, dr = args.donor
    else:
        # rank all cells by plainness, skip the sprite cell; pick the plainest
        cands = []
        for col in range(1, N + 1):
            for row in range(1, N + 1):
                if (col, row) == (sc, sr):
                    continue
                cands.append((patch_plainness(arr, cell_box(col, row)), col, row))
        cands.sort()
        _, dc, dr = cands[0]
        print(f"auto-picked donor cell ({dc},{dr}) "
              f"(plainness {cands[0][0]:.1f}; next {cands[1][1:]}={cands[1][0]:.1f})")

    dbox = cell_box(dc, dr)
    sbox = cell_box(sc, sr)
    th, tw = sbox[3] - sbox[1], sbox[2] - sbox[0]
    donor = np.asarray(Image.fromarray(
        arr[dbox[1]:dbox[3], dbox[0]:dbox[2]]).resize((tw, th))).astype(float)

    # Only replace the SPRITE'S FOOTPRINT (dilated), not the whole cell — this keeps
    # the original grid lines and surrounding snow, erasing just where the badge was.
    from scipy import ndimage
    mask_path = Path("runs/sprite_santa/sprite_meta.json")
    # rebuild a footprint mask from the sprite bbox if a saved mask isn't handy:
    # use the alpha of the extracted sprite, placed at the sprite cell.
    sprite_rgba = np.asarray(Image.open("runs/sprite_santa/sprite_rgba.png").convert("RGBA"))
    alpha = sprite_rgba[..., 3] > 10
    # place alpha into a cell-sized canvas (sprite was tight-cropped ~ badge size)
    fp = np.zeros((th, tw), bool)
    ah, aw = alpha.shape
    oy, ox = max(0, (th - ah) // 2), max(0, (tw - aw) // 2)
    fp[oy:oy + min(ah, th), ox:ox + min(aw, tw)] = alpha[:min(ah, th), :min(aw, tw)]
    fp = ndimage.binary_dilation(fp, iterations=8)          # cover halo
    # feathered alpha (soft edge) so the paste has no hard rectangle seam
    soft = ndimage.gaussian_filter(fp.astype(float), sigma=3)
    soft = np.clip(soft, 0, 1)[..., None]

    region = arr[sbox[1]:sbox[3], sbox[0]:sbox[2]].astype(float)
    blended = soft * donor + (1 - soft) * region
    arr[sbox[1]:sbox[3], sbox[0]:sbox[2]] = blended.round().astype(np.uint8)

    Image.fromarray(arr).save(out / "plate.png")
    print(f"copied empty cell ({dc},{dr}) -> ({sc},{sr}) footprint-only, feathered")
    print(f"-> {out}/plate.png  (empty board, native pixels, no seam)")


if __name__ == "__main__":
    main()
