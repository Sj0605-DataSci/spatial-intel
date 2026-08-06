"""RDDL-driven scene decomposition for the clean Santa game.

Parses the game's instance.rddl for every object's TYPE and CELL, maps each cell
to a pixel center via the known 8x8 geometry, SAM2-masks each object at its cell
(point prompt -> clean labeled mask, no over-segmentation), and LaMa-inpaints the
union of all masks into one clean object-free BACKGROUND layer.

Output (runs/decompose_rddl/):
  layers/{type}__{name}.png   RGBA sprite per object (tight crop)
  background.png              clean board, all objects removed
  layers.json                 [{name, type, cell, bbox, movable}]
  overlay.png                 masks drawn on the frame (verify)

    python scripts/decompose_rddl.py \
        --rddl "game_ref/New Folder With Items/instance.rddl" \
        --frame game_ref/New\\ Folder\\ With\\ Items/scene.png \
        --out runs/decompose_rddl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.rl.policy import _load_hf_token_from_env_file

# known geometry of this 8x8 board (solved earlier from scene.png)
ORIGIN = (227, 227); CELL = 87.5
PLAYER_TYPES = {"santa_class", "player"}   # the movable character


def parse_rddl(rddl_path):
    """-> {name: {'type':t, 'col':c, 'row':r}} from `<type>-x(name, xC)` / `-y(...)`."""
    text = Path(rddl_path).read_text()
    objs = {}
    for kind, name, axis, num in re.findall(
            r"([a-z_]+)-([xy])\(([a-z0-9_]+),\s*[xy](\d+)\)", text):
        # note: regex groups are (type, name, ...) — fix below
        pass
    # explicit two-pass: capture type, name, axisletter, index
    for m in re.finditer(r"([a-z_]+)-([xy])\(\s*([a-z0-9_]+)\s*,\s*([xy])(\d+)\s*\)", text):
        typ, axis, name, _al, idx = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        o = objs.setdefault(name, {"type": typ})
        o["col" if axis == "x" else "row"] = idx
    # keep only fully-located objects
    return {n: o for n, o in objs.items() if "col" in o and "row" in o}


def cell_center(col, row):
    return (ORIGIN[0] + (col - 1) * CELL, ORIGIN[1] + (row - 1) * CELL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rddl", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", default="runs/decompose_rddl")
    args = ap.parse_args()

    _load_hf_token_from_env_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); (out / "layers").mkdir(parents=True, exist_ok=True)

    objs = parse_rddl(args.rddl)
    print(f"RDDL objects ({len(objs)}):")
    for n, o in objs.items():
        print(f"  {o['type']:14} {n:16} cell ({o['col']},{o['row']})"
              f"{'  <- MOVABLE' if o['type'] in PLAYER_TYPES else ''}")

    img = Image.open(args.frame).convert("RGB")
    arr = np.asarray(img)

    # SAM2: one point prompt per object at its cell center
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-large",
                                      torch_dtype=torch.float32).to(device).eval()

    union = np.zeros(arr.shape[:2], bool)
    meta = []
    for name, o in objs.items():
        cx, cy = cell_center(o["col"], o["row"])
        inputs = proc(images=img, input_points=[[[[float(cx), float(cy)]]]],
                      input_labels=[[[1]]], return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs, multimask_output=False)
        mask = proc.post_process_masks(res.pred_masks, inputs["original_sizes"])[0][0, 0].bool().cpu().numpy()
        # guard: if SAM grabbed a huge region (bg), skip
        if mask.sum() > 0.15 * mask.size or mask.sum() < 80:
            print(f"  ! {name}: mask {int(mask.sum())}px looks wrong, skipping")
            continue
        union |= mask
        ys, xs = np.where(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        rgba[..., :3] = arr[y0:y1, x0:x1]
        rgba[..., 3] = mask[y0:y1, x0:x1] * 255
        Image.fromarray(rgba).save(out / "layers" / f"{o['type']}__{name}.png")
        meta.append(dict(name=name, type=o["type"], cell=[o["col"], o["row"]],
                         bbox=[x0, y0, x1, y1], movable=o["type"] in PLAYER_TYPES))
    del model; torch.cuda.empty_cache()
    print(f"  masked {len(meta)} objects")

    # LaMa: clean background from union of all object masks
    from simple_lama_inpainting import SimpleLama
    from scipy import ndimage
    lama = SimpleLama()
    union_d = ndimage.binary_dilation(union, iterations=6)
    bg = lama(img, Image.fromarray((union_d * 255).astype(np.uint8))).resize(img.size).convert("RGB")
    bg.save(out / "background.png")

    # overlay preview
    prev = arr.copy()
    for m in meta:
        x0, y0, x1, y1 = m["bbox"]
        c = (0, 255, 255) if m["movable"] else (255, 200, 0)
        prev[y0:y0 + 2, x0:x1] = c; prev[y1 - 2:y1, x0:x1] = c
        prev[y0:y1, x0:x0 + 2] = c; prev[y0:y1, x1 - 2:x1] = c
    Image.fromarray(prev).save(out / "overlay.png")

    (out / "layers.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}/layers/  ({len(meta)} labeled sprite layers)")
    print(f"-> {out}/background.png  (clean board)")
    print(f"-> {out}/overlay.png  (verify)")


if __name__ == "__main__":
    main()
