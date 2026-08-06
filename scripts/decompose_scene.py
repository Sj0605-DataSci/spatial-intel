"""Prove general scene decomposition on ONE frame: split into sprite layers + clean bg.

  SAM2 (point-grid 'segment everything')  -> a mask per object/sprite
  filter/merge masks (drop tiny + near-duplicate)  -> clean sprite set
  LaMa inpaint (union of all sprite masks)  -> one object-free BACKGROUND layer
  each sprite saved as tight RGBA + its cell/bbox

No game-specific geometry or colors — works on any frame. This is the general
alternative to the Santa-only plate hacks.

    python scripts/decompose_scene.py \
        --frame runs/diamond_dev/playthrough_demo/step00_start.png \
        --out runs/decompose_santa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.rl.policy import _load_hf_token_from_env_file


def sam2_everything(frame_path, device, grid=24, min_area=400, max_area_frac=0.25):
    """Point-grid SAM2 -> list of boolean masks (one per object)."""
    from transformers import Sam2Model, Sam2Processor
    mid = "facebook/sam2-hiera-large"
    proc = Sam2Processor.from_pretrained(mid)
    model = Sam2Model.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    # uniform grid of query points across the image
    xs = np.linspace(W * 0.06, W * 0.94, grid)
    ys = np.linspace(H * 0.06, H * 0.94, grid)
    pts = [[float(x), float(y)] for y in ys for x in xs]

    masks = []
    B = 64  # batch points to keep memory bounded
    for i in range(0, len(pts), B):
        chunk = pts[i:i + B]
        inputs = proc(images=img, input_points=[[[p] for p in chunk]],
                      input_labels=[[[1]] * len(chunk)], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, multimask_output=False)
        m = proc.post_process_masks(out.pred_masks, inputs["original_sizes"])[0]  # (n,1,H,W)
        for j in range(m.shape[0]):
            masks.append(m[j, 0].bool().cpu().numpy())
    del model; torch.cuda.empty_cache()

    # filter: drop tiny and huge (background) masks
    area = W * H
    kept = [mk for mk in masks if min_area <= mk.sum() <= max_area_frac * area]
    # dedup near-identical masks (IoU > 0.85)
    uniq = []
    for mk in sorted(kept, key=lambda a: -a.sum()):
        dup = False
        for u in uniq:
            inter = (mk & u).sum()
            if inter / (mk | u).sum() > 0.85:
                dup = True; break
        if not dup:
            uniq.append(mk)
    return img, uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--out", default="runs/decompose_out")
    args = ap.parse_args()

    _load_hf_token_from_env_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "sprites").mkdir(exist_ok=True)

    print(f"SAM2 segment-everything ({args.grid}x{args.grid} point grid)...")
    img, masks = sam2_everything(args.frame, device, grid=args.grid)
    print(f"  {len(masks)} sprite layers after filter/dedup")

    arr = np.asarray(img)
    union = np.zeros(arr.shape[:2], bool)
    meta = []
    for i, mk in enumerate(masks):
        union |= mk
        ys, xs = np.where(mk)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        rgba[..., :3] = arr[y0:y1, x0:x1]
        rgba[..., 3] = mk[y0:y1, x0:x1] * 255
        Image.fromarray(rgba).save(out / "sprites" / f"sprite_{i:02d}.png")
        meta.append(dict(id=i, bbox=[x0, y0, x1, y1], area=int(mk.sum())))

    # LaMa: inpaint the union of all sprites -> clean background layer
    print("LaMa: inpainting all sprite holes -> clean background...")
    from simple_lama_inpainting import SimpleLama
    from scipy import ndimage
    lama = SimpleLama()
    union_d = ndimage.binary_dilation(union, iterations=6)
    bg = lama(img, Image.fromarray((union_d * 255).astype(np.uint8)))
    bg = bg.resize(img.size).convert("RGB")
    bg.save(out / "background.png")

    # overlay preview: color every sprite mask
    prev = arr.copy()
    rng = np.random.default_rng(0)
    for mk in masks:
        c = rng.integers(60, 255, 3)
        edge = mk ^ ndimage.binary_erosion(mk, iterations=2)
        prev[edge] = c
    Image.fromarray(prev).save(out / "layers_overlay.png")

    (out / "layers.json").write_text(json.dumps(meta, indent=2))
    print(f"-> {out}/sprites/       ({len(masks)} sprite RGBA layers)")
    print(f"-> {out}/background.png  (clean object-free layer)")
    print(f"-> {out}/layers_overlay.png (verify object coverage)")


if __name__ == "__main__":
    main()
