"""Next-frame via DIFFUSION INPAINT (sd-turbo, no new download).

World-model step: given frame_t + action, mask the OLD S cell + NEW S cell and let
sd-turbo inpaint them — synthesise clean snow where S left, and (attempt to) render
S at the new cell. Then game-logic detects the resulting S position.

Compared to cut-paste, inpaint should give a seamless background fill; the open
question is whether it correctly renders S in the new cell (the accuracy axis).

    python scripts/move_s_inpaint.py --move right --out runs/move_inpaint
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw

from spatial_intel.game.geometry import SceneGeom, apply_move

SCENE = "game_ref/New Folder With Items/scene.png"


def cell_mask(geom, cells, size, pad=6):
    """White mask over the given cells (list of (cx,cy)), on a `size`x`size` canvas."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    for (cx, cy) in cells:
        l, t, r, b = geom.cell_bbox(cx, cy)
        d.rectangle([l - pad, t - pad, r + pad, b + pad], fill=255)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", choices=["left", "right", "up", "down"], default="right")
    ap.add_argument("--s-cell", type=int, nargs=2, default=[1, 1])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default="runs/move_inpaint")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    geom = SceneGeom()
    frame = Image.open(SCENE).convert("RGB")
    size = frame.size[0]
    s_cell = tuple(args.s_cell)
    new_cell = apply_move(s_cell, args.move, geom)
    print(f"move '{args.move}': S {s_cell} -> {new_cell}")

    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    from diffusers import AutoPipelineForInpainting
    print("loading sd-turbo inpaint...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # sd-turbo is 512-native; work at 512 then map back
    work = 512
    fr = frame.resize((work, work))
    gscale = work / size
    g2 = SceneGeom(origin_x=geom.origin_x*gscale, origin_y=geom.origin_y*gscale,
                   cell=geom.cell*gscale, n=geom.n)
    mask = cell_mask(g2, [s_cell, new_cell], work, pad=4)

    prompt = ("a red circular badge with white letter S on snowy ground, "
              "top-down pixel-art christmas game grid, clean snow")
    # warm-up
    _ = pipe(prompt=prompt, image=fr, mask_image=mask, num_inference_steps=args.steps,
             guidance_scale=0.0, strength=0.99,
             generator=torch.Generator("cuda").manual_seed(0)).images[0]
    torch.cuda.synchronize()
    t0 = time.time()
    result = pipe(prompt=prompt, image=fr, mask_image=mask, num_inference_steps=args.steps,
                  guidance_scale=0.0, strength=0.99,
                  generator=torch.Generator("cuda").manual_seed(1)).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0

    result = result.resize((size, size))
    result.save(out / f"next_{args.move}.png")
    mask.resize((size, size)).save(out / "mask.png")
    print(f"inpaint time: {dt:.3f}s | saved {out}/next_{args.move}.png")


if __name__ == "__main__":
    main()
