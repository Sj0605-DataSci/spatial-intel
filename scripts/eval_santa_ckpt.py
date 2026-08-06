"""Eval a weighted-loss LoRA checkpoint on the 6 Santa moves (the honest test).

Load the LoRA, feed each before-frame + coordinate prompt, detect where the S
badge lands, score hit vs target cell. Uses data/santa_real test moves.

    python scripts/eval_santa_ckpt.py --ckpt runs/lora_santa_weighted/ckpt_600 --out runs/eval_santa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import Flux2KleinPipeline

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
DATA = Path("data/santa_real")
ORIGIN = (227, 227); CELL = 87.5


def santa_cell(img):
    """Largest bright-red blob -> grid cell (1-indexed)."""
    from scipy import ndimage
    a = np.asarray(img.convert("RGB").resize((1024, 1024)))
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
    col = round((xs.mean() - ORIGIN[0]) / CELL) + 1
    row = round((ys.mean() - ORIGIN[1]) / CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--out", default="runs/eval_santa")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((DATA / "manifest.json").read_text())
    exs = man["test"]

    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    wpath = Path(args.ckpt)
    if wpath.is_dir():
        wpath = wpath / "pytorch_lora_weights.safetensors"
    pipe.transformer.load_lora_adapter(load_file(str(wpath)), prefix=None)
    print(f"loaded LoRA: {wpath} | active: {pipe.transformer.active_adapters()}\n")

    results = []
    for i, ex in enumerate(exs):
        start, target = tuple(ex["start"]), tuple(ex["target"])
        src = Image.open(DATA / ex["input"]).convert("RGB").resize((512, 512))
        gen = torch.Generator("cuda").manual_seed(i)
        img = pipe(image=src, prompt=ex["prompt_coords"], num_inference_steps=args.steps,
                   guidance_scale=args.guidance, height=512, width=512, generator=gen).images[0]
        det = santa_cell(img)
        hit = (det == target)
        img.save(out / f"eval_{i}_{ex['direction']}_{start[0]}{start[1]}_to_{target[0]}{target[1]}.png")
        results.append(dict(direction=ex["direction"], start=list(start), target=list(target),
                            detected=list(det) if det else None, hit=hit))
        print(f"[{i}] {ex['direction']:5} {start}->{target}: detected {det} "
              f"{'✓ HIT' if hit else '✗ miss'}", flush=True)

    n_hit = sum(r["hit"] for r in results)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== SANTA LoRA EVAL: {n_hit}/{len(results)} moves correct ===")
    print(f"-> {out}/  (open eval_*.png)")


if __name__ == "__main__":
    main()
