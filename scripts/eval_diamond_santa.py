"""Eval a trained DIAMOND denoiser on the 6 Santa moves (the honest world-model test).

For each move: feed the before-frame + action into DIAMOND's EDM sampler, generate
the predicted next frame, detect where the S badge lands, score hit vs target cell.
Saves before|predicted|target triptychs so we can SEE what it learned.

    python scripts/eval_diamond_santa.py --ckpt runs/diamond_santa/denoiser_final.pt --res 64
"""
import argparse
import json
import sys
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from train_diamond_santa import make_denoiser, load_img, ACTIONS, DATA

ORIGIN = (227, 227); CELL = 87.5  # in 1024px space


def santa_cell_from_arr(a1024):
    """Largest bright-red blob -> grid cell (1-indexed), on a 1024x1024 uint8 array."""
    from scipy import ndimage
    R, G, B = a1024[..., 0].astype(int), a1024[..., 1].astype(int), a1024[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
    col = round((xs.mean() - ORIGIN[0]) / CELL) + 1
    row = round((ys.mean() - ORIGIN[1]) / CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def to_uint8(chw):  # (C,H,W) [-1,1] -> HWC uint8
    a = chw.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--out", default="runs/eval_diamond_santa")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    net = make_denoiser(args.res, device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    man = json.loads((DATA / "manifest.json").read_text())
    # 6 unique moves (test split == the 6)
    seen, rows = set(), []
    for r in man["test"]:
        k = (r["direction"], tuple(r["start"]), tuple(r["target"]))
        if k not in seen:
            seen.add(k); rows.append(r)

    n_hit = 0
    results = []
    for i, r in enumerate(rows):
        start, target = tuple(r["start"]), tuple(r["target"])
        before = load_img(DATA / r["input"], args.res, device)          # (C,H,W)
        prev_obs = before.unsqueeze(0).unsqueeze(0)                     # (1,1,C,H,W)
        act = torch.tensor([[ACTIONS[r["direction"]]]], device=device)  # (1,1)
        with torch.no_grad():
            pred, _ = sampler.sample(prev_obs, act)                     # (1,C,H,W)
        pred_arr = to_uint8(pred[0])
        # upscale prediction to 1024 for the geometry-based detector
        pred_1024 = np.asarray(Image.fromarray(pred_arr).resize((1024, 1024), Image.NEAREST))
        det = santa_cell_from_arr(pred_1024)
        hit = (det == target); n_hit += hit

        # triptych: before | predicted | ground-truth target
        b_im = Image.fromarray(to_uint8(before)).resize((256, 256))
        p_im = Image.fromarray(pred_arr).resize((256, 256))
        t_im = Image.open(DATA / r["target_img"]).convert("RGB").resize((256, 256))
        strip = Image.new("RGB", (768, 256))
        strip.paste(b_im, (0, 0)); strip.paste(p_im, (256, 0)); strip.paste(t_im, (512, 0))
        strip.save(out / f"eval_{i}_{r['direction']}_{start[0]}{start[1]}_to_{target[0]}{target[1]}.png")

        results.append(dict(direction=r["direction"], start=list(start),
                            target=list(target), detected=list(det) if det else None, hit=hit))
        print(f"[{i}] {r['direction']:5} {start}->{target}: predicted S at {det} "
              f"{'✓ HIT' if hit else '✗ miss'}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== DIAMOND SANTA EVAL: {n_hit}/{len(rows)} moves correct ===")
    print(f"-> {out}/  (each png = before | DIAMOND-predicted | target)")


if __name__ == "__main__":
    main()
