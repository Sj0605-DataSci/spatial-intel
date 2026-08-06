"""ONE image + action -> next frame, varying ONLY the sampling, to isolate the blur.

Picks a single transition. Encodes obs_t, feeds action, and decodes the predicted
next frame under several samplers:
  - denoise steps: 5, 15, 30, 50, 100        (does more steps sharpen?)
  - deterministic (s_churn=0) vs stochastic  (does a mode-committing sampler sharpen?)
Saves a labeled grid + the input & ground-truth into the project dir.

If the player stays blurry across ALL of these, the blur is in the predicted
LATENT (conditional-mean), not the sampling.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
ID_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}


def fnt(sz, b=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bundle_sharp/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--game", default="7_worlds_of_wonder")
    ap.add_argument("--out", default="runs/one_pred_test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    # pick ONE val transition where the player moved, from the chosen game
    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    row = next(r for r in idx["val"] if r["game"] == args.game and r["moved"])
    obs_t = Image.open(BUNDLE / row["obs_t"]).convert("RGB")
    gt = Image.open(BUNDLE / row["obs_next"]).convert("RGB")
    action = row["action"]
    print(f"transition: {args.game}  action={ID_KEY[action]}  {row['obs_t']} -> {row['obs_next']}")

    vae = load_vae(device)
    net = make_latent_denoiser(device, num_actions=N_ACTIONS, cond_channels=args.cond_channels)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    lat = encode(vae, obs_t, args.res, device).unsqueeze(1)     # (1,1,C,h,w)
    act = torch.tensor([[action]], device=device)

    configs = [
        ("det 5",   dict(num_steps_denoising=5,   s_churn=0)),
        ("det 15",  dict(num_steps_denoising=15,  s_churn=0)),
        ("det 30",  dict(num_steps_denoising=30,  s_churn=0)),
        ("det 50",  dict(num_steps_denoising=50,  s_churn=0)),
        ("det 100", dict(num_steps_denoising=100, s_churn=0)),
        ("stoch 30", dict(num_steps_denoising=30, s_churn=20, s_tmin=0.05, s_tmax=5, s_noise=1.0)),
        ("stoch 50", dict(num_steps_denoising=50, s_churn=40, s_tmin=0.05, s_tmax=5, s_noise=1.0)),
    ]
    preds = []
    for name, cfg in configs:
        sampler = DiffusionSampler(net, DiffusionSamplerConfig(
            sigma_min=2e-3, sigma_max=20, rho=7, order=1, **cfg))
        torch.manual_seed(0)
        with torch.no_grad():
            pl, _ = sampler.sample(lat, act)
        preds.append((name, decode(vae, pl).convert("RGB")))
        print(f"  {name}: done")

    # grid: row of [input] [gt] [each pred]
    P = 256
    tiles = [("input", obs_t), ("ground truth", gt)] + preds
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * (P + 6) + 6, rows * (P + 26) + 6
    cv = Image.new("RGB", (W, H), (17, 24, 33))
    d = ImageDraw.Draw(cv)
    for i, (name, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = 6 + c * (P + 6), 6 + r * (P + 26)
        cv.paste(im.resize((P, P)), (x, y))
        col = (46, 125, 91) if name in ("input", "ground truth") else (198, 40, 40)
        d.text((x, y + P + 2), name, col, fnt(14, True))
    cv.save(out / "sampling_sweep.png")
    obs_t.save(out / "input.png"); gt.save(out / "ground_truth.png")
    print(f"-> {out}/sampling_sweep.png  (input | GT | 7 samplers)")


if __name__ == "__main__":
    main()
