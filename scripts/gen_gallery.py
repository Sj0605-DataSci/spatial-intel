"""Generate prediction-gallery figures for the report: per game, a few
(input | predicted | ground-truth) strips from TRAIN and VAL, saved as
individual PNGs into report/figs/gallery/ for LaTeX.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
OUT = Path(__file__).resolve().parents[1] / "report" / "figs" / "gallery"
N_ACTIONS = 5
ID_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}
P = 300


def fnt(sz, b=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def strip(game, split, i, obs_t, pred, gt, action):
    W, H = P * 3 + 24, P + 46
    cv = Image.new("RGB", (W, H), (17, 24, 33))
    d = ImageDraw.Draw(cv)
    d.text((8, 6), f"{game}  [{split}]  action={ID_KEY[action]}", (231, 238, 245), fnt(15, True))
    for k, (im, cap, col) in enumerate([(obs_t, "input", (140, 160, 180)),
                                        (pred, "predicted", (198, 40, 40)),
                                        (gt, "ground truth", (140, 160, 180))]):
        x = 8 + k * (P + 4)
        cv.paste(im.convert("RGB").resize((P, P)), (x, 30))
        d.text((x, 30 + P + 1), cap, col, fnt(12))
    cv.save(OUT / f"{game}_{split}_{i}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bundle_sharp/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--per", type=int, default=2, help="samples per game per split")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = json.loads((BUNDLE / "wm_index.json").read_text())

    vae = load_vae(device)
    net = make_latent_denoiser(device, num_actions=N_ACTIONS, cond_channels=args.cond_channels)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    rng = np.random.default_rng(1)
    for split in ["train", "val"]:
        by_game = defaultdict(list)
        for r in idx[split]:
            if r["moved"]:
                by_game[r["game"]].append(r)
        for g in sorted(by_game):
            rows = by_game[g]
            for j, ri in enumerate(rng.choice(len(rows), min(args.per, len(rows)), replace=False)):
                r = rows[ri]
                obs_t = Image.open(BUNDLE / r["obs_t"])
                gt = Image.open(BUNDLE / r["obs_next"])
                lat = encode(vae, obs_t, args.res, device)
                torch.manual_seed(0)
                with torch.no_grad():
                    pl, _ = sampler.sample(lat.unsqueeze(1),
                                           torch.tensor([[r["action"]]], device=device))
                pred = decode(vae, pl)
                strip(g, split, j, obs_t, pred, gt, r["action"])
            print(f"{split} {g}: {min(args.per,len(rows))} strips", flush=True)
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
