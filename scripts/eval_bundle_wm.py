"""Honest eval of the bundle world model: decode predictions on HELD-OUT val
transitions across all games.

For sampled val rows: encode obs_t -> sample next latent (given action) -> VAE
decode -> compare to the real obs_next. Builds a per-game contact sheet:
   [ obs_t (input) | DECODED prediction | ground-truth obs_next ]
labeled with game + action, so we can SEE whether low loss = correct frames.

    python scripts/eval_bundle_wm.py --ckpt runs/bundle_wm/denoiser_final.pt \
        --per-game 2 --out runs/bundle_wm_eval
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
N_ACTIONS = 5
ID_TO_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}
PANEL = 240
BG = (14, 22, 32); INK = (231, 238, 245); BADGE = (198, 40, 40); SLATE = (95, 114, 133)


def fnt(sz, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bundle_wm/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--out", default="runs/bundle_wm_eval")
    ap.add_argument("--cond-channels", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    val = idx["val"]
    # pick per-game samples (prefer transitions where the player moved)
    by_game = defaultdict(list)
    for r in val:
        by_game[r["game"]].append(r)
    rng = np.random.default_rng(0)
    picks = []
    for g in sorted(by_game):
        rows = [r for r in by_game[g] if r["moved"]] or by_game[g]
        for i in rng.choice(len(rows), size=min(args.per_game, len(rows)), replace=False):
            picks.append(rows[i])

    vae = load_vae(device)
    net = make_latent_denoiser(device, num_actions=N_ACTIONS, cond_channels=args.cond_channels)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    rows_img = []
    l1s = []
    for r in picks:
        obs_t = Image.open(BUNDLE / r["obs_t"])
        gt = Image.open(BUNDLE / r["obs_next"]).convert("RGB")
        lat = encode(vae, obs_t, args.res, device)
        act = torch.tensor([[r["action"]]], device=device)
        torch.manual_seed(0)
        with torch.no_grad():
            pred_lat, _ = sampler.sample(lat.unsqueeze(1), act)
        pred = decode(vae, pred_lat).convert("RGB")
        # pixel L1 predicted vs GT (proxy quality)
        a = np.asarray(pred.resize(gt.size)).astype(float)
        b = np.asarray(gt).astype(float)
        l1s.append(np.abs(a - b).mean())

        # compose a labeled row: input | predicted | ground-truth
        strip = Image.new("RGB", (PANEL * 3 + 16, PANEL + 40), BG)
        d = ImageDraw.Draw(strip)
        d.text((8, 6), f"{r['game']}   action={ID_TO_KEY[r['action']]}   "
                       f"moved={'yes' if r['moved'] else 'no'}", INK, fnt(15, True))
        for k, (im, cap, col) in enumerate([
                (obs_t.convert("RGB"), "input", SLATE),
                (pred, "PREDICTED", BADGE),
                (gt, "ground truth", SLATE)]):
            x = 8 + k * (PANEL + 4)
            strip.paste(im.resize((PANEL, PANEL)), (x, 32))
            d.text((x, 32 + PANEL + 2), cap, col, fnt(12))
        rows_img.append(strip)

    # stack all rows into one contact sheet
    W = rows_img[0].width
    H = sum(im.height for im in rows_img)
    sheet = Image.new("RGB", (W, H), BG)
    y = 0
    for im in rows_img:
        sheet.paste(im, (0, y)); y += im.height
    sheet.save(out / "val_predictions.png")

    print(f"decoded {len(picks)} held-out val transitions across {len(by_game)} games")
    print(f"mean pixel L1 (pred vs GT): {np.mean(l1s):.1f}/255")
    print(f"-> {out}/val_predictions.png")


if __name__ == "__main__":
    main()
