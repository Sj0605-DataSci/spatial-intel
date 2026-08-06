"""Quantify what the bundle WM gets RIGHT (beyond eyeballing blur):

  1. ACTION applied? For each val transition, the model should predict obs_next,
     NOT copy obs_t. We measure, in the region that actually changed
     (|obs_next - obs_t| > 0), whether the prediction matches obs_next better than
     obs_t. 'change-region accuracy' = fraction where pred is closer to the true
     next state than to the (copied) input.

  2. HUD correct? The HUD is the top strip. We compare pred-HUD vs GT-next-HUD
     (L1) and also whether pred-HUD matches the NEXT hud (not the old one) when
     the HUD changed.

Writes a per-game table to the project dir (not /tmp).

    python scripts/measure_bundle_wm.py --ckpt runs/bundle_wm_1k/denoiser_final.pt \
        --res 1024 --per-game 15 --out runs/bundle_wm_1k_eval
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
from PIL import Image

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
HUD_FRAC = 0.12   # top ~12% of frame is the HUD strip


def arr(im, size):
    return np.asarray(im.convert("RGB").resize(size)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bundle_wm_1k/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--per-game", type=int, default=15)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--out", default="runs/bundle_wm_1k_eval")
    ap.add_argument("--cond-channels", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    by_game = defaultdict(list)
    for r in idx["val"]:
        if r["moved"]:
            by_game[r["game"]].append(r)

    vae = load_vae(device)
    net = make_latent_denoiser(device, num_actions=N_ACTIONS, cond_channels=args.cond_channels)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    rng = np.random.default_rng(0)
    stats = {}
    for g in sorted(by_game):
        rows = by_game[g]
        pick = [rows[i] for i in rng.choice(len(rows), min(args.per_game, len(rows)), replace=False)]
        change_acc, hud_next_better, hud_l1 = [], [], []
        for r in pick:
            obs_t = Image.open(BUNDLE / r["obs_t"])
            gt = Image.open(BUNDLE / r["obs_next"])
            size = gt.size
            a_t = arr(obs_t, size); a_n = arr(gt, size)
            lat = encode(vae, obs_t, args.res, device)
            act = torch.tensor([[r["action"]]], device=device)
            torch.manual_seed(0)
            with torch.no_grad():
                pred_lat, _ = sampler.sample(lat.unsqueeze(1), act)
            a_p = arr(decode(vae, pred_lat), size)

            # (1) change-region accuracy: where the true frame changed, is pred
            #     closer to next than to input?
            diff = np.abs(a_n - a_t).mean(-1)
            region = diff > 15                        # pixels that actually changed
            if region.sum() > 50:
                d_next = np.abs(a_p - a_n).mean(-1)[region].mean()
                d_inp = np.abs(a_p - a_t).mean(-1)[region].mean()
                change_acc.append(float(d_next < d_inp))

            # (2) HUD: top strip
            hh = int(size[1] * HUD_FRAC)
            hud_p, hud_n, hud_t = a_p[:hh], a_n[:hh], a_t[:hh]
            hud_l1.append(np.abs(hud_p - hud_n).mean())
            if np.abs(hud_n - hud_t).mean() > 1.0:    # HUD changed this step
                hud_next_better.append(float(
                    np.abs(hud_p - hud_n).mean() < np.abs(hud_p - hud_t).mean()))
        stats[g] = dict(
            change_acc=float(np.mean(change_acc)) if change_acc else None,
            hud_l1=float(np.mean(hud_l1)),
            hud_next_better=float(np.mean(hud_next_better)) if hud_next_better else None,
            n=len(pick))

    # table
    print(f"\n{'game':22} {'change-acc':>11} {'HUD L1':>8} {'HUD->next':>10}  n")
    for g, s in stats.items():
        ca = f"{s['change_acc']*100:.0f}%" if s['change_acc'] is not None else "  -"
        hn = f"{s['hud_next_better']*100:.0f}%" if s['hud_next_better'] is not None else "static"
        print(f"{g:22} {ca:>11} {s['hud_l1']:>8.1f} {hn:>10}  {s['n']}")
    allca = [s['change_acc'] for s in stats.values() if s['change_acc'] is not None]
    print(f"\nOVERALL change-region accuracy: {np.mean(allca)*100:.0f}% "
          f"(pred matches NEXT state better than copied input)")
    (out / "measure.json").write_text(json.dumps(stats, indent=2))
    print(f"-> {out}/measure.json")


if __name__ == "__main__":
    main()
