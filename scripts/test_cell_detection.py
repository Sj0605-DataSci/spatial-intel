"""Crux test for the WM-in-the-loop hybrid: can we read the ARRIVAL CELL out of
the WM's blurry prediction reliably?

For val transitions: WM predicts next frame; we diff prediction vs input, take the
changed-region centroid, snap to a cell via the calibrated geometry, and compare
to the TRUE arrival cell (states.jsonl px,py of obs_next). Reports cell-accuracy
per game. If high, the hybrid works: paste the real sprite at the detected cell.
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
from hybrid_geometry import pixel_to_cell

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5


def state_index(game):
    return {json.loads(l)["id"]: json.loads(l) for l in open(BUNDLE / game / "states.jsonl")}


def arrival_cell_from_pred(pred_rgb, inp_rgb, geom):
    """Changed-region centroid of (pred vs input) -> cell."""
    a = np.asarray(inp_rgb.convert("RGB")).astype(int)
    b = np.asarray(pred_rgb.convert("RGB").resize(inp_rgb.size)).astype(int)
    d = np.abs(a - b).sum(-1)
    ys, xs = np.where(d > 30)
    if len(xs) < 15:
        return None
    return pixel_to_cell(geom, xs.mean(), ys.mean())


def obs_id(path):
    return int(Path(path).stem.split("_")[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bundle_sharp/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--per-game", type=int, default=12)
    ap.add_argument("--denoise-steps", type=int, default=20)
    ap.add_argument("--cond-channels", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    overall_hit = 0; overall_n = 0
    print(f"{'game':22} {'cell-acc':>9} {'mean-dist':>10}  n")
    for g in sorted(by_game):
        geom = json.loads((BUNDLE / g / "geometry.json").read_text())
        states = state_index(g)
        rows = by_game[g]
        pick = [rows[i] for i in rng.choice(len(rows), min(args.per_game, len(rows)), replace=False)]
        hits, dists = [], []
        for r in pick:
            inp = Image.open(BUNDLE / r["obs_t"])
            lat = encode(vae, inp, args.res, device)
            act = torch.tensor([[r["action"]]], device=device)
            torch.manual_seed(0)
            with torch.no_grad():
                pl, _ = sampler.sample(lat.unsqueeze(1), act)
            pred = decode(vae, pl)
            det = arrival_cell_from_pred(pred, inp, geom)
            true = states[obs_id(r["obs_next"])]
            tc = (true["px"], true["py"])
            if det is not None:
                hits.append(det == tc)
                dists.append(np.hypot(det[0] - tc[0], det[1] - tc[1]))
        acc = np.mean(hits) if hits else 0
        overall_hit += sum(hits); overall_n += len(hits)
        print(f"{g:22} {acc*100:>7.0f}% {np.mean(dists) if dists else 0:>9.2f}  {len(pick)}")
    print(f"\nOVERALL arrival-cell accuracy: {100*overall_hit/max(overall_n,1):.0f}% "
          f"(can we read the moved cell from the blur -> hybrid viability)")


if __name__ == "__main__":
    main()
