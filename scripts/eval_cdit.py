"""Eval CDiT + Self-Flow: flow-matching sampling -> VAE decode -> images + SSIM/LPIPS.

Flow-matching sampler: integrate the velocity field from noise (t=1) to data (t=0)
with Euler steps. x_{t-dt} = x_t - v_pred * dt, using the model's predicted velocity.
(Self-Flow's per-token timesteps are a TRAINING trick; at inference we use a single
shared t per step, ramped 1->0.)

Saves viewable input/predicted/groundtruth per sample + summary metrics.

    python scripts/eval_cdit.py --ckpt runs/cdit_hospital/cdit_final.pt --game hospital \
        --n 8 --out runs/eval_cdit_hospital
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
import lpips as lpips_lib

from latent_wm import load_vae, encode, decode
from cdit_selfflow import CDiTSelfFlow, LAT_CH, LAT

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5


def to_t(pil, size, device):
    a = np.asarray(pil.convert("RGB").resize(size)).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def sample(net, cur, act, steps=30):
    """Flow-matching Euler integration from noise (t=1) to data (t=0)."""
    b = cur.size(0)
    x = torch.randn(b, LAT_CH, LAT, LAT, device=cur.device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=cur.device)
    for i in range(steps):
        t = ts[i]
        t_tok = torch.full((b, net.n_tok), float(t), device=cur.device)
        v, _ = net(x, cur, act, t_tok, return_distill=False)   # predicted velocity
        dt = ts[i] - ts[i + 1]
        x = x - v * dt                                         # move toward data
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/eval_cdit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    (out / "samples").mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    net = CDiTSelfFlow(dim=args.dim, depth=args.depth).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)); net.eval()

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    rows = [r for r in idx["val"] if r["game"] == args.game and r["moved"]][:args.n]

    ssims, lps = [], []
    for i, r in enumerate(rows):
        obs_t = Image.open(BUNDLE / r["obs_t"]).convert("RGB")
        gt = Image.open(BUNDLE / r["obs_next"]).convert("RGB")
        cur = encode(vae, obs_t, args.res, device).float()
        act = torch.tensor([r["action"]], device=device)
        pred_lat = sample(net, cur, act, steps=args.steps)
        # decode expects normalized latent; CDiT trained on the normalized latent
        pred = decode(vae, pred_lat)
        sz = gt.size
        tp, tg = to_t(pred, sz, device), to_t(gt, sz, device)
        ssims.append(ssim_fn(tp, tg, data_range=1.0).item())
        with torch.no_grad():
            lps.append(lpips_fn(tp * 2 - 1, tg * 2 - 1).item())
        sd = out / "samples" / f"{i:02d}"
        sd.mkdir(exist_ok=True)
        obs_t.save(sd / "1_input.png"); pred.convert("RGB").save(sd / "2_predicted.png")
        gt.save(sd / "3_groundtruth.png")

    print(f"\n=== CDiT+Self-Flow eval — {args.game} ===")
    print(f"  SSIM {np.mean(ssims):.4f}  LPIPS {np.mean(lps):.4f}  (n={len(rows)})")
    print(f"-> {out}/samples/")


if __name__ == "__main__":
    main()
