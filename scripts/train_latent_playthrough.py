"""Train a LATENT-space DIAMOND world model on the Santa playthrough.

Frames are encoded once with FLUX's VAE (1024px -> 128x128x32 latent), then the
DIAMOND denoiser learns (latent, action) -> next latent in that space. Output is
decoded back to CRISP 1024px — no pixelation, no attention blowup.

    python scripts/train_latent_playthrough.py --steps 4000 --out runs/latent_play
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from latent_wm import load_vae, encode, make_latent_denoiser, ACTIONS
from train_diamond_santa import Batch
from train_diamond_playthrough import parse_playthrough


def build_latent_episodes(vae, res, device):
    """Encode each playthrough transition -> (obs_lat (2,32,h,w), act (2,))."""
    trans = parse_playthrough()
    eps, meta = [], []
    for before_png, act, bc, ac, after_png in trans:
        lb = encode(vae, Image.open(before_png), res, device)[0]   # (32,h,w)
        la = encode(vae, Image.open(after_png), res, device)[0]
        obs = torch.stack([lb, la], 0)                             # (2,32,h,w)
        a = torch.tensor([ACTIONS[act], 0], device=device)
        eps.append((obs, a))
        meta.append((act, bc, ac))
    return eps, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=1024, help="frame res fed to VAE")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--out", default="runs/latent_play")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    from latent_wm import VAE_BACKEND
    print(f"loading VAE (frozen): {VAE_BACKEND}...")
    vae = load_vae(device)
    eps, meta = build_latent_episodes(vae, args.res, device)
    lat_hw = tuple(eps[0][0].shape[-2:])
    print(f"playthrough: {len(eps)} transitions | latent {eps[0][0].shape[0]}x{lat_hw[0]}x{lat_hw[1]} "
          f"(from {args.res}px)")
    for (mv, b, t) in meta:
        print(f"  {b} --{mv:5}--> {t}")

    obs = torch.stack([e[0] for e in eps], 0)          # (N,2,32,h,w)
    act = torch.stack([e[1] for e in eps], 0)
    mask = torch.ones(len(eps), 2, dtype=torch.bool, device=device)

    # free VAE weights from GPU during training (we re-load for decode/eval)
    del vae; torch.cuda.empty_cache()

    net = make_latent_denoiser(device)
    print(f"latent denoiser: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params | {device}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    log, t0 = [], time.time()
    for step in range(args.steps):
        idx = torch.tensor(rng.choice(len(eps), size=min(args.batch, len(eps)), replace=False), device=device)
        loss, _ = net(Batch(obs=obs[idx], act=act[idx], mask_padding=mask[idx]))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        lv = float(loss.detach())
        log.append(dict(step=step, loss=lv, t=round(time.time() - t0, 1)))
        if step % 50 == 0 or step == args.steps - 1:
            med = float(np.median([r["loss"] for r in log[-50:]]))
            print(f"[{step:4d}/{args.steps}] loss={lv:.4f} (med50={med:.4f}) {time.time()-t0:.0f}s", flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        if args.save_every and step and step % args.save_every == 0:
            torch.save(net.state_dict(), out / f"denoiser_{step}.pt")

    torch.save(net.state_dict(), out / "denoiser_final.pt")
    print(f"done -> {out}/denoiser_final.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
