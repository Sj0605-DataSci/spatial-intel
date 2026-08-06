"""Train the REAL DIAMOND denoiser (eloialonso/diamond) on our 6 Santa moves.

Not a reimplementation — we import DIAMOND's actual `Denoiser` / `InnerModel`
(EDM diffusion world model, NeurIPS 2024) and feed it our data. Each of the 6
real moves is a 2-frame episode [before, after]; with num_steps_conditioning=1
the model learns: (before frame, action) -> after frame. This is the honest
"can a purpose-built world model even memorize these 6 moves?" test.

Frames downscaled 1024 -> RES px (DIAMOND is a tiny conv UNet meant for small
frames, e.g. 64px Atari). Actions: up/down/left/right -> {0,1,2,3}.

    python scripts/train_diamond_santa.py --steps 3000 --res 64 --out runs/diamond_santa
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# import DIAMOND's real model code
DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.diffusion.denoiser import (
    Denoiser, DenoiserConfig, SigmaDistributionConfig,
)
from models.diffusion.inner_model import InnerModelConfig

ACTIONS = {"up": 0, "down": 1, "left": 2, "right": 3}
DATA = Path(__file__).resolve().parents[1] / "data" / "santa_real"


@dataclass
class Batch:
    """Minimal stand-in for DIAMOND's data.Batch (only fields Denoiser.forward uses)."""
    obs: torch.Tensor          # (B, T, C, H, W) in [-1, 1]
    act: torch.Tensor          # (B, T) long
    mask_padding: torch.Tensor  # (B, T) bool


def load_img(path, res, device):
    im = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    a = torch.from_numpy(np.asarray(im).copy()).float().div(127.5).sub(1.0)
    return a.permute(2, 0, 1).to(device)  # (C,H,W)


def build_episodes(res, device):
    """6 unique moves -> list of (obs (2,C,H,W), act (2,))."""
    man = json.loads((DATA / "manifest.json").read_text())
    seen, eps = set(), []
    for r in man["train"]:
        key = (r["direction"], tuple(r["start"]), tuple(r["target"]))
        if key in seen:
            continue
        seen.add(key)
        before = load_img(DATA / r["input"], res, device)
        after = load_img(DATA / r["target_img"], res, device)
        obs = torch.stack([before, after], 0)          # (2,C,H,W)
        # act[t] is the action taken AT frame t leading to frame t+1.
        # frame0=before -> action -> frame1=after ; act[1] unused (no frame2).
        act = torch.tensor([ACTIONS[r["direction"]], 0], device=device)
        eps.append((obs, act, r["direction"], tuple(r["start"]), tuple(r["target"])))
    return eps


def make_denoiser(res, device):
    # tiny UNet, DIAMOND Atari-style knobs scaled for a 6-sample toy
    inner = InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=1,
        cond_channels=256,
        depths=[2, 2, 2],
        channels=[64, 96, 128],
        attn_depths=[False, False, True],
        num_actions=len(ACTIONS),
    )
    cfg = DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3)
    d = Denoiser(cfg).to(device)
    d.setup_training(SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=6, help="episodes per step (<=6)")
    ap.add_argument("--out", default="runs/diamond_santa")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seconds", type=int, default=0, help="stop after N seconds (0=off, Karpathy 20min budget)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    eps = build_episodes(args.res, device)
    print(f"episodes: {len(eps)} unique moves @ {args.res}px")
    for _, _, mv, b, t in eps:
        print(f"  {mv:6} {b}->{t}")

    obs = torch.stack([e[0] for e in eps], 0)   # (6,2,C,H,W)
    act = torch.stack([e[1] for e in eps], 0)   # (6,2)
    mask = torch.ones(len(eps), 2, dtype=torch.bool, device=device)

    net = make_denoiser(args.res, device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DIAMOND denoiser params: {n_params/1e6:.2f}M | device={device}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    log, t0 = [], time.time()
    for step in range(args.steps):
        idx = rng.choice(len(eps), size=min(args.batch, len(eps)), replace=False)
        idx = torch.tensor(idx, device=device)
        batch = Batch(obs=obs[idx], act=act[idx], mask_padding=mask[idx])
        loss, _ = net(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        lv = float(loss.detach())
        log.append(dict(step=step, loss=lv, t=round(time.time() - t0, 1)))
        if step % 50 == 0 or step == args.steps - 1:
            med = float(np.median([r["loss"] for r in log[-50:]]))
            print(f"[{step:4d}/{args.steps}] loss={lv:.4f} (med50={med:.4f}) "
                  f"{time.time()-t0:.0f}s", flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        if args.save_every and step and step % args.save_every == 0:
            torch.save(net.state_dict(), out / f"denoiser_{step}.pt")
        if args.seconds and time.time() - t0 > args.seconds:
            print(f"hit {args.seconds}s budget at step {step}"); break

    torch.save(net.state_dict(), out / "denoiser_final.pt")
    print(f"done -> {out}/denoiser_final.pt  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
