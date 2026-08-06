"""Train the real DIAMOND denoiser on the Santa PLAYTHROUGH demo
(runs/diamond_dev/playthrough_demo): a coherent 9-move path where Santa walks
(1,1) -> right x3 -> (4,1) -> down x6 -> (4,7), ending on house B.

Frames are named stepNN_<action>_<col>-<row>.png. Transition i is
(frame i-1, action-on-frame-i) -> frame i. Each is a 2-frame episode
[before, after] with num_steps_conditioning=1. Prompt/action = the move
direction (up/down/left/right) as DIAMOND's discrete action embedding.

Trained at higher native resolution (default 128px) for a crisp demo video.

    python scripts/train_diamond_playthrough.py --res 128 --steps 4000 --out runs/diamond_play
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from train_diamond_santa import Batch, make_denoiser, load_img, ACTIONS

PLAY = Path(__file__).resolve().parents[1] / "runs" / "diamond_dev" / "playthrough_demo"


def parse_playthrough():
    """-> list of transitions (before_png, action, before_cell, after_cell, after_png)."""
    files = sorted(p for p in PLAY.glob("step*.png") if "groundtruth" not in p.name)
    seq = []
    for p in files:
        m = re.match(r"step(\d+)_(\w+?)(?:_(\d+)-(\d+))?\.png", p.name)
        cell = (int(m.group(3)), int(m.group(4))) if m.group(3) else (1, 1)
        seq.append((p, m.group(2), cell))
    trans = []
    for i in range(1, len(seq)):
        pb, _, bc = seq[i - 1]
        pa, act, ac = seq[i]
        trans.append((pb, act, bc, ac, pa))
    return trans


def build_episodes(res, device):
    trans = parse_playthrough()
    eps = []
    for before_png, act, bc, ac, after_png in trans:
        before = load_img(before_png, res, device)
        after = load_img(after_png, res, device)
        obs = torch.stack([before, after], 0)
        a = torch.tensor([ACTIONS[act], 0], device=device)
        eps.append((obs, a, act, bc, ac))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=9)
    ap.add_argument("--out", default="runs/diamond_play")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    eps = build_episodes(args.res, device)
    print(f"playthrough: {len(eps)} transitions @ {args.res}px")
    for _, _, mv, b, t in eps:
        print(f"  {b} --{mv:5}--> {t}")

    obs = torch.stack([e[0] for e in eps], 0)
    act = torch.stack([e[1] for e in eps], 0)
    mask = torch.ones(len(eps), 2, dtype=torch.bool, device=device)

    net = make_denoiser(args.res, device)
    print(f"DIAMOND denoiser: {sum(p.numel() for p in net.parameters())/1e6:.2f}M params | {device}")
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
