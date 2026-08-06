"""Train the real DIAMOND denoiser on the 10-step Santa trajectory (smoke_data4).

Unlike train_diamond_santa.py (6 isolated relocations), this uses the full
ordered 10-step playthrough — INCLUDING blocked wall-bumps (left/up that keep
Santa put), so the model also learns "wall => no move". Trained at higher native
resolution (default 128px) for crisper frames in the demo video.

Each step is a 2-frame episode [before, after] with num_steps_conditioning=1.

    python scripts/train_diamond_traj.py --res 128 --steps 4000 --out runs/diamond_traj
"""
import argparse
import json
import sys
import time
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from train_diamond_santa import Batch, make_denoiser, load_img, ACTIONS

TRAJ = Path(__file__).resolve().parents[1] / "runs" / "diamond_dev" / "smoke_data4"


def build_traj_episodes(res, device):
    meta = json.loads((TRAJ / "manifest.json").read_text())
    eps = []
    for x in meta:
        i = x["idx"]
        before = load_img(TRAJ / "frames" / f"{i:05d}_before.png", res, device)
        after = load_img(TRAJ / "frames" / f"{i:05d}_after.png", res, device)
        obs = torch.stack([before, after], 0)
        act = torch.tensor([ACTIONS[x["move"]], 0], device=device)
        eps.append((obs, act, x["move"], tuple(x["santa_cell_before"]),
                    tuple(x["santa_cell_after"]), x.get("blocked", False)))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--out", default="runs/diamond_traj")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    eps = build_traj_episodes(args.res, device)
    print(f"trajectory: {len(eps)} steps @ {args.res}px")
    for _, _, mv, b, t, blk in eps:
        print(f"  {mv:5} {b}->{t}{'  (blocked/wall)' if blk else ''}")

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
