"""Train the CDiT + Self-Flow world model on one game (flow-matching objective).

Flow matching: x_t = (1-t)*x0 + t*noise; predict velocity (noise - x0). Self-Flow's
PER-TOKEN timesteps: each token's t drawn independently (dual-timestep asymmetry).
Change-weighted (like our DIAMOND runs) + self-distillation aux loss.

    python scripts/train_cdit.py --game hospital --steps 500 --out runs/cdit_hospital
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

from latent_wm import load_vae, encode
from cdit_selfflow import CDiTSelfFlow, LAT_CH, LAT

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--change-weight", type=float, default=20.0)
    ap.add_argument("--distill-weight", type=float, default=0.5)
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--per-token", action="store_true", default=True,
                    help="Self-Flow per-token timesteps (dual-timestep asymmetry)")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--out", default="runs/cdit_hospital")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    train = [r for r in idx["train"] if r["game"] == args.game]
    val = [r for r in idx["val"] if r["game"] == args.game]
    print(f"{args.game}: train {len(train)} val {len(val)}")

    print("loading Qwen VAE (frozen)...")
    vae = load_vae(device)
    net = CDiTSelfFlow(dim=args.dim, depth=args.depth, mask_ratio=args.mask_ratio).to(device)
    print(f"CDiT+Self-Flow: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params | "
          f"dim={args.dim} depth={args.depth} mask={args.mask_ratio} per-token=on")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    n_tok = net.n_tok

    def encode_pair(rows):
        # wm_index.json paths are relative to BUNDLE and already include the game dir
        cur = torch.stack([encode(vae, Image.open(BUNDLE / r["obs_t"]), args.res, device)[0]
                           for r in rows])
        nxt = torch.stack([encode(vae, Image.open(BUNDLE / r["obs_next"]), args.res, device)[0]
                           for r in rows])
        act = torch.tensor([r["action"] for r in rows], device=device)
        return cur.float(), nxt.float(), act

    def step_loss(cur, nxt, act, train_mode=True):
        b = cur.size(0)
        # per-token timesteps t in (0,1): (B, n_tok) -> broadcast to (B,1,H,W) per token
        if args.per_token:
            t_tok = torch.rand(b, n_tok, device=device)
        else:
            t_tok = torch.rand(b, 1, device=device).expand(b, n_tok)
        # expand per-token t to a spatial map matching the latent (each token = patch x patch)
        gh = net.gh
        t_map = t_tok.view(b, 1, gh, gh)
        t_map = torch.nn.functional.interpolate(t_map, size=(LAT, LAT), mode="nearest")
        noise = torch.randn_like(nxt)
        x_t = (1 - t_map) * nxt + t_map * noise
        v_target = noise - nxt                                    # flow-matching velocity
        # sigma proxy for embedding = t (monotonic); per-token
        pred, distill = net(x_t, cur, act, t_tok, return_distill=train_mode)
        w = 1.0 + args.change_weight * (nxt - cur).abs()
        fm = ((pred - v_target) ** 2 * w).mean()
        return fm, distill

    rng = np.random.default_rng(0)

    @torch.no_grad()
    def validate():
        net.eval(); ls = []
        for _ in range(8):
            rows = [val[i] for i in rng.integers(len(val), size=args.batch)]
            cur, nxt, act = encode_pair(rows)
            fm, _ = step_loss(cur, nxt, act, train_mode=False)
            ls.append(float(fm))
        net.train(); return float(np.mean(ls))

    log, t0 = [], time.time()
    for step in range(args.steps):
        rows = [train[i] for i in rng.integers(len(train), size=args.batch)]
        cur, nxt, act = encode_pair(rows)
        fm, distill = step_loss(cur, nxt, act, train_mode=True)
        loss = fm + args.distill_weight * distill
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        rec = dict(step=step, loss=float(loss.detach()), fm=float(fm.detach()),
                   distill=float(distill.detach()), t=round(time.time() - t0, 1))
        if step % args.val_every == 0 or step == args.steps - 1:
            rec["val"] = validate()
            print(f"[{step:4d}/{args.steps}] loss={rec['loss']:.4f} fm={rec['fm']:.4f} "
                  f"distill={rec['distill']:.4f} val={rec['val']:.4f} {time.time()-t0:.0f}s", flush=True)
        elif step % 50 == 0:
            print(f"[{step:4d}/{args.steps}] loss={rec['loss']:.4f} fm={rec['fm']:.4f} "
                  f"distill={rec['distill']:.4f} {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log))

    torch.save(net.state_dict(), out / "cdit_final.pt")
    print(f"done -> {out}/cdit_final.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
