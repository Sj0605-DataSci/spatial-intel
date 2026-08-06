"""The 'everything' model: RichCDiT (CDiT + Self-Flow + heatmap/vector conditioning)
+ LPIPS perceptual loss. All wins combined, on one game.

    python scripts/train_everything.py --game hospital --steps 800 --out runs/everything_hospital
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
import lpips as lpips_lib

from latent_wm import load_vae, encode, decode_tensor
from cdit_selfflow import RichCDiT, LAT_CH, LAT, HEATMAP_CH
from rich_wm import RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
LPIPS_SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--change-weight", type=float, default=20.0)
    ap.add_argument("--distill-weight", type=float, default=0.5)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--out", default="runs/everything_hospital")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    # full transitions (need effects for heatmap/rich)
    trans = [json.loads(l) for l in open(BUNDLE / args.game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL]
    import random
    random.Random(0).shuffle(trans)
    k = int(len(trans) * 0.2)
    val, train = trans[:k], trans[k:]
    print(f"{args.game}: train {len(train)} val {len(val)}")

    geom = load_geom(args.game)
    rf = RichFeatures(args.game)

    print("loading Qwen VAE + LPIPS...")
    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    net = RichCDiT(dim=args.dim, depth=args.depth, mask_ratio=args.mask_ratio,
                   n_targets=len(rf.targets), n_hud=len(rf.hud_keys)).to(device)
    print(f"EVERYTHING (RichCDiT+SelfFlow+LPIPS): {sum(p.numel() for p in net.parameters())/1e6:.1f}M | "
          f"dim={args.dim} depth={args.depth} mask={args.mask_ratio}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    n_tok = net.n_tok

    def prep(rows):
        cur = torch.stack([encode(vae, Image.open(BUNDLE / args.game / t["obs_t"]), args.res, device)[0]
                           for t in rows]).float()
        nxt = torch.stack([encode(vae, Image.open(BUNDLE / args.game / t["obs_next"]), args.res, device)[0]
                           for t in rows]).float()
        act = torch.tensor([CTRL[t["control"]] for t in rows], device=device)
        heat = torch.stack([build_heatmap(geom, t["effects"], device) for t in rows])
        fs = [rf.encode(t) for t in rows]
        net.rich = dict(
            heatmap=heat,
            target=torch.tensor([f["target"] for f in fs], device=device),
            hud=torch.tensor([f["hud"] for f in fs], device=device),
            hud_delta=torch.tensor([f["hud_delta"] for f in fs], device=device, dtype=torch.float32))
        gtr = None
        if args.lpips_weight > 0:
            g = []
            for t in rows:
                im = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB").resize((LPIPS_SIZE, LPIPS_SIZE))
                g.append(torch.from_numpy(np.asarray(im).copy()).float().div(127.5).sub(1).permute(2, 0, 1))
            gtr = torch.stack(g).to(device)
        return cur, nxt, act, gtr

    def losses(cur, nxt, act, gtr, train_mode=True):
        b = cur.size(0)
        t_tok = torch.rand(b, n_tok, device=device)
        t_map = torch.nn.functional.interpolate(
            t_tok.view(b, 1, net.gh, net.gh), size=(LAT, LAT), mode="nearest")
        noise = torch.randn_like(nxt)
        x_t = (1 - t_map) * nxt + t_map * noise
        v_target = noise - nxt
        pred, distill = net(x_t, cur, act, t_tok, return_distill=train_mode)
        w = 1.0 + args.change_weight * (nxt - cur).abs()
        fm = ((pred - v_target) ** 2 * w).mean()
        # LPIPS on the implied clean x0 = x_t - t*v_pred  (only for low-t / cleaner samples)
        perc = torch.zeros((), device=device)
        if train_mode and gtr is not None and args.lpips_weight > 0:
            x0_pred = x_t - t_map * pred                    # predicted clean latent
            j = int(t_tok.mean(1).argmin())                # the cleanest sample in batch
            pr = decode_tensor(vae, x0_pred[j:j+1], out_size=LPIPS_SIZE)
            perc = lpips_fn(pr, gtr[j:j+1]).mean()
        return fm, distill, perc

    rng = np.random.default_rng(0)

    @torch.no_grad()
    def validate():
        net.eval(); ls = []
        for _ in range(8):
            rows = [val[i] for i in rng.integers(len(val), size=args.batch)]
            cur, nxt, act, _ = prep(rows)
            fm, _, _ = losses(cur, nxt, act, None, train_mode=False)
            ls.append(float(fm))
        net.train(); return float(np.mean(ls))

    log, t0 = [], time.time()
    for step in range(args.steps):
        rows = [train[i] for i in rng.integers(len(train), size=args.batch)]
        cur, nxt, act, gtr = prep(rows)
        fm, distill, perc = losses(cur, nxt, act, gtr, train_mode=True)
        loss = fm + args.distill_weight * distill + args.lpips_weight * perc
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        rec = dict(step=step, loss=float(loss.detach()), fm=float(fm.detach()),
                   distill=float(distill.detach()), lpips=float(perc.detach()),
                   t=round(time.time() - t0, 1))
        if step % args.val_every == 0 or step == args.steps - 1:
            rec["val"] = validate()
            print(f"[{step:4d}/{args.steps}] loss={rec['loss']:.4f} fm={rec['fm']:.4f} "
                  f"distill={rec['distill']:.4f} lpips={rec['lpips']:.4f} val={rec['val']:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        elif step % 50 == 0:
            print(f"[{step:4d}/{args.steps}] loss={rec['loss']:.4f} fm={rec['fm']:.4f} "
                  f"lpips={rec['lpips']:.4f} {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log))
        if args.save_every and step and step % args.save_every == 0:
            torch.save(net.state_dict(), out / f"cdit_{step}.pt")

    torch.save(net.state_dict(), out / "cdit_final.pt")
    print(f"done -> {out}/cdit_final.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
