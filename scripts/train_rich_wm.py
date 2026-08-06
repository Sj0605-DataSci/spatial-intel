"""Axis 4 trainer: rich-conditioned controllable renderer on ONE game (hospital).

Builds, per transition: the heatmap (from geometry + moved_player cells) and the
structured features (action/target/HUD), and trains the RichDenoiser. 80/20 split
on the game's own transitions.

    python scripts/train_rich_wm.py --game hospital --steps 800 --sigma-data 0.612 \
        --out runs/rich_hospital
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image

from latent_wm import load_vae, encode
from rich_wm import (make_rich_denoiser, RichFeatures, build_heatmap, load_geom,
                     HEATMAP_CH, LATENT_CH)
from train_diamond_santa import Batch

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5


def load_transitions(game, val_frac=0.2, seed=0):
    trans = [json.loads(l) for l in open(BUNDLE / game / "transitions.jsonl")
             if json.loads(l).get("control") in {"W", "S", "A", "D", "Space"}]
    import random
    random.Random(seed).shuffle(trans)
    k = int(len(trans) * val_frac)
    return trans[k:], trans[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--lpips-weight", type=float, default=0.0,
                    help=">0 adds noise-gated LPIPS perceptual loss (rich + LPIPS combo)")
    ap.add_argument("--x-prediction", action="store_true",
                    help="DreamerV4 x-prediction: supervise clean frame directly (anti-drift)")
    ap.add_argument("--context-noise", type=float, default=0.0,
                    help="GameNGen: noise added to context input in training (anti-drift)")
    ap.add_argument("--resume", default=None, help="resume from a checkpoint .pt")
    ap.add_argument("--out", default="runs/rich_hospital")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    train, val = load_transitions(args.game)
    print(f"{args.game}: train {len(train)} | val {len(val)}")
    geom = load_geom(args.game)
    rf = RichFeatures(args.game)
    ctrl2id = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
    print(f"vocab: {len(rf.actions)} act, {len(rf.targets)} targets, {len(rf.hud_keys)} hud")

    print("loading Qwen VAE (frozen)...")
    vae = load_vae(device)
    net = make_rich_denoiser(device, N_ACTIONS, len(rf.targets), len(rf.hud_keys),
                             cond_channels=args.cond_channels, sigma_data=args.sigma_data,
                             x_prediction=args.x_prediction, context_noise=args.context_noise)
    if args.x_prediction or args.context_noise > 0:
        print(f"ANTI-DRIFT: x_prediction={args.x_prediction} context_noise={args.context_noise}")
    if args.lpips_weight > 0:
        import lpips as lpips_lib
        from latent_wm import decode_tensor
        lp = lpips_lib.LPIPS(net="alex").to(device).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
        net.set_perceptual(vae, lp, lambda l: decode_tensor(vae, l, 256),
                           weight=args.lpips_weight, gate=1.0, perc_max=1)
        print(f"RICH + LPIPS (weight={args.lpips_weight})")
    print(f"rich denoiser: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    start_step = 0
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=device))
        # infer resume step from filename like denoiser_1000.pt
        import re as _re
        mm = _re.search(r"denoiser_(\d+)\.pt", str(args.resume))
        start_step = int(mm.group(1)) if mm else 0
        print(f"RESUMED from {args.resume} (start_step={start_step})")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    import numpy as _np
    def make_batch(rows):
        lt, ln, heat, tgt, hud, hdel, act, gtr = [], [], [], [], [], [], [], []
        for t in rows:
            lt.append(encode(vae, Image.open(BUNDLE / args.game / t["obs_t"]), args.res, device)[0])
            ln.append(encode(vae, Image.open(BUNDLE / args.game / t["obs_next"]), args.res, device)[0])
            heat.append(build_heatmap(geom, t["effects"], device))
            f = rf.encode(t)
            tgt.append(f["target"]); hud.append(f["hud"]); hdel.append(f["hud_delta"])
            act.append(ctrl2id[t["control"]])
            if args.lpips_weight > 0:
                im = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB").resize((256, 256))
                gtr.append(torch.from_numpy(_np.asarray(im).copy()).float().div(127.5).sub(1).permute(2, 0, 1))
        obs = torch.stack([torch.stack(lt), torch.stack(ln)], 1)     # (B,2,C,H,W)
        net.heatmap = torch.stack(heat)                              # (B,2,H,W)
        if args.lpips_weight > 0:
            net.gt_next_rgb = torch.stack(gtr).to(device)
        net.inner_model.rich = dict(
            target=torch.tensor(tgt, device=device),
            hud=torch.tensor(hud, device=device),
            hud_delta=torch.tensor(hdel, device=device, dtype=torch.float32))
        a = torch.tensor([[x] for x in act], device=device)
        m = torch.ones(len(rows), 2, dtype=torch.bool, device=device)
        return Batch(obs=obs, act=a, mask_padding=m)

    rng = np.random.default_rng(0)

    @torch.no_grad()
    def validate():
        net.eval(); ls = []
        for _ in range(10):
            rows = [val[i] for i in rng.integers(len(val), size=args.batch)]
            ls.append(float(net(make_batch(rows))[0]))
        net.train(); return float(np.mean(ls))

    log, t0 = [], time.time()
    best_val = float("inf")
    prev_val = None
    for step in range(start_step, args.steps):
        rows = [train[i] for i in rng.integers(len(train), size=args.batch)]
        loss, logs = net(make_batch(rows))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        rec = dict(step=step, loss=float(loss.detach()),
                   lpips=float(logs.get("loss_lpips", 0)), t=round(time.time() - t0, 1))
        if step % args.val_every == 0 or step == args.steps - 1:
            rec["val"] = validate()
            # track direction: is val improving vs last check?
            arrow = ""
            if prev_val is not None:
                arrow = "down" if rec["val"] < prev_val else "up"
            prev_val = rec["val"]
            # SAVE BEST checkpoint (minimal val loss)
            if rec["val"] < best_val:
                best_val = rec["val"]
                torch.save(net.state_dict(), out / "denoiser_best.pt")
                (out / "best.json").write_text(json.dumps(
                    {"step": step, "val": best_val, "loss": rec["loss"],
                     "lpips": rec["lpips"]}, indent=2))
                rec["is_best"] = True
                arrow += " *BEST*"
            print(f"[{step:4d}/{args.steps}] train={rec['loss']:.4f} lpips={rec['lpips']:.4f} "
                  f"val={rec['val']:.4f} {arrow} {time.time()-t0:.0f}s", flush=True)
        elif step % 50 == 0:
            print(f"[{step:4d}/{args.steps}] train={rec['loss']:.4f} lpips={rec['lpips']:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log))
        if args.save_every and step and step % args.save_every == 0:
            torch.save(net.state_dict(), out / f"denoiser_{step}.pt")

    torch.save(net.state_dict(), out / "denoiser_final.pt")
    print(f"done -> {out}/denoiser_final.pt  (best val={best_val:.4f} -> denoiser_best.pt)"
          f" ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
