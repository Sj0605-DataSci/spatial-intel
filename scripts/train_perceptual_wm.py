"""LPIPS perceptual-loss trainer (noise-gated) on ONE game. Change-weighted MSE +
LPIPS on decoded low-noise predictions. Verifies loss decreases & sharpens.

    python scripts/train_perceptual_wm.py --game hospital --steps 500 --sigma-data 0.612 \
        --lpips-weight 0.5 --out runs/perc_hospital
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
import lpips as lpips_lib

from latent_wm import (load_vae, encode, decode, decode_tensor,
                       make_perceptual_denoiser, LATENT_CH)
from train_diamond_santa import Batch

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
LPIPS_SIZE = 256   # decode/compare at 256px to bound memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--sigma-gate", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--out", default="runs/perc_hospital")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    train = [r for r in idx["train"] if r["game"] == args.game]
    val = [r for r in idx["val"] if r["game"] == args.game]
    print(f"{args.game}: train {len(train)} val {len(val)}")

    print("loading Qwen VAE (frozen) + LPIPS...")
    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    net = make_perceptual_denoiser(device, N_ACTIONS, change_weight=20.0,
                                   cond_channels=args.cond_channels, sigma_data=args.sigma_data,
                                   lpips_weight=args.lpips_weight, sigma_gate=args.sigma_gate)
    net.set_perceptual(vae, lpips_fn, lambda lat: decode_tensor(vae, lat, out_size=LPIPS_SIZE))
    print(f"perceptual denoiser: {sum(p.numel() for p in net.parameters())/1e6:.1f}M | "
          f"lpips_w={args.lpips_weight} gate={args.sigma_gate}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def make_batch(rows):
        lt, ln, gt = [], [], []
        for r in rows:
            lt.append(encode(vae, Image.open(BUNDLE / r["obs_t"]), args.res, device)[0])
            ln.append(encode(vae, Image.open(BUNDLE / r["obs_next"]), args.res, device)[0])
            # real next frame in RGB [-1,1] at LPIPS_SIZE (target for perceptual loss)
            im = Image.open(BUNDLE / r["obs_next"]).convert("RGB").resize((LPIPS_SIZE, LPIPS_SIZE))
            gt.append(torch.from_numpy(np.asarray(im)).float().div(127.5).sub(1).permute(2, 0, 1))
        obs = torch.stack([torch.stack(lt), torch.stack(ln)], 1)
        net.gt_next_rgb = torch.stack(gt).to(device)
        act = torch.tensor([[r["action"], 0] for r in rows], device=device)
        m = torch.ones(len(rows), 2, dtype=torch.bool, device=device)
        return Batch(obs=obs, act=act, mask_padding=m)

    rng = np.random.default_rng(0)

    @torch.no_grad()
    def validate():
        net.eval(); ls = []
        for _ in range(8):
            rows = [val[i] for i in rng.integers(len(val), size=args.batch)]
            ls.append(float(net(make_batch(rows))[0]))
        net.train(); return float(np.mean(ls))

    log, t0 = [], time.time()
    for step in range(args.steps):
        rows = [train[i] for i in rng.integers(len(train), size=args.batch)]
        loss, logs = net(make_batch(rows))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        rec = dict(step=step, loss=float(loss.detach()),
                   mse=float(logs["loss_denoising"]),
                   lpips=float(logs.get("loss_lpips", 0)), t=round(time.time() - t0, 1))
        if step % args.val_every == 0 or step == args.steps - 1:
            rec["val"] = validate()
            print(f"[{step:4d}/{args.steps}] loss={rec['loss']:.4f} "
                  f"mse={rec['mse']:.4f} lpips={rec['lpips']:.4f} val={rec['val']:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        elif step % 25 == 0:
            print(f"[{step:4d}/{args.steps}] loss={rec['loss']:.4f} "
                  f"mse={rec['mse']:.4f} lpips={rec['lpips']:.4f} {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log))

    torch.save(net.state_dict(), out / "denoiser_final.pt")
    print(f"done -> {out}/denoiser_final.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
