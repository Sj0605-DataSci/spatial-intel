"""Eval the EVERYTHING model (RichCDiT + SelfFlow + LPIPS): flow-matching sample
with rich conditioning -> decode -> images + SSIM/LPIPS.
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
from cdit_selfflow import RichCDiT, LAT_CH, LAT
from rich_wm import RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}


def to_t(pil, size, device):
    a = np.asarray(pil.convert("RGB").resize(size)).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def sample(net, cur, act, steps=30):
    b = cur.size(0)
    x = torch.randn(b, LAT_CH, LAT, LAT, device=cur.device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=cur.device)
    for i in range(steps):
        t_tok = torch.full((b, net.n_tok), float(ts[i]), device=cur.device)
        v, _ = net(x, cur, act, t_tok, return_distill=False)
        x = x - v * (ts[i] - ts[i + 1])
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/everything_hospital/cdit_final.pt")
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/eval_everything")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    (out / "samples").mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    rf = RichFeatures(args.game); geom = load_geom(args.game)
    net = RichCDiT(dim=args.dim, depth=args.depth,
                   n_targets=len(rf.targets), n_hud=len(rf.hud_keys)).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)); net.eval()

    trans = [json.loads(l) for l in open(BUNDLE / args.game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL and json.loads(l)["effects"].get("moved_player")]
    import random
    random.Random(1).shuffle(trans); trans = trans[:args.n]

    ssims, lps = [], []
    for i, t in enumerate(trans):
        obs_t = Image.open(BUNDLE / args.game / t["obs_t"]).convert("RGB")
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")
        cur = encode(vae, obs_t, args.res, device).float()
        act = torch.tensor([CTRL[t["control"]]], device=device)
        f = rf.encode(t)
        net.rich = dict(
            heatmap=build_heatmap(geom, t["effects"], device).unsqueeze(0),
            target=torch.tensor([f["target"]], device=device),
            hud=torch.tensor([f["hud"]], device=device),
            hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))
        pred = decode(vae, sample(net, cur, act, steps=args.steps))
        sz = gt.size
        tp, tg = to_t(pred, sz, device), to_t(gt, sz, device)
        ssims.append(ssim_fn(tp, tg, data_range=1.0).item())
        with torch.no_grad():
            lps.append(lpips_fn(tp*2-1, tg*2-1).item())
        sd = out / "samples" / f"{i:02d}"; sd.mkdir(exist_ok=True)
        obs_t.save(sd / "1_input.png"); pred.convert("RGB").save(sd / "2_predicted.png")
        gt.save(sd / "3_groundtruth.png")

    print(f"\n=== EVERYTHING model eval — {args.game} ===")
    print(f"  SSIM {np.mean(ssims):.4f}  LPIPS {np.mean(lps):.4f}  (n={len(trans)})")
    print(f"-> {out}/samples/")


if __name__ == "__main__":
    main()
