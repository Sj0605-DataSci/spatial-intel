"""Infer a single RICH (rich+LPIPS conv-UNet) model on one game: decode N val
predictions, save input|predicted|groundtruth + SSIM/LPIPS + latency.

    python scripts/infer_rich.py --game hospital --sigma-data 0.612 \
        --ckpt runs/final/hospital/denoiser_final.pt --n 8 --out runs/infer_hospital
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
from pytorch_msssim import ssim as ssim_fn
import lpips as lpips_lib

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode
from rich_wm import make_rich_denoiser, RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
ID_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}


def to_t(pil, size, device):
    a = np.asarray(pil.convert("RGB").resize(size)).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/infer_rich")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    (out / "samples").mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    rf = RichFeatures(args.game); geom = load_geom(args.game)
    net = make_rich_denoiser(device, N_ACTIONS, len(rf.targets), len(rf.hud_keys),
                             cond_channels=args.cond_channels, sigma_data=args.sigma_data)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)); net.eval()

    trans = [json.loads(l) for l in open(BUNDLE / args.game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL and json.loads(l)["effects"].get("moved_player")]
    import random
    random.Random(1).shuffle(trans); trans = trans[:args.n]

    ssims, lps, times = [], [], []
    for i, t in enumerate(trans):
        obs_t = Image.open(BUNDLE / args.game / t["obs_t"]).convert("RGB")
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")
        lat = encode(vae, obs_t, args.res, device).unsqueeze(1)
        act = torch.tensor([[CTRL[t["control"]]]], device=device)
        net.heatmap = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        net.inner_model.rich = dict(
            target=torch.tensor([f["target"]], device=device),
            hud=torch.tensor([f["hud"]], device=device),
            hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))
        sm = DiffusionSampler(net, DiffusionSamplerConfig(
            num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))
        torch.manual_seed(0)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            pred = decode(vae, sm.sample(lat, act)[0])
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        sz = gt.size
        tp, tg = to_t(pred, sz, device), to_t(gt, sz, device)
        ssims.append(ssim_fn(tp, tg, data_range=1.0).item())
        with torch.no_grad():
            lps.append(lpips_fn(tp*2-1, tg*2-1).item())
        sd = out / "samples" / f"{i:02d}_{ID_KEY[CTRL[t['control']]]}"
        sd.mkdir(exist_ok=True)
        obs_t.save(sd / "1_input.png"); pred.convert("RGB").save(sd / "2_predicted.png")
        gt.save(sd / "3_groundtruth.png")
        mp = t["effects"]["moved_player"]
        (sd / "info.txt").write_text(
            f"game {args.game}  action {ID_KEY[CTRL[t['control']]]}  "
            f"move {tuple(mp['from'])}->{tuple(mp['to'])}\n"
            f"SSIM {ssims[-1]:.4f}  LPIPS {lps[-1]:.4f}\n")

    print(f"\n=== INFER {args.game} ({Path(args.ckpt).name}) ===")
    print(f"  SSIM {np.mean(ssims):.4f}  LPIPS {np.mean(lps):.4f}  (n={len(trans)})")
    print(f"  latency (excl. warmup): {np.median(times[1:]):.0f} ms/frame  ({1000/np.median(times[1:]):.1f} fps)")
    print(f"-> {out}/samples/  (each: 1_input 2_predicted 3_groundtruth + info.txt)")


if __name__ == "__main__":
    main()
