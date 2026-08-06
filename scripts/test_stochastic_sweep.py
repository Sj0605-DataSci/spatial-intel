"""Push stochastic sampling harder + measure with MS-SSIM.

One transition, many samplers (deterministic baseline + a stochastic churn sweep),
each scored by MS-SSIM vs ground-truth (higher = more structurally similar) and
pixel-L1. Also averages over a few transitions per config for a stable number.

Note (honest): MS-SSIM is a DISTORTION metric, so it may still favor the blurry
mean. We report it, but the visual grid is the real judge of sharpness.

    python scripts/test_stochastic_sweep.py --ckpt runs/bundle_sharp/denoiser_final.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ms_ssim
import lpips as lpips_lib

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5


def to_t(pil, size):
    a = np.asarray(pil.convert("RGB").resize(size)).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/bundle_sharp/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--game", default="7_worlds_of_wonder")
    ap.add_argument("--n-eval", type=int, default=6, help="transitions to average metrics over")
    ap.add_argument("--out", default="runs/stoch_sweep")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    rows = [r for r in idx["val"] if r["game"] == args.game and r["moved"]]

    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()   # perceptual metric
    net = make_latent_denoiser(device, num_actions=N_ACTIONS, cond_channels=args.cond_channels)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    configs = [
        ("det 30",       dict(num_steps_denoising=30,  s_churn=0)),
        ("stoch c20",    dict(num_steps_denoising=30,  s_churn=20,  s_tmin=0.05, s_tmax=5,  s_noise=1.0)),
        ("stoch c40",    dict(num_steps_denoising=50,  s_churn=40,  s_tmin=0.05, s_tmax=5,  s_noise=1.0)),
        ("stoch c80",    dict(num_steps_denoising=80,  s_churn=80,  s_tmin=0.02, s_tmax=8,  s_noise=1.0)),
        ("stoch c80 n1.1", dict(num_steps_denoising=80, s_churn=80, s_tmin=0.02, s_tmax=8, s_noise=1.1)),
        ("stoch c150",   dict(num_steps_denoising=120, s_churn=150, s_tmin=0.02, s_tmax=10, s_noise=1.0)),
    ]

    # --- metrics averaged over n_eval transitions ---
    print(f"game={args.game}  averaging over {args.n_eval} val transitions\n")
    print(f"{'sampler':16} {'MS-SSIM^':>9} {'LPIPS_v':>9} {'pixel-L1':>9}")
    print(f"{'':16} {'(distort)':>9} {'(percept)':>9}")
    results = {}
    for name, cfg in configs:
        sampler = DiffusionSampler(net, DiffusionSamplerConfig(
            sigma_min=2e-3, sigma_max=20, rho=7, order=1, **cfg))
        mss, lps, l1s = [], [], []
        for r in rows[:args.n_eval]:
            obs_t = Image.open(BUNDLE / r["obs_t"]); gt = Image.open(BUNDLE / r["obs_next"])
            lat = encode(vae, obs_t, args.res, device).unsqueeze(1)
            act = torch.tensor([[r["action"]]], device=device)
            torch.manual_seed(0)
            with torch.no_grad():
                pl, _ = sampler.sample(lat, act)
            pred = decode(vae, pl)
            sz = gt.size
            tp, tg = to_t(pred, sz).to(device), to_t(gt, sz).to(device)
            m = ms_ssim(tp, tg, data_range=1.0).item()
            with torch.no_grad():
                lp = lpips_fn(tp * 2 - 1, tg * 2 - 1).item()   # LPIPS wants [-1,1]
            l1 = float(np.abs(np.asarray(pred.resize(sz)).astype(float) -
                              np.asarray(gt.convert("RGB")).astype(float)).mean())
            mss.append(m); lps.append(lp); l1s.append(l1)
        results[name] = (np.mean(mss), np.mean(lps), np.mean(l1s))
        print(f"{name:16} {np.mean(mss):>9.4f} {np.mean(lps):>9.4f} {np.mean(l1s):>9.2f}", flush=True)
    print("  ^MS-SSIM: higher=better (distortion, favors blur)  "
          "LPIPS: LOWER=better (perceptual, favors sharp)")

    # --- visual grid on ONE transition ---
    r = rows[0]
    obs_t = Image.open(BUNDLE / r["obs_t"]).convert("RGB")
    gt = Image.open(BUNDLE / r["obs_next"]).convert("RGB")
    P = 256
    def fnt(s, b=False):
        try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", s)
        except Exception: return ImageFont.load_default()
    tiles = [("input", obs_t), ("ground truth", gt)]
    for name, cfg in configs:
        sampler = DiffusionSampler(net, DiffusionSamplerConfig(
            sigma_min=2e-3, sigma_max=20, rho=7, order=1, **cfg))
        torch.manual_seed(0)
        with torch.no_grad():
            pl, _ = sampler.sample(encode(vae, obs_t, args.res, device).unsqueeze(1),
                                   torch.tensor([[r["action"]]], device=device))
        tiles.append((f"{name} (mss {results[name][0]:.3f})", decode(vae, pl).convert("RGB")))
    cols = 4; rows_n = (len(tiles) + cols - 1) // cols
    cv = Image.new("RGB", (cols * (P + 6) + 6, rows_n * (P + 26) + 6), (17, 24, 33))
    d = ImageDraw.Draw(cv)
    for i, (name, im) in enumerate(tiles):
        rr, cc = divmod(i, cols); x, y = 6 + cc * (P + 6), 6 + rr * (P + 26)
        cv.paste(im.resize((P, P)), (x, y))
        col = (46, 125, 91) if name in ("input", "ground truth") else (198, 40, 40)
        d.text((x, y + P + 2), name, col, fnt(12, True))
    cv.save(out / "stoch_sweep.png")
    print(f"\n-> {out}/stoch_sweep.png")


if __name__ == "__main__":
    main()
