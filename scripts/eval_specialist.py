"""Eval a specialist (or generalist) checkpoint on one game:
  - LATENCY benchmark (warm-up separated; encode/denoise/decode breakdown)
  - per-sample folder: input.png, next_predicted.png, next_groundtruth.png + info.txt
  - SSIM + LPIPS (blurriness) + grid-coord accuracy summary

    python scripts/eval_specialist.py --ckpt runs/spec_hospital_800/denoiser_final.pt \
        --game hospital --sigma-data 0.612 --n 12 --out runs/eval_spec_hospital
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
from latent_wm import load_vae, encode, decode, make_latent_denoiser
from hybrid_geometry import pixel_to_cell

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
ID_KEY = {0: "W(up)", 1: "S(down)", 2: "A(left)", 3: "D(right)", 4: "Space(act)"}


def to_t(pil, size, device):
    a = np.asarray(pil.convert("RGB").resize(size)).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def state_index(game):
    return {json.loads(l)["id"]: json.loads(l) for l in open(BUNDLE / game / "states.jsonl")}


def obs_id(p):
    return int(Path(p).stem.split("_")[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--churn", type=float, default=40.0, help="stochastic churn (0=det)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="runs/eval_spec")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # ---- load ----
    t_load = time.time()
    vae = load_vae(device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    net = make_latent_denoiser(device, num_actions=N_ACTIONS, cond_channels=args.cond_channels,
                               sigma_data=args.sigma_data)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)); net.eval()
    load_s = time.time() - t_load
    cfg = dict(num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1)
    if args.churn > 0:
        cfg.update(s_churn=args.churn, s_tmin=0.05, s_tmax=5, s_noise=1.0)
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(**cfg))

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    rows = [r for r in idx["val"] if r["game"] == args.game and r["moved"]][:args.n]
    geom = json.loads((BUNDLE / args.game / "geometry.json").read_text())
    states = state_index(args.game)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # ---- warm-up (one-time) ----
    r0 = rows[0]
    lat0 = encode(vae, Image.open(BUNDLE / r0["obs_t"]), args.res, device).unsqueeze(1)
    a0 = torch.tensor([[r0["action"]]], device=device)
    sync(); t = time.time()
    with torch.no_grad():
        _ = decode(vae, sampler.sample(lat0, a0)[0])
    sync(); warm_first = time.time() - t
    for _ in range(2):
        with torch.no_grad():
            _ = decode(vae, sampler.sample(lat0, a0)[0])
    sync()

    # ---- per-sample eval + timing ----
    enc_t, den_t, dec_t = [], [], []
    ssims, lps, coord_hit = [], [], []
    print(f"eval {len(rows)} val samples of {args.game}  (churn={args.churn})")
    for i, r in enumerate(rows):
        obs_t = Image.open(BUNDLE / r["obs_t"]); gt = Image.open(BUNDLE / r["obs_next"])
        sync(); t0 = time.time()
        lat = encode(vae, obs_t, args.res, device).unsqueeze(1); sync(); t1 = time.time()
        act = torch.tensor([[r["action"]]], device=device)
        with torch.no_grad():
            pl, _ = sampler.sample(lat, act); sync(); t2 = time.time()
            pred = decode(vae, pl); sync(); t3 = time.time()
        enc_t.append((t1-t0)*1000); den_t.append((t2-t1)*1000); dec_t.append((t3-t2)*1000)

        sz = gt.size
        tp, tg = to_t(pred, sz, device), to_t(gt, sz, device)
        ssims.append(ssim_fn(tp, tg, data_range=1.0).item())
        with torch.no_grad():
            lps.append(lpips_fn(tp*2-1, tg*2-1).item())
        # grid-coord: detect player cell in prediction vs GT px,py
        a = np.asarray(obs_t.convert("RGB")).astype(int)
        b = np.asarray(pred.convert("RGB").resize(sz)).astype(int)
        dd = np.abs(a-b).sum(-1); ys, xs = np.where(dd > 30)
        det = pixel_to_cell(geom, xs.mean(), ys.mean()) if len(xs) > 15 else None
        true = states[obs_id(r["obs_next"])]; tc = (true["px"], true["py"])
        coord_hit.append(det == tc)

        # save viewable sample folder
        sd = out / "samples" / f"{i:02d}_{ID_KEY[r['action']].split('(')[0]}"
        sd.mkdir(exist_ok=True)
        obs_t.convert("RGB").save(sd / "1_input.png")
        pred.convert("RGB").save(sd / "2_predicted.png")
        gt.convert("RGB").save(sd / "3_groundtruth.png")
        (sd / "info.txt").write_text(
            f"game: {args.game}\naction: {ID_KEY[r['action']]}\n"
            f"true next cell (px,py): {tc}\ndetected cell in prediction: {det}\n"
            f"SSIM: {ssims[-1]:.4f}  LPIPS: {lps[-1]:.4f}\n")

    def st(a): return f"{np.mean(a):.1f}+-{np.std(a):.1f}ms"
    tot = np.array(enc_t)+np.array(den_t)+np.array(dec_t)
    summary = {
        "game": args.game, "ckpt": args.ckpt, "churn": args.churn,
        "load_s": round(load_s, 1), "warmup_first_step_ms": round(warm_first*1000),
        "latency_encode_ms": round(float(np.mean(enc_t)), 1),
        "latency_denoise_ms": round(float(np.mean(den_t)), 1),
        "latency_decode_ms": round(float(np.mean(dec_t)), 1),
        "latency_full_step_ms": round(float(np.median(tot)), 1),
        "fps": round(1000/float(np.median(tot)), 2),
        "SSIM": round(float(np.mean(ssims)), 4),
        "LPIPS": round(float(np.mean(lps)), 4),
        "grid_coord_acc": round(float(np.mean(coord_hit)), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(f"  load: {summary['load_s']}s | warmup first step: {summary['warmup_first_step_ms']}ms")
    print(f"  latency: encode {st(enc_t)}  denoise {st(den_t)}  decode {st(dec_t)}")
    print(f"  FULL STEP: {summary['latency_full_step_ms']}ms  ({summary['fps']} fps)")
    print(f"  SSIM {summary['SSIM']} | LPIPS {summary['LPIPS']} | grid-coord-acc {summary['grid_coord_acc']}")
    print(f"-> {out}/samples/  (each: 1_input 2_predicted 3_groundtruth + info.txt)")


if __name__ == "__main__":
    main()
