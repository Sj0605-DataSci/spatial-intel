"""Latency benchmark for the latent world model, played as a game.

A "game step" = the model advancing one frame given an action. We benchmark the
closed-loop path a live engine actually runs:

    encode(current frame) -> sample next latent (N denoise steps) -> decode -> frame

Warm-up (one-time: model load, CUDA kernel autotune, cuDNN benchmark) is measured
and reported SEPARATELY from steady-state per-step latency, with a stage breakdown
(encode / denoise / decode) and a denoise-steps sweep.

    ( ulimit -v 104857600
      ../.venv/bin/python scripts/bench_latent_latency.py \
          --ckpt runs/latent_qwen/denoiser_final.pt )
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser, ACTIONS, VAE_BACKEND
from train_diamond_playthrough import parse_playthrough


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/latent_qwen/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=30, help="timed steady-state steps")
    ap.add_argument("--warmup", type=int, default=5, help="warm-up steps (excluded)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== latent WM latency benchmark ({VAE_BACKEND} VAE, {dev}) ===\n")

    # ---------- WARM-UP PHASE (one-time costs) ----------
    t_load0 = time.time()
    vae = load_vae(dev)
    net = make_latent_denoiser(dev)
    net.load_state_dict(torch.load(args.ckpt, map_location=dev))
    net.eval()
    sync(dev)
    load_s = time.time() - t_load0
    print(f"[warm-up] model load (VAE + denoiser + ckpt): {load_s:.2f} s")

    # a real before-frame to drive the loop
    before_png = parse_playthrough()[0][0]
    frame = Image.open(before_png)
    act = torch.tensor([[ACTIONS["right"]]], device=dev)

    def make_sampler(steps):
        return DiffusionSampler(net, DiffusionSamplerConfig(
            num_steps_denoising=steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    # first-call warm-up: triggers CUDA kernel compile / cuDNN autotune / VAE graph
    sampler = make_sampler(30)
    t_first0 = time.time()
    with torch.no_grad():
        lat = encode(vae, frame, args.res, dev)
        pred, _ = sampler.sample(lat.unsqueeze(1), act)
        _ = decode(vae, pred)
    sync(dev)
    first_s = time.time() - t_first0
    print(f"[warm-up] first full step (incl. kernel autotune): {first_s*1000:.0f} ms")
    # extra warm-up iterations (discarded)
    with torch.no_grad():
        for _ in range(args.warmup):
            lat = encode(vae, frame, args.res, dev)
            pred, _ = sampler.sample(lat.unsqueeze(1), act)
            _ = decode(vae, pred)
    sync(dev)
    print(f"[warm-up] {args.warmup} extra warm iterations done\n")

    # ---------- STEADY-STATE: stage breakdown @ 30 steps ----------
    print(f"=== steady-state per-step latency ({args.iters} iters) ===")
    enc_t, den_t, dec_t = [], [], []
    with torch.no_grad():
        for _ in range(args.iters):
            sync(dev); t0 = time.time()
            lat = encode(vae, frame, args.res, dev); sync(dev); t1 = time.time()
            pred, _ = sampler.sample(lat.unsqueeze(1), act); sync(dev); t2 = time.time()
            _ = decode(vae, pred); sync(dev); t3 = time.time()
            enc_t.append((t1 - t0) * 1000); den_t.append((t2 - t1) * 1000); dec_t.append((t3 - t2) * 1000)

    def stat(a):
        return f"{np.mean(a):6.1f} ± {np.std(a):4.1f} ms  (med {np.median(a):6.1f})"
    tot = np.array(enc_t) + np.array(den_t) + np.array(dec_t)
    print(f"  encode (VAE) : {stat(enc_t)}")
    print(f"  denoise (30) : {stat(den_t)}")
    print(f"  decode (VAE) : {stat(dec_t)}")
    print(f"  FULL STEP    : {stat(tot)}   -> {1000/np.median(tot):.1f} fps\n")

    # ---------- DENOISE-STEPS SWEEP (fps knob) ----------
    print("=== denoise-steps sweep (full step: encode+denoise+decode) ===")
    print(f"  {'steps':>5} | {'full ms':>8} | {'fps':>6} | {'denoise ms':>10}")
    for steps in [30, 20, 10, 5, 3, 1]:
        smp = make_sampler(steps)
        with torch.no_grad():  # warm this sampler
            lat = encode(vae, frame, args.res, dev); smp.sample(lat.unsqueeze(1), act)
        sync(dev)
        fulls, dens = [], []
        with torch.no_grad():
            for _ in range(15):
                sync(dev); t0 = time.time()
                lat = encode(vae, frame, args.res, dev); sync(dev); t1 = time.time()
                pred, _ = smp.sample(lat.unsqueeze(1), act); sync(dev); t2 = time.time()
                decode(vae, pred); sync(dev); t3 = time.time()
                fulls.append((t3 - t0) * 1000); dens.append((t2 - t1) * 1000)
        fm = np.median(fulls)
        print(f"  {steps:>5} | {fm:>8.1f} | {1000/fm:>6.1f} | {np.median(dens):>10.1f}")

    print(f"\nGPU: {torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'}")


if __name__ == "__main__":
    main()
