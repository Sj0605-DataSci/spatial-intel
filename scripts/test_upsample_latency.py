"""Upscale a real predicted frame + benchmark upsampling latency (GB10) and project A100.

Takes a predicted frame, sharpens/upscales it with Real-ESRGAN (via spandrel),
saves before/after, and times the SR pass with warm-up separated. Projects A100
latency from the GB10-vs-A100 memory-bandwidth ratio (SR is bandwidth-bound conv).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

# GB10 vs A100 memory bandwidth (both SR and VAE are bandwidth-bound conv nets)
GB10_BW = 273.0      # GB/s (LPDDR5X, measured)
A100_BW = 1935.0     # GB/s (A100 80GB SXM HBM2e)
BW_RATIO = A100_BW / GB10_BW   # ~7.1x


def load_sr(path, device):
    from spandrel import ModelLoader
    m = ModelLoader().load_from_file(path).to(device).eval()

    @torch.no_grad()
    def up(pil):
        a = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
        out = m(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((out * 255).round().astype(np.uint8))
    return up, m.scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="runs/cmp_hospital/plain_vs_rich.png",
                    help="a predicted frame to upscale")
    ap.add_argument("--sr", default="models/esrgan/RealESRGAN_x4plus.pth")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="runs/upsample_test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    # use a single RICH prediction crop if available, else the given image
    img = Image.open(args.img).convert("RGB")
    # if it's the comparison sheet, crop the RICH column of row 1 (approx)
    if "plain_vs_rich" in args.img:
        W = img.width // 4
        img = img.crop((2 * W + 8, 28, 3 * W, 28 + (W - 12)))   # RICH panel, row1
    img.save(out / "before.png")
    print(f"input frame: {img.size}")

    up, scale = load_sr(args.sr, device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # warm-up
    sync(); t = time.time()
    hi = up(img)
    sync(); warm = time.time() - t
    for _ in range(3):
        up(img)
    sync()

    # timed
    sync(); t0 = time.time()
    for _ in range(args.iters):
        hi = up(img)
    sync()
    ms = (time.time() - t0) / args.iters * 1000
    hi.save(out / "after_x{}.png".format(scale))

    a100_ms = ms / BW_RATIO
    print(f"\n=== Real-ESRGAN x{scale} upsample: {img.size} -> {hi.size} ===")
    print(f"  warm-up first call : {warm*1000:.0f} ms")
    print(f"  GB10 (measured)    : {ms:.1f} ms  ({1000/ms:.1f} fps)")
    print(f"  A100 (projected)   : {a100_ms:.1f} ms  ({1000/a100_ms:.1f} fps)  "
          f"[bandwidth {GB10_BW:.0f}->{A100_BW:.0f} GB/s, {BW_RATIO:.1f}x]")
    print(f"-> {out}/before.png , after_x{scale}.png")


if __name__ == "__main__":
    main()
