"""Record the LATENT world model playing the Santa playthrough — CRISP 1024px.

For each of the 9 moves: encode the before-frame -> latent, sample the next
latent with the trained denoiser (conditioned on the action), VAE-decode -> a
sharp 1024px predicted frame, show it beside the ground truth.

Run GUARDED so a stray huge alloc can't OOM the machine:
    ( ulimit -v 104857600
      ../.venv/bin/python scripts/record_latent_playthrough.py \
          --ckpt runs/latent_play/denoiser_final.pt )

Writes runs/latent_play_video/:  diamond_play.gif, contact_sheet.png, diamond_play.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# DIAMOND src must be on the path before importing `models.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser, ACTIONS, LATENT_CH
from train_diamond_playthrough import parse_playthrough

PANEL = 420
PAD = 24
LABEL_H = 88
BG = (14, 22, 32); INK = (231, 238, 245); MUTED = (147, 164, 182)
BADGE = (198, 40, 40); HIT = (46, 125, 91); MISS = (192, 135, 58); SLATE = (95, 114, 133)
ORIGIN = (227, 227); CELL = 87.5


def fnt(sz, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def santa_cell(arr1024):
    from scipy import ndimage
    R, G, B = arr1024[..., 0].astype(int), arr1024[..., 1].astype(int), arr1024[..., 2].astype(int)
    red = (R > 150) & (R - G > 60) & (R - B > 60)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    best, bs = None, -1.0
    for k in range(1, n + 1):
        ys, xs = np.where(lbl == k)
        if len(xs) < 300:
            continue
        h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
        score = (len(xs) / (h * w + 1e-6)) * (min(h, w) / max(h, w)) * (len(xs) ** 0.25)
        if score > bs:
            bs, best = score, (xs.mean(), ys.mean())
    if best is None:
        return None
    col = round((best[0] - ORIGIN[0]) / CELL) + 1
    row = round((best[1] - ORIGIN[1]) / CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def compose(i, n, move, bc, ac, gt_img, pred_img, det):
    W = PANEL * 2 + PAD * 3
    H = PANEL + PAD * 2 + LABEL_H
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}[move]
    d.text((PAD, 16), f"move {i}/{n}   {move} {arrow}   ({bc[0]},{bc[1]}) → ({ac[0]},{ac[1]})", INK, fnt(26, True))
    d.text((PAD, 52), "latent world model · VAE-decoded to 1024px (no pixelation)", MUTED, fnt(15))
    y = LABEL_H
    for k, (img, cap, col) in enumerate([(gt_img, "GROUND TRUTH", SLATE),
                                         (pred_img, "LATENT-WM PREDICTED", BADGE)]):
        x = PAD + k * (PANEL + PAD)
        cv.paste(img.resize((PANEL, PANEL), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + PANEL, y + PANEL], outline=SLATE, width=1)
        d.text((x, y + PANEL + 8), cap, col, fnt(17, True))
    ok = (det == ac)
    d.text((PAD + PANEL + PAD, y + PANEL + 34),
           f"S detected at {det}  {'✓ matches' if ok else '✗ differs'}",
           HIT if ok else MISS, fnt(15))
    return cv, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/latent_play/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--fps", type=float, default=1.2)
    ap.add_argument("--out", default="runs/latent_play_video")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    net = make_latent_denoiser(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    trans = parse_playthrough()
    frames, hits = [], 0
    for i, (before_png, move, bc, ac, after_png) in enumerate(trans, 1):
        gt_img = Image.open(after_png).convert("RGB")
        lat_before = encode(vae, Image.open(before_png), args.res, device)  # (1,32,128,128)
        prev = lat_before.unsqueeze(1)                                      # (1,1,32,128,128)
        act = torch.tensor([[ACTIONS[move]]], device=device)
        torch.manual_seed(0)
        with torch.no_grad():
            pred_lat, _ = sampler.sample(prev, act)                        # (1,32,128,128)
        pred_img = decode(vae, pred_lat)                                   # crisp 1024px PIL
        det = santa_cell(np.asarray(pred_img.resize((1024, 1024), Image.LANCZOS)))
        cv, ok = compose(i, len(trans), move, bc, ac, gt_img, pred_img, det)
        hits += ok
        frames.append(cv)
        print(f"  move {i}: {bc} --{move}--> {ac} | pred S at {det} {'✓' if ok else '✗'}", flush=True)

    print(f"\nLatent WM reproduced {hits}/{len(trans)} playthrough moves (by detector)")

    tw, th = frames[0].width // 2, frames[0].height // 2
    sheet = Image.new("RGB", (tw * 3, th * 3), BG)
    for k, fr in enumerate(frames):
        sheet.paste(fr.resize((tw, th), Image.LANCZOS), ((k % 3) * tw, (k // 3) * th))
    sheet.save(out / "contact_sheet.png")
    print(f"-> {out}/contact_sheet.png")

    gif = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    gif[0].save(out / "diamond_play.gif", save_all=True, append_images=gif[1:],
                duration=int(1000 / args.fps), loop=0)
    print(f"-> {out}/diamond_play.gif")

    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(out / "diamond_play.mp4", fps=args.fps, codec="libx264", quality=8)
        for f in frames:
            a = np.asarray(f)
            a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]
            w.append_data(a)
        w.close()
        print(f"-> {out}/diamond_play.mp4")
    except Exception as e:
        print(f"(mp4 skipped: {e}; use the gif)")


if __name__ == "__main__":
    main()
