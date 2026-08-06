"""Record 'DIAMOND plays the Santa playthrough': for each of the 9 moves in
runs/diamond_dev/playthrough_demo, show the ground-truth frame beside DIAMOND's
prediction (before-frame + action). Upscaled, captioned, with detector verdict.

Writes to runs/diamond_play_video/:
  diamond_play.gif       (always)
  contact_sheet.png      (always; 3x3 of the 9 steps)
  diamond_play.mp4       (if imageio available)

    python scripts/record_diamond_playthrough.py --ckpt runs/diamond_play/denoiser_final.pt --res 128
"""
import argparse
import sys
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from train_diamond_santa import make_denoiser, load_img, ACTIONS
from train_diamond_playthrough import parse_playthrough

PANEL = 384
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
    """Locate the S badge = a solid ROUND red disk. Blurry 128px predictions
    desaturate the red, so use relaxed thresholds AND score blobs by how
    disk-like they are (the badge is round; presents are square/bowed)."""
    from scipy import ndimage
    R, G, B = arr1024[..., 0].astype(int), arr1024[..., 1].astype(int), arr1024[..., 2].astype(int)
    # relaxed red mask — tolerate the dim, soft badge in blurry frames
    red = (R > 140) & (R - G > 45) & (R - B > 45)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    best, best_score = None, -1.0
    for k in range(1, n + 1):
        ys, xs = np.where(lbl == k)
        area = len(xs)
        if area < 120:                      # ignore specks
            continue
        h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
        # roundness: badge fills its bbox (~pi/4) AND is near-square
        fill = area / (h * w + 1e-6)
        aspect = min(h, w) / max(h, w)
        roundness = fill * aspect
        score = roundness * (area ** 0.25)  # prefer round, then larger
        if score > best_score:
            best_score, best = score, (xs.mean(), ys.mean())
    if best is None:
        return None
    cx, cy = best
    col = round((cx - ORIGIN[0]) / CELL) + 1
    row = round((cy - ORIGIN[1]) / CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def to_uint8(chw):
    return chw.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()


def load_sr(path, device):
    """Load a Real-ESRGAN weight via spandrel -> callable upscaler (PIL->PIL, 4x)."""
    from spandrel import ModelLoader
    model = ModelLoader().load_from_file(path).to(device).eval()

    @torch.no_grad()
    def upscale(pil_img):
        a = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((out * 255).round().astype(np.uint8))
    return upscale


def compose(i, n, move, bc, ac, gt_img, pred_img, det):
    W = PANEL * 2 + PAD * 3
    H = PANEL + PAD * 2 + LABEL_H
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}[move]
    d.text((PAD, 16), f"move {i}/{n}   {move} {arrow}   ({bc[0]},{bc[1]}) → ({ac[0]},{ac[1]})", INK, fnt(26, True))
    d.text((PAD, 52), "DIAMOND predicts the next frame from (current frame + action)", MUTED, fnt(15))
    y = LABEL_H
    for k, (img, cap, col) in enumerate([(gt_img, "GROUND TRUTH", SLATE),
                                         (pred_img, "DIAMOND PREDICTED", BADGE)]):
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
    ap.add_argument("--ckpt", default="runs/diamond_play/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--fps", type=float, default=1.2)
    ap.add_argument("--out", default="runs/diamond_play_video")
    ap.add_argument("--sr", default="models/esrgan/RealESRGAN_x4plus.pth",
                    help="Real-ESRGAN weight to sharpen predictions; '' to disable")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    sr = None
    if args.sr and Path(args.sr).exists():
        sr = load_sr(args.sr, device)
        print(f"super-res: Real-ESRGAN x4 ({args.sr})")

    net = make_denoiser(args.res, device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    trans = parse_playthrough()
    frames, hits = [], 0
    for i, (before_png, move, bc, ac, after_png) in enumerate(trans, 1):
        gt_img = Image.open(after_png).convert("RGB")
        before = load_img(before_png, args.res, device)
        act = torch.tensor([[ACTIONS[move]]], device=device)
        torch.manual_seed(0)
        with torch.no_grad():
            pred, _ = sampler.sample(before.unsqueeze(0).unsqueeze(0), act)
        raw_img = Image.fromarray(to_uint8(pred[0]))         # native 128px
        # detect on the RAW prediction (SR sharpens presents into badge-like blobs)
        det = santa_cell(np.asarray(raw_img.resize((1024, 1024), Image.NEAREST)))
        disp_img = sr(raw_img) if sr is not None else raw_img  # sharpened for display
        cv, ok = compose(i, len(trans), move, bc, ac, gt_img, disp_img, det)
        hits += ok
        frames.append(cv)
        print(f"  move {i}: {bc} --{move}--> {ac} | DIAMOND S at {det} {'✓' if ok else '✗'}", flush=True)

    print(f"\nDIAMOND reproduced {hits}/{len(trans)} playthrough moves (by detector)")

    # contact sheet 3x3
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
