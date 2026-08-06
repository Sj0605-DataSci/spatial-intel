"""Record the 'DIAMOND learned to play' video: for each of the 10 trajectory
steps, show the ground-truth playthrough frame beside DIAMOND's prediction
(from that step's before-frame + action). Upscaled, captioned.

Writes (whatever encoders are available):
  runs/diamond_traj_video/diamond_play.mp4   (if imageio-ffmpeg / cv2 present)
  runs/diamond_traj_video/diamond_play.gif   (always)
  runs/diamond_traj_video/contact_sheet.png  (always — all 10 steps at a glance)

    python scripts/record_diamond_traj.py --ckpt runs/diamond_traj/denoiser_final.pt --res 128
"""
import argparse
import json
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
from train_diamond_traj import TRAJ

PANEL = 384          # each frame drawn this big
PAD = 24
LABEL_H = 90
BG = (14, 22, 32); INK = (231, 238, 245); MUTED = (147, 164, 182)
BADGE = (198, 40, 40); HIT = (46, 125, 91); MISS = (192, 135, 58); SLATE = (95, 114, 133)
ORIGIN = (227, 227); CELL = 87.5


def font(sz, bold=False):
    for name in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def santa_cell(arr1024):
    from scipy import ndimage
    R, G, B = arr1024[..., 0].astype(int), arr1024[..., 1].astype(int), arr1024[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
    col = round((xs.mean() - ORIGIN[0]) / CELL) + 1
    row = round((ys.mean() - ORIGIN[1]) / CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def to_uint8(chw):
    return chw.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()


def compose(step, n, move, before_cell, after_cell, gt_img, pred_img, det, blocked):
    W = PANEL * 2 + PAD * 3
    H = PANEL + PAD * 2 + LABEL_H
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    f_h = font(26, True); f_b = font(17); f_s = font(15)

    # header
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}[move]
    title = f"step {step+1}/{n}   {move} {arrow}   ({before_cell[0]},{before_cell[1]}) → ({after_cell[0]},{after_cell[1]})"
    d.text((PAD, 18), title, INK, f_h)
    tag = "wall bump — Santa stays put" if blocked else "single-cell move"
    d.text((PAD, 54), tag, MISS if blocked else MUTED, f_s)

    y = LABEL_H
    for i, (img, cap) in enumerate([(gt_img, "GROUND TRUTH"), (pred_img, "DIAMOND PREDICTED")]):
        x = PAD + i * (PANEL + PAD)
        im = img.resize((PANEL, PANEL), Image.LANCZOS)
        cv.paste(im, (x, y))
        d.rectangle([x, y, x + PANEL, y + PANEL], outline=SLATE, width=1)
        cc = BADGE if i == 1 else SLATE
        d.text((x, y + PANEL + 8), cap, cc, f_b)
    # verdict on the predicted panel
    ok = (det == after_cell)
    vx = PAD + PANEL + PAD
    vtxt = f"S detected at {det}  {'✓ matches' if ok else '✗ differs'}"
    d.text((vx, y + PANEL + 34), vtxt, HIT if ok else MISS, f_s)
    return cv, (det == after_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/diamond_traj/denoiser_final.pt")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--fps", type=float, default=1.2)
    ap.add_argument("--out", default="runs/diamond_traj_video")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    net = make_denoiser(args.res, device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    meta = json.loads((TRAJ / "manifest.json").read_text())
    frames, hits = [], 0
    for x in meta:
        i = x["idx"]; move = x["move"]
        bcell = tuple(x["santa_cell_before"]); acell = tuple(x["santa_cell_after"])
        gt_img = Image.open(TRAJ / "frames" / f"{i:05d}_after.png").convert("RGB")
        before = load_img(TRAJ / "frames" / f"{i:05d}_before.png", args.res, device)
        act = torch.tensor([[ACTIONS[move]]], device=device)
        torch.manual_seed(0)
        with torch.no_grad():
            pred, _ = sampler.sample(before.unsqueeze(0).unsqueeze(0), act)
        pred_arr = to_uint8(pred[0])
        pred_img = Image.fromarray(pred_arr)
        det = santa_cell(np.asarray(pred_img.resize((1024, 1024), Image.NEAREST)))
        canvas, ok = compose(i, len(meta), move, bcell, acell, gt_img, pred_img, det, x.get("blocked", False))
        hits += ok
        frames.append(canvas)

    print(f"DIAMOND reproduced {hits}/{len(meta)} trajectory steps (by detector)")

    # contact sheet (2 cols x 5 rows)
    tw = frames[0].width // 2; th = frames[0].height // 2
    cols, rows = 2, 5
    sheet = Image.new("RGB", (tw * cols, th * rows), BG)
    for k, fr in enumerate(frames):
        sheet.paste(fr.resize((tw, th), Image.LANCZOS), ((k % cols) * tw, (k // cols) * th))
    sheet.save(out / "contact_sheet.png")
    print(f"-> {out}/contact_sheet.png")

    # GIF (always works, no deps)
    gif = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    dur = int(1000 / args.fps)
    gif[0].save(out / "diamond_play.gif", save_all=True, append_images=gif[1:],
                duration=dur, loop=0)
    print(f"-> {out}/diamond_play.gif")

    # MP4 if an encoder is available
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(out / "diamond_play.mp4", fps=args.fps,
                               codec="libx264", quality=8)
        for f in frames:
            a = np.asarray(f)
            a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]  # even dims for h264
            w.append_data(a)
        w.close()
        print(f"-> {out}/diamond_play.mp4")
    except Exception as e:
        print(f"(mp4 skipped: {e}; use the gif)")


if __name__ == "__main__":
    main()
