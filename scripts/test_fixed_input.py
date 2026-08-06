"""Does the model LEARN DYNAMICS or just edit a matched input?

Feed ONE FIXED input image (state 0) for EVERY transition, varying only the
action + conditioning (heatmap/effects). If the model renders the correct DIFFERENT
next-states from the same fixed input, it's computing dynamics from the conditioning
(real research). If it just reproduces the fixed input, it's edit-copying.

Compares, per transition:
  [ fixed input | model(fixed input, action) | model(REAL matched input, action) | ground truth ]
so we see fixed-input vs matched-input vs truth.

    python scripts/test_fixed_input.py --game hospital \
        --ckpt runs/final/hospital/denoiser_final.pt --sigma-data 0.612 --n 6
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
from pytorch_msssim import ssim as ssim_fn

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode
from rich_wm import make_rich_denoiser, RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
ID_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}


def fnt(s, b=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", s)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--fixed-frame", default=None, help="which frame to fix; default=state 0")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="runs/fixed_input_test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    rf = RichFeatures(args.game); geom = load_geom(args.game)
    net = make_rich_denoiser(device, N_ACTIONS, len(rf.targets), len(rf.hud_keys),
                             cond_channels=args.cond_channels, sigma_data=args.sigma_data)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)); net.eval()
    sampler = DiffusionSampler(net, DiffusionSamplerConfig(
        num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))

    # THE FIXED input: state 0 (player at 1,1), used for every transition
    fixed_path = args.fixed_frame or (BUNDLE / args.game / "states/state_00000.png")
    fixed_img = Image.open(fixed_path).convert("RGB")
    fixed_lat = encode(vae, fixed_img, args.res, device).unsqueeze(1)

    # pick transitions with DIFFERENT destinations (to show varied dynamics)
    trans = [json.loads(l) for l in open(BUNDLE / args.game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL and json.loads(l)["effects"].get("moved_player")]
    import random
    random.Random(3).shuffle(trans)
    # dedup by (from,to) so destinations vary
    seen, pick = set(), []
    for t in trans:
        mp = t["effects"]["moved_player"]
        k = (tuple(mp["from"]), tuple(mp["to"]))
        if k not in seen:
            seen.add(k); pick.append(t)
        if len(pick) >= args.n:
            break

    def render(lat, t):
        net.heatmap = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        net.inner_model.rich = dict(
            target=torch.tensor([f["target"]], device=device),
            hud=torch.tensor([f["hud"]], device=device),
            hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))
        act = torch.tensor([[CTRL[t["control"]]]], device=device)
        torch.manual_seed(0)
        with torch.no_grad():
            return decode(vae, sampler.sample(lat, act)[0])

    P = 240
    rows_img = []
    ss_fixed, ss_matched = [], []
    for t in pick:
        mp = t["effects"]["moved_player"]
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")
        real_in = Image.open(BUNDLE / args.game / t["obs_t"]).convert("RGB")
        real_lat = encode(vae, real_in, args.res, device).unsqueeze(1)

        pred_fixed = render(fixed_lat, t)      # FIXED input -> can it still get it right?
        pred_matched = render(real_lat, t)     # matched input (normal teacher-forcing)

        sz = gt.size
        def s(im): return ssim_fn(
            torch.from_numpy(np.asarray(im.resize(sz)).astype(np.float32)/255).permute(2,0,1).unsqueeze(0),
            torch.from_numpy(np.asarray(gt).astype(np.float32)/255).permute(2,0,1).unsqueeze(0),
            data_range=1.0).item()
        ss_fixed.append(s(pred_fixed)); ss_matched.append(s(pred_matched))

        strip = Image.new("RGB", (P*4 + 20, P + 44), (17, 24, 33))
        d = ImageDraw.Draw(strip)
        d.text((8, 6), f"{t['control']} {tuple(mp['from'])}->{tuple(mp['to'])}   "
                       f"fixed-SSIM {ss_fixed[-1]:.2f}  matched-SSIM {ss_matched[-1]:.2f}",
               (231, 238, 245), fnt(13, True))
        for k, (im, cap, col) in enumerate([
                (fixed_img, "FIXED input (state 0)", (140, 160, 180)),
                (pred_fixed, "model(FIXED, action)", (198, 40, 40)),
                (pred_matched, "model(matched, action)", (60, 110, 165)),
                (gt, "ground truth", (140, 160, 180))]):
            x = 8 + k*(P+4)
            strip.paste(im.convert("RGB").resize((P, P)), (x, 30))
            d.text((x, 30+P+1), cap, col, fnt(11, True))
        rows_img.append(strip)
        print(f"  {t['control']} {tuple(mp['from'])}->{tuple(mp['to'])}: "
              f"fixed {ss_fixed[-1]:.2f} matched {ss_matched[-1]:.2f}", flush=True)

    sheet = Image.new("RGB", (rows_img[0].width, sum(r.height for r in rows_img)), (17, 24, 33))
    y = 0
    for r in rows_img:
        sheet.paste(r, (0, y)); y += r.height
    sheet.save(out / "fixed_vs_matched.png")
    print(f"\n=== FIXED-input vs MATCHED-input (n={len(pick)}) ===")
    print(f"  model(FIXED input):   mean SSIM {np.mean(ss_fixed):.3f}")
    print(f"  model(matched input): mean SSIM {np.mean(ss_matched):.3f}")
    print(f"  if FIXED ~ matched: model USES DYNAMICS (not just input-editing)")
    print(f"  if FIXED << matched: model relies on the matched input (edit-copying)")
    print(f"-> {out}/fixed_vs_matched.png")


if __name__ == "__main__":
    main()
