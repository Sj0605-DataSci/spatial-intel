"""Closed-loop drift: ORIGINAL model vs NOISE-AUG model, both PLAIN closed-loop
(feed own output forward) over a full trajectory. Does noise-augmentation stop drift?

  [ ORIGINAL closed-loop | NOISE-AUG closed-loop | ground truth ] per step.

    python scripts/closed_loop_compare.py --game hospital --sigma-data 0.612 \
        --orig runs/final/hospital/denoiser_final.pt \
        --noiseaug runs/hospital_noiseaug/denoiser_best.pt \
        --chain-file /tmp/hospital_full_game.json
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

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode
from rich_wm import make_rich_denoiser, RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
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
    ap.add_argument("--orig", required=True)
    ap.add_argument("--noiseaug", required=True)
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--chain-file", default="/tmp/hospital_full_game.json")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--out", default="runs/closed_loop_compare")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    vae = load_vae(device)
    rf = RichFeatures(args.game); geom = load_geom(args.game)

    def load(path):
        m = make_rich_denoiser(device, 5, len(rf.targets), len(rf.hud_keys),
                               cond_channels=args.cond_channels, sigma_data=args.sigma_data)
        m.load_state_dict(torch.load(path, map_location=device)); m.eval()
        return m, DiffusionSampler(m, DiffusionSamplerConfig(
            num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1))
    orig, s_orig = load(args.orig)
    naug, s_naug = load(args.noiseaug)

    chain = json.loads(Path(args.chain_file).read_text())
    print(f"{args.game} closed-loop: ORIGINAL vs NOISE-AUG, {len(chain)} steps")

    def set_cond(net, t):
        net.heatmap = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        net.inner_model.rich = dict(
            target=torch.tensor([f["target"]], device=device),
            hud=torch.tensor([f["hud"]], device=device),
            hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))

    start = Image.open(BUNDLE / args.game / chain[0]["obs_t"]).convert("RGB")
    orig_lat = encode(vae, start, args.res, device)
    naug_lat = orig_lat.clone()

    P = 280
    frames = []
    for i, t in enumerate(chain):
        act = torch.tensor([[CTRL[t["control"]]]], device=device)
        set_cond(orig, t)
        torch.manual_seed(0)
        with torch.no_grad():
            orig_next = s_orig.sample(orig_lat.unsqueeze(1), act)[0]
        orig_lat = orig_next
        set_cond(naug, t)
        torch.manual_seed(0)
        with torch.no_grad():
            naug_next = s_naug.sample(naug_lat.unsqueeze(1), act)[0]
        naug_lat = naug_next
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")

        panel = Image.new("RGB", (P*3 + 16, P + 44), (17, 24, 33))
        d = ImageDraw.Draw(panel)
        mp = t["effects"].get("moved_player")
        desc = f"{tuple(mp['from'])}->{tuple(mp['to'])}" if mp else "treat"
        d.text((8, 6), f"step {i+1}/{len(chain)}  {ID_KEY[CTRL[t['control']]]}  {desc}",
               (231, 238, 245), fnt(15, True))
        for k, (im, cap, col) in enumerate([
                (decode(vae, orig_next), "ORIGINAL closed-loop", (198, 40, 40)),
                (decode(vae, naug_next), "NOISE-AUG closed-loop", (46, 125, 91)),
                (gt, "ground truth", (140, 160, 180))]):
            x = 8 + k*(P+4)
            panel.paste(im.convert("RGB").resize((P, P)), (x, 32))
            d.text((x, 32+P+1), cap, col, fnt(12, True))
        frames.append(panel)
        print(f"  step {i+1} done", flush=True)

    gif = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    gif[0].save(out / "closed_loop_compare.gif", save_all=True, append_images=gif[1:],
                duration=int(1000/args.fps), loop=0)
    print(f"-> {out}/closed_loop_compare.gif")
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(out / "closed_loop_compare.mp4", fps=args.fps, codec="libx264", quality=8)
        for f in frames:
            a = np.asarray(f); a = a[:a.shape[0]//2*2, :a.shape[1]//2*2]
            w.append_data(a)
        w.close()
        print(f"-> {out}/closed_loop_compare.mp4")
    except Exception as e:
        print(f"(mp4 skipped: {e})")


if __name__ == "__main__":
    main()
