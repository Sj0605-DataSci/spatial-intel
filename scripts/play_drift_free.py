"""Drift-free neural game engine: play a full game where each step's INPUT is a
CLEAN COMPOSITED frame (actor@cell + HUD + effects from the MDP state), NOT the
model's own previous output. So errors never compound -> no deterioration.

Per step:  MDP state -> composite clean frame -> model renders next -> re-composite
next clean state (from MDP). Video: composited-input | model-rendered | ground-truth.

    python scripts/play_drift_free.py --game hospital \
        --ckpt runs/final/hospital/denoiser_final.pt --sigma-data 0.612
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
from composite_frame import composite, _hud_state_index, rddl_positions

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
ID_KEY = {0: "W", 1: "S", 2: "A", 3: "D", 4: "Space"}


def sid(p):
    return int(Path(p).stem.split("_")[-1])


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
    ap.add_argument("--chain-file", default="/tmp/hospital_full_game.json")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--out", default="runs/drift_free")
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

    states = {json.loads(l)["id"]: json.loads(l) for l in open(BUNDLE / args.game / "states.jsonl")}
    hud_idx = _hud_state_index(args.game)
    chain = json.loads(Path(args.chain_file).read_text())
    print(f"{args.game} drift-free playthrough: {len(chain)} steps")

    P = 300
    frames = []
    for i, t in enumerate(chain):
        s_before = states[sid(t["obs_t"])]
        # 1) COMPOSITE the clean current-state frame (actor@cell + HUD from ints)
        clean_input = composite(args.game, (s_before["px"], s_before["py"]),
                                ints=s_before["ints"], hud_index=hud_idx)
        # 2) model renders the next frame from the CLEAN input + action + effects
        lat = encode(vae, clean_input, args.res, device).unsqueeze(1)
        act = torch.tensor([[CTRL[t["control"]]]], device=device)
        net.heatmap = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        net.inner_model.rich = dict(
            target=torch.tensor([f["target"]], device=device),
            hud=torch.tensor([f["hud"]], device=device),
            hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))
        torch.manual_seed(0)
        with torch.no_grad():
            rendered = decode(vae, sampler.sample(lat, act)[0])
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")

        panel = Image.new("RGB", (P * 3 + 16, P + 44), (17, 24, 33))
        d = ImageDraw.Draw(panel)
        mp = t["effects"].get("moved_player")
        desc = f"{tuple(mp['from'])}->{tuple(mp['to'])}" if mp else f"treat {t['target']}"
        d.text((8, 6), f"step {i+1}/{len(chain)}   {ID_KEY[CTRL[t['control']]]}   {desc}",
               (231, 238, 245), fnt(15, True))
        for k, (im, cap, col) in enumerate([
                (clean_input, "COMPOSITED input (clean)", (46, 125, 91)),
                (rendered, "MODEL rendered", (198, 40, 40)),
                (gt, "ground truth", (140, 160, 180))]):
            x = 8 + k * (P + 4)
            panel.paste(im.convert("RGB").resize((P, P)), (x, 32))
            d.text((x, 32 + P + 1), cap, col, fnt(12, True))
        frames.append(panel)
        print(f"  step {i+1} rendered", flush=True)

    gif = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    gif[0].save(out / "drift_free.gif", save_all=True, append_images=gif[1:],
                duration=int(1000 / args.fps), loop=0)
    print(f"-> {out}/drift_free.gif")
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(out / "drift_free.mp4", fps=args.fps, codec="libx264", quality=8)
        for f in frames:
            a = np.asarray(f); a = a[:a.shape[0]//2*2, :a.shape[1]//2*2]
            w.append_data(a)
        w.close()
        print(f"-> {out}/drift_free.mp4")
    except Exception as e:
        print(f"(mp4 skipped: {e})")


if __name__ == "__main__":
    main()
