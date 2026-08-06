"""Neural game engine video: the world model renders each frame from (frame, action).

Walks a real MDP action chain from a start state. At each step, produces TWO frames:
  - TEACHER-FORCED: real current frame + action -> model renders next (per-step quality)
  - CLOSED-LOOP: the model's OWN previous output + action -> model renders (self-driving)
Both are conditioned on the heatmap of the destination cell, which the MDP tracks from
the action chain (the model is a controllable renderer; MDP supplies the target cell).

Builds a side-by-side GIF/MP4:  [ teacher-forced | closed-loop | ground truth ] per step.

    python scripts/neural_engine_video.py --game hospital \
        --ckpt runs/final/hospital/denoiser_final.pt --sigma-data 0.612 --steps 8
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
N_ACTIONS = 5
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
ID_KEY = {0: "W(up)", 1: "S(down)", 2: "A(left)", 3: "D(right)", 4: "Space"}


def sid(p):
    return int(Path(p).stem.split("_")[-1])


def build_chain(game, max_len=10):
    """Greedy walk of consecutive transitions from state 0 (prefer moves)."""
    trans = [json.loads(l) for l in open(BUNDLE / game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL]
    by_from = {}
    for t in trans:
        by_from.setdefault(sid(t["obs_t"]), []).append(t)
    chain, cur, seen = [], 0, set()
    while len(chain) < max_len and cur not in seen:
        opts = [t for t in by_from.get(cur, []) if t["effects"].get("moved_player")]
        if not opts:
            break
        seen.add(cur); t = opts[0]; chain.append(t); cur = sid(t["obs_next"])
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--chain-file", default=None,
                    help="JSON trajectory (list of transitions) to play; else greedy walk")
    ap.add_argument("--fps", type=float, default=1.5)
    ap.add_argument("--out", default="runs/neural_engine")
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

    if args.chain_file:
        chain = json.loads(Path(args.chain_file).read_text())
    else:
        chain = build_chain(args.game, args.steps)
    print(f"{args.game} action chain ({len(chain)} steps):")
    for t in chain:
        mp = t["effects"].get("moved_player")
        desc = f"{tuple(mp['from'])}->{tuple(mp['to'])}" if mp \
            else f"ACT {t['effects'].get('completed_object')}"
        print(f"  {t['control']:5} {desc}")

    def set_cond(t):
        net.heatmap = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        net.inner_model.rich = dict(
            target=torch.tensor([f["target"]], device=device),
            hud=torch.tensor([f["hud"]], device=device),
            hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))

    @torch.no_grad()
    def render(cur_lat, act):
        return sampler.sample(cur_lat.unsqueeze(1), torch.tensor([[act]], device=device))[0]

    start_img = Image.open(BUNDLE / args.game / chain[0]["obs_t"]).convert("RGB")
    cl_lat = encode(vae, start_img, args.res, device)      # closed-loop input (1,C,H,W), self-fed

    P = 300
    def fnt(s, b=False):
        try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf", s)
        except Exception: return ImageFont.load_default()

    frames = []
    for i, t in enumerate(chain):
        act = CTRL[t["control"]]
        set_cond(t)
        # teacher-forced: render from the REAL current frame's latent
        real_cur = encode(vae, Image.open(BUNDLE / args.game / t["obs_t"]).convert("RGB"),
                          args.res, device)
        torch.manual_seed(0)
        tf_pred = decode(vae, render(real_cur, act))
        # closed-loop: render from the model's OWN previous latent (keep 4D (1,C,H,W))
        torch.manual_seed(0)
        cl_lat_next = render(cl_lat, act)                   # (1,C,H,W)
        cl_pred = decode(vae, cl_lat_next)
        cl_lat = cl_lat_next                                # feed own output forward (stays 4D)
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")

        panel = Image.new("RGB", (P * 3 + 16, P + 44), (17, 24, 33))
        d = ImageDraw.Draw(panel)
        mp = t["effects"].get("moved_player")
        desc = f"{tuple(mp['from'])}->{tuple(mp['to'])}" if mp \
            else f"treat {t['effects'].get('completed_object')}"
        d.text((8, 6), f"step {i+1}/{len(chain)}   action {ID_KEY[act]}   {desc}",
               (231, 238, 245), fnt(15, True))
        for k, (im, cap, col) in enumerate([
                (tf_pred, "TEACHER-FORCED", (60, 110, 165)),
                (cl_pred, "CLOSED-LOOP (engine)", (198, 40, 40)),
                (gt, "ground truth", (140, 160, 180))]):
            x = 8 + k * (P + 4)
            panel.paste(im.convert("RGB").resize((P, P)), (x, 32))
            d.text((x, 32 + P + 1), cap, col, fnt(12, True))
        frames.append(panel)
        print(f"  rendered step {i+1}", flush=True)

    gif = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    gif[0].save(out / "neural_engine.gif", save_all=True, append_images=gif[1:],
                duration=int(1000 / args.fps), loop=0)
    print(f"-> {out}/neural_engine.gif")
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(out / "neural_engine.mp4", fps=args.fps, codec="libx264", quality=8)
        for f in frames:
            a = np.asarray(f); a = a[:a.shape[0]//2*2, :a.shape[1]//2*2]
            w.append_data(a)
        w.close()
        print(f"-> {out}/neural_engine.mp4")
    except Exception as e:
        print(f"(mp4 skipped: {e})")


if __name__ == "__main__":
    main()
