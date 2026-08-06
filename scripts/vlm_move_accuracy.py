"""VLM (InternVL3) move-accuracy: ask the VLM which grid cell the movable character
is in, for BOTH the model's predicted frame AND the ground-truth frame, and score
against the game's true cell. Robust replacement for the brittle blob detector.

Coordinate convention (0-indexed, top-left = (0,0)):
  grid is COLS x ROWS; a cell is (col, row) with col in 0..COLS-1, row in 0..ROWS-1.
  The game's states.jsonl is 1-indexed with py=1 at the BOTTOM (y-axis flipped),
  so the true cell (top-left origin) = (px-1, ROWS-py).

For each val transition and each model:
  - decode the predicted next frame
  - ask InternVL "which cell is <char> in?" on prediction AND ground truth
  - compare to the true arrival cell.
Reports, per model: VLM-acc on predictions, and (sanity) VLM-acc on ground truth.

    python scripts/vlm_move_accuracy.py --game hospital --sigma-data 0.612 \
        --ckpts plain=runs/spec_hospital_800/... rich=... richlpips=... --n 12
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image

from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from latent_wm import load_vae, encode, decode, make_latent_denoiser
from rich_wm import make_rich_denoiser, RichFeatures, build_heatmap, load_geom

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5
CTRL = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
INTERNVL = "OpenGVLab/InternVL3-2B-hf"

# short human description of the movable character per game (helps the VLM find it)
CHAR_DESC = {
    "hospital": "the man in a blue business suit (the admin player), NOT the doctor, nurse, or patients",
}


def true_cell(state, rows):
    """states.jsonl px,py (1-indexed, py=1 bottom) -> 0-indexed top-left (col,row)."""
    return (state["px"] - 1, rows - state["py"])


def state_index(game):
    return {json.loads(l)["id"]: json.loads(l) for l in open(BUNDLE / game / "states.jsonl")}


def obs_id(p):
    return int(Path(p).stem.split("_")[-1])


class VLMJudge:
    def __init__(self, device):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(INTERNVL, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            INTERNVL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
        self.device = device

    def cell(self, img, cols, rows, char_desc):
        prompt = (
            f"This is a top-down {cols}x{rows} grid game board. Columns are numbered "
            f"0 to {cols-1} left-to-right; rows are numbered 0 to {rows-1} top-to-bottom, "
            f"so the top-left cell is (0,0). Find {char_desc}. "
            f"Answer with ONLY its cell as (col,row), e.g. (2,3). If you cannot find it, answer (-1,-1)."
        )
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        inputs = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        txt = self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", txt)
        return (int(m.group(1)), int(m.group(2))) if m else None, txt.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--sigma-data", type=float, default=0.612)
    ap.add_argument("--cond-channels", type=int, default=512)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--denoise-steps", type=int, default=30)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name=path entries, e.g. plain=... rich=... richlpips=...")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="runs/vlm_acc")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    cols, rows = json.load(open(BUNDLE / args.game / "meta.json"))["grid"]
    char = CHAR_DESC.get(args.game, "the movable player character")
    states = state_index(args.game)
    rf = RichFeatures(args.game); geom = load_geom(args.game)

    vae = load_vae(device)
    ckpts = dict(kv.split("=", 1) for kv in args.ckpts)
    models = {}
    for name, path in ckpts.items():
        if "rich" in name:
            m = make_rich_denoiser(device, N_ACTIONS, len(rf.targets), len(rf.hud_keys),
                                   cond_channels=args.cond_channels, sigma_data=args.sigma_data)
        else:
            m = make_latent_denoiser(device, num_actions=N_ACTIONS,
                                     cond_channels=args.cond_channels, sigma_data=args.sigma_data)
        m.load_state_dict(torch.load(path, map_location=device)); m.eval()
        models[name] = m

    trans = [json.loads(l) for l in open(BUNDLE / args.game / "transitions.jsonl")
             if json.loads(l).get("control") in CTRL and json.loads(l)["effects"].get("moved_player")]
    import random
    random.Random(2).shuffle(trans); trans = trans[:args.n]

    print("loading InternVL3-2B judge...")
    judge = VLMJudge(device)

    scfg = dict(num_steps_denoising=args.denoise_steps, sigma_min=2e-3, sigma_max=20, rho=7, order=1)
    res = {name: {"pred_hit": [], "gt_hit": []} for name in models}
    for t in trans:
        obs_t = Image.open(BUNDLE / args.game / t["obs_t"]).convert("RGB")
        gt = Image.open(BUNDLE / args.game / t["obs_next"]).convert("RGB")
        tc = true_cell(states[obs_id(t["obs_next"])], rows)   # true arrival cell (0-idx top-left)
        lat = encode(vae, obs_t, args.res, device).unsqueeze(1)
        act = torch.tensor([[CTRL[t["control"]]]], device=device)
        # VLM on ground truth once (sanity: can the VLM even read it?)
        gt_cell, _ = judge.cell(gt, cols, rows, char)

        hm = build_heatmap(geom, t["effects"], device).unsqueeze(0)
        f = rf.encode(t)
        richd = dict(target=torch.tensor([f["target"]], device=device),
                     hud=torch.tensor([f["hud"]], device=device),
                     hud_delta=torch.tensor([f["hud_delta"]], device=device, dtype=torch.float32))
        for name, net in models.items():
            if hasattr(net, "heatmap"):
                net.heatmap = hm; net.inner_model.rich = richd
            sm = DiffusionSampler(net, DiffusionSamplerConfig(**scfg))
            torch.manual_seed(0)
            with torch.no_grad():
                pred = decode(vae, sm.sample(lat, act)[0])
            pcell, _ = judge.cell(pred, cols, rows, char)
            res[name]["pred_hit"].append(pcell == tc)
            res[name]["gt_hit"].append(gt_cell == tc)

    print(f"\n=== VLM (InternVL3) move-accuracy — {args.game} ({cols}x{rows}, 0-indexed) ===")
    print(f"  {'model':12} {'pred-acc':>9} {'(GT-read sanity)':>18}")
    for name, d in res.items():
        pa = np.mean(d["pred_hit"]); ga = np.mean(d["gt_hit"])
        print(f"  {name:12} {pa*100:>7.0f}% {ga*100:>16.0f}%")
    print(f"  n={args.n}. pred-acc = VLM sees the actor in the correct cell in the "
          f"predicted frame. GT-read = VLM accuracy on the real frame (upper bound).")
    (out / "vlm_acc.json").write_text(json.dumps(
        {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in res.items()}, indent=2))


if __name__ == "__main__":
    main()
