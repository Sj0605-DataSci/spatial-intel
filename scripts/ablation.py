"""6-arm reasoning-conditioning ablation (the notebook's experiment ladder).

The image task is held TRIVIAL (move the red triangle from cell A to cell B). The
variable under study is HOW MUCH reasoning scaffolding conditions the diffusion
model. Each arm builds a different prompt; all generate with the SAME base-4B and
are scored with the SAME spatial-accuracy metric (did the triangle land on the
exact target cell).

Arms (matching the notebook):
  1 text            : bare instruction (baseline)
  2 text+coords     : instruction + explicit "TARGET=(col,row)"
  3 text+cot+coords : instruction + a chain-of-thought derivation + coords
  4 text+vlm+coords : instruction + a small-VLM's grounded reasoning + coords
  5 nonverbose      : terse localised instruction (low verbosity)
  6 verbose         : rich localised instruction (high verbosity)

Run one arm at a time (each loads heavy models):
    python scripts/ablation.py --arm 4 --vlm qwen   --n 12 --out runs/abl_arm4_qwen
    python scripts/ablation.py --arm 4 --vlm vigorl --n 12 --out runs/abl_arm4_vigorl
    python scripts/ablation.py --arm 1 --n 12 --out runs/abl_arm1

Uses the SAME fixed eval seed as eval_spatial.py so every arm sees identical
commands -> the success rates are directly comparable across arms.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import (GridSpec, render_state, to_array, apply_move,
                                 clamp_topleft, max_steps_for, DIRECTION_NAMES)
from spatial_intel.dataset import instruction_text
from spatial_intel.reward import RewardConfig, score_attempt

FLUX = "black-forest-labs/FLUX.2-klein-base-4B"


def sample_commands(spec, n, rng):
    cmds = []
    while len(cmds) < n:
        start = (int(rng.integers(0, spec.max_topleft + 1)),
                 int(rng.integers(0, spec.max_topleft + 1)))
        d = DIRECTION_NAMES[int(rng.integers(len(DIRECTION_NAMES)))]
        hi = max_steps_for(spec, start, d)
        if hi < 1:
            continue
        mv = int(rng.integers(1, hi + 1))
        target = clamp_topleft(spec, apply_move(spec, start, d, mv))
        if target == start:
            continue
        cmds.append((start, d, mv, target))
    return cmds


# -- the six conditioning builders ---------------------------------------
def cot_text(start, d, mv, target):
    return (f"Reasoning: the triangle is at cell {tuple(start)}. The command is "
            f"'{d} by {mv}'. Moving {d} by {mv} from {tuple(start)} gives "
            f"{tuple(target)}. So it must be drawn in cell {tuple(target)}.")


def build_prompt(arm, spec, start, d, mv, target, vlm_reasoning=None):
    instr = instruction_text(d, mv)
    base = ("A square grid with numbered columns and rows and a single solid red "
            "triangle occupying one square block. " + instr +
            " Keep the grid, axes and labels identical; move ONLY the red triangle; "
            "output one solid red triangle snapped to the grid.")
    coords = f" The triangle must end up in grid cell TARGET=({target[0]},{target[1]})."
    if arm == 1:                                   # text only
        return base
    if arm == 2:                                   # text + coords
        return base + coords
    if arm == 3:                                   # text + CoT + coords
        return base + " " + cot_text(start, d, mv, target) + coords
    if arm == 4:                                   # text + VLM reasoning + coords
        return base + " " + (vlm_reasoning or "") + coords
    if arm == 5:                                   # non-verbose (terse)
        return (f"Grid image. Move red triangle to cell ({target[0]},{target[1]}). "
                "One red triangle, snapped to that cell.")
    if arm == 6:                                   # verbose
        return (base + coords + f" Specifically, place the triangle so its centre "
                f"sits exactly inside the single grid square at column {target[0]} "
                f"and row {target[1]}, fully contained within that cell's borders, "
                "not overlapping neighbouring cells, with the same size and crisp "
                "triangular shape as the original, and leave every grid line, axis "
                "tick and coordinate label unchanged.")
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--vlm", choices=["qwen", "vigorl"], default="qwen",
                    help="which VLM for arm 4")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=12345, help="MUST match eval_spatial for comparability")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    rng = np.random.default_rng(args.seed)
    cmds = sample_commands(spec, args.n, rng)

    # arm 4 needs a VLM reasoner
    reasoner = None
    if args.arm == 4:
        from spatial_intel.rl.vlm_reasoner import VLMReasoner, VLMConfig, QWEN_ID, VIGORL_ID
        mid = QWEN_ID if args.vlm == "qwen" else VIGORL_ID
        print(f"loading VLM reasoner: {mid}")
        reasoner = VLMReasoner(VLMConfig(model_id=mid, max_new_tokens=128))
        reasoner.load()

    print("loading base-4B...")
    pipe = Flux2KleinPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    results = []
    for ci, (start, d, mv, target) in enumerate(cmds):
        vlm_reasoning = None
        if args.arm == 4:
            vlm_reasoning = reasoner.reason(render_state(spec, start), instruction_text(d, mv))
        prompt = build_prompt(args.arm, spec, start, d, mv, target, vlm_reasoning)
        gen = torch.Generator(device="cuda").manual_seed(1000 + ci)
        img = pipe(image=render_state(spec, start).resize((512, 512)), prompt=prompt,
                   num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=512, width=512, generator=gen).images[0]
        sc = score_attempt(spec, to_array(img.resize((spec.img_w, spec.img_h))),
                           start_topleft=start, target_topleft=target, direction=d,
                           cfg=RewardConfig())
        if args.save_images:
            img.save(out / f"cmd{ci:02d}.png")
        rec = dict(cmd=ci, start=list(start), direction=d, move=mv, target=list(target),
                   success=bool(sc.success), cell_distance=round(sc.cell_distance, 2),
                   detected=sc.detected_topleft, vlm_reasoning=vlm_reasoning)
        results.append(rec)
        print(f"cmd {ci:02d} {d:>10} x{mv} {start}->{target}: "
              f"success={sc.success} dist={sc.cell_distance:.1f}"
              + (f" | vlm: {vlm_reasoning[:60]}" if vlm_reasoning else ""), flush=True)

    succ = float(np.mean([r["success"] for r in results]))
    mdist = float(np.mean([r["cell_distance"] for r in results]))
    agg = dict(arm=args.arm, vlm=args.vlm if args.arm == 4 else None, n=len(results),
               success_rate=round(succ, 3), mean_cell_distance=round(mdist, 2))
    (out / "ablation.json").write_text(json.dumps(dict(agg=agg, results=results), indent=2))
    print(f"\n=== ARM {args.arm}"
          + (f" ({args.vlm})" if args.arm == 4 else "")
          + f" === success_rate={succ:.3f} mean_dist={mdist:.2f}  -> {out}/ablation.json")


if __name__ == "__main__":
    main()
