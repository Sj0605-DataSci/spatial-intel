"""Evaluate spatial accuracy across MANY varied commands — the metric that shows
generalization (spatial intelligence), not overfitting to one task.

Runs a fixed, seeded set of N commands (varied start / direction / distance) on a
model and reports, per command and in aggregate:
  - first-try success (did the triangle land on the exact target block at iter 0?)
  - iters-to-converge under the feedback loop (cap --max-iters)
  - mean cell-distance error

Run it on the BASE model first (baseline), then re-run with --lora <path> after
RLVR training to see whether spatial accuracy improved on the SAME held-out
commands. A real improvement here = RLVR helped spatial intelligence; a jump only
on trained commands but not these = overfitting.

    python scripts/eval_spatial.py --n 20 --max-iters 8 --out runs/eval_base
    python scripts/eval_spatial.py --n 20 --max-iters 8 --lora runs/rlvr1/lora_final --out runs/eval_rlvr
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

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


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


def build_prompt(instr, feedback=None):
    base = ("A square grid with numbered columns and rows and a single solid red "
            "triangle occupying one square block. " + instr +
            " Keep the grid, axes and labels identical; move ONLY the red triangle "
            "to its new grid block; output one solid red triangle snapped to the grid.")
    if feedback:
        base += " Correction from the previous attempt: " + feedback
    return base


def feedback_text(detected, target):
    if detected is None:
        return "Could not find one red triangle; draw a single solid red triangle on one block."
    dx, dy = target[0] - detected[0], target[1] - detected[1]
    p = []
    if dy < 0: p.append(f"UP {-dy}")
    elif dy > 0: p.append(f"DOWN {dy}")
    if dx < 0: p.append(f"LEFT {-dx}")
    elif dx > 0: p.append(f"RIGHT {dx}")
    return f"It is on block {detected}, target {target}. Move " + ", ".join(p) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--lora", type=str, default=None, help="LoRA adapter dir (post-RLVR)")
    ap.add_argument("--out", type=str, default="runs/eval_base")
    ap.add_argument("--seed", type=int, default=12345, help="FIXED so eval set is identical across runs")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    rng = np.random.default_rng(args.seed)        # fixed -> same held-out commands every run
    cmds = sample_commands(spec, args.n, rng)

    print(f"loading base-4B{' + LoRA' if args.lora else ''}...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        print("loaded LoRA:", args.lora)

    results = []
    for ci, (start, d, mv, target) in enumerate(cmds):
        instr = instruction_text(d, mv)
        current = render_state(spec, start).resize((512, 512))
        feedback = None
        first_try_success = False
        iters_to_converge = None
        best_dist = float("inf")
        for it in range(args.max_iters):
            gen = torch.Generator(device="cuda").manual_seed(1000 * ci + it)
            img = pipe(image=current, prompt=build_prompt(instr, feedback),
                       num_inference_steps=args.steps, guidance_scale=args.guidance,
                       height=512, width=512, generator=gen).images[0]
            sc = score_attempt(spec, to_array(img.resize((spec.img_w, spec.img_h))),
                               start_topleft=start, target_topleft=target,
                               direction=d, cfg=RewardConfig())
            best_dist = min(best_dist, sc.cell_distance)
            if it == 0:
                first_try_success = bool(sc.success)
            if args.save_images:
                img.save(out / f"cmd{ci:02d}_iter{it:02d}.png")
            if sc.success:
                iters_to_converge = it
                break
            feedback = feedback_text(sc.detected_topleft, target)
            current = img
        rec = dict(cmd=ci, start=list(start), direction=d, move=mv, target=list(target),
                   first_try_success=first_try_success,
                   converged=iters_to_converge is not None,
                   iters_to_converge=iters_to_converge,
                   best_cell_distance=round(best_dist, 2))
        results.append(rec)
        print(f"cmd {ci:02d} {d:>10} x{mv} {start}->{target}: "
              f"first_try={first_try_success} converged@{iters_to_converge} "
              f"best_dist={best_dist:.1f}", flush=True)

    n = len(results)
    agg = dict(
        n=n,
        first_try_success_rate=round(np.mean([r["first_try_success"] for r in results]), 3),
        converged_rate=round(np.mean([r["converged"] for r in results]), 3),
        mean_iters_to_converge=round(np.mean([r["iters_to_converge"] for r in results
                                              if r["iters_to_converge"] is not None] or [float("nan")]), 2),
        mean_best_cell_distance=round(np.mean([r["best_cell_distance"] for r in results]), 2),
        lora=args.lora,
    )
    (out / "eval.json").write_text(json.dumps(dict(agg=agg, results=results), indent=2))
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"-> {out}/eval.json")


if __name__ == "__main__":
    main()
