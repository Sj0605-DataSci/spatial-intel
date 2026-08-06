"""Per-iteration feedback loop on base-4B: keep editing until the triangle is on
the EXACT target block, saving the image at every iteration so you can watch it
converge.

This is the inference-time feedback reloop (NOT weight-updating RL): each step
FLUX edits the board, the env reads where the triangle landed, scores it, and
feeds a textual correction back into the next prompt. Runs up to --max-iters
(default 20) and STOPS early the moment success() is true.

    python scripts/feedback_loop.py --max-iters 20 --out runs/loop1

Every iteration writes runs/loop1/iter_NN.png plus a summary.json with the
detected block + reward per iteration.
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

from spatial_intel.grid import (GridSpec, render_state, to_array, from_array,
                                 apply_move, clamp_topleft, detect_object_topleft)
from spatial_intel.dataset import instruction_text
from spatial_intel.reward import RewardConfig, score_attempt

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


def build_prompt(instr: str, feedback: str | None) -> str:
    base = (
        "A square grid with numbered columns and rows and a single solid red "
        "triangle occupying one square block. " + instr +
        " Keep the grid, axes and labels identical; move ONLY the red triangle "
        "to its new grid block; output one solid red triangle snapped to the grid."
    )
    if feedback:
        base += " Correction from the previous attempt: " + feedback
    return base


def make_feedback(detected, target) -> str:
    if detected is None:
        return ("Could not find a single red triangle. Draw ONE solid red "
                "triangle aligned to exactly one grid block.")
    dx = target[0] - detected[0]
    dy = target[1] - detected[1]
    parts = []
    if dy < 0: parts.append(f"move it UP by {-dy} cell(s)")
    elif dy > 0: parts.append(f"move it DOWN by {dy} cell(s)")
    if dx < 0: parts.append(f"move it LEFT by {-dx} cell(s)")
    elif dx > 0: parts.append(f"move it RIGHT by {dx} cell(s)")
    where = f"It is on block {detected}; the target is block {target}."
    return where + (" Adjust: " + ", ".join(parts) + "." if parts else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--out", type=str, default="runs/loop1")
    ap.add_argument("--steps", type=int, default=28, help="denoise steps (base wants ~28)")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--direction", type=str, default="right")
    ap.add_argument("--move", type=int, default=1, help="cells to move")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    start = (spec.max_topleft // 2, spec.max_topleft // 2)
    target = clamp_topleft(spec, apply_move(spec, start, args.direction, args.move))
    instr = instruction_text(args.direction, args.move)
    print(f"start={start} target={target} | {instr}")

    # reference images
    Image.fromarray(to_array(render_state(spec, start))).save(out / "input.png")
    Image.fromarray(to_array(render_state(spec, target))).save(out / "target.png")

    print("loading base-4B...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # the board the model edits; starts as the input, then becomes its own last output
    current = render_state(spec, start).resize((512, 512))
    feedback = None
    history = []

    for it in range(args.max_iters):
        prompt = build_prompt(instr, feedback)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + it)
        out_img = pipe(image=current, prompt=prompt, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, height=512, width=512,
                       generator=gen).images[0]
        out_img.save(out / f"iter_{it:02d}.png")

        # score: read where the triangle landed (at env-native resolution)
        arr = to_array(out_img.resize((spec.img_w, spec.img_h)))
        sc = score_attempt(spec, arr, start_topleft=start, target_topleft=target,
                           direction=args.direction, cfg=RewardConfig())
        detected = sc.detected_topleft
        feedback = make_feedback(detected, target)
        rec = dict(iter=it, detected=detected, target=list(target),
                   success=bool(sc.success), reward=round(sc.reward, 3),
                   cell_distance=round(sc.cell_distance, 2))
        history.append(rec)
        print(f"iter {it:02d}: detected={detected} target={target} "
              f"success={sc.success} dist={sc.cell_distance:.1f}  -> iter_{it:02d}.png",
              flush=True)

        if sc.success:
            print(f"*** CONVERGED at iteration {it} — triangle on exact target block ***")
            break
        # feed the model its own last output to refine
        current = out_img

    (out / "summary.json").write_text(json.dumps(
        dict(start=list(start), target=list(target), instruction=instr,
             converged=history[-1]["success"], iters=len(history), history=history),
        indent=2))
    print(f"done -> {out} ({len(history)} iters, "
          f"{'CONVERGED' if history[-1]['success'] else 'did NOT converge'})")


if __name__ == "__main__":
    main()
