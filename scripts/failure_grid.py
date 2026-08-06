"""K x P x n failure-characterisation grid (Aradhya's design).

For K initial positions, P target placements each, run the SAME fixed (terse)
prompt n times. Because diffusion is stochastic, the same query fails differently
across seeds — we classify every run into the failure taxonomy
(failure_modes.classify) and report the distribution.

  total generations = K * P * n   (default 3 * 5 * 10 = 150)

    python scripts/failure_grid.py --K 3 --P 5 --n 10 --out runs/fgrid1

Output:
  runs/fgrid1/results.json     : every run's classification
  runs/fgrid1/summary.json     : failure-mode distribution, per-init and overall
  runs/fgrid1/img/...          : (optional) saved images with --save-images
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import (GridSpec, render_state, to_array, apply_move,
                                 clamp_topleft, max_steps_for, DIRECTION_NAMES)
from spatial_intel.rl.failure_modes import classify
from spatial_intel.rl.attention import AttentionCapture, find_token_ids

FLUX = "black-forest-labs/FLUX.2-klein-base-4B"


def attention_target_score(heat, spec, target):
    """How much of the (normalised) text->image attention lands ON the target cell
    vs the global mean — a scalar 'is the model attending to the right place'."""
    import numpy as np
    side = heat.shape[0]
    cw = side / spec.n
    r0, r1 = int(target[1] * cw), int((target[1] + spec.object_size) * cw)
    c0, c1 = int(target[0] * cw), int((target[0] + spec.object_size) * cw)
    on = float(heat[r0:r1, c0:c1].mean()) if (r1 > r0 and c1 > c0) else 0.0
    return round(on, 4), round(float(heat.mean()), 4)


def target_coord_token_ids(pipe, prompt, target):
    """Token positions for 'triangle' + the TARGET coord digits (those after the
    'Move it to' clause), so the start coords aren't accidentally tracked."""
    tok = pipe.tokenizer
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    move_char = prompt.index("Move it to")
    tracked = []
    char = 0
    for i, tid in enumerate(ids):
        piece = tok.decode([tid])
        seg = piece.strip().lower()
        # character offset of this token in the prompt (approx via running decode)
        after_move = char >= move_char
        if "triangle" in seg:
            tracked.append(i)
        elif seg in (str(target[0]), str(target[1])) and after_move:
            tracked.append(i)
        char += len(piece)
    return tracked or find_token_ids(pipe, prompt, ["triangle"])


def coord_prompt(start, target):
    """Terse prompt with BOTH start and target coordinates (the spatial transform
    stated explicitly), per the corrected design."""
    return (f"Grid image. A red triangle is currently at cell ({start[0]},{start[1]}). "
            f"Move it to cell ({target[0]},{target[1]}). "
            "One red triangle, snapped to that cell.")


def init_positions(spec, K):
    """K spread-out initial positions: corners + center, sampled top-left->bottom-right."""
    m = spec.max_topleft
    pool = [(0, 0), (m, m), (m, 0), (0, m), (m // 2, m // 2)]
    return pool[:K]


def target_set(spec, start, P, rng):
    """P distinct reachable target cells from `start` (varied direction/distance)."""
    targets, seen = [], {tuple(start)}
    tries = 0
    while len(targets) < P and tries < 500:
        tries += 1
        d = DIRECTION_NAMES[int(rng.integers(len(DIRECTION_NAMES)))]
        hi = max_steps_for(spec, start, d)
        if hi < 1:
            continue
        mv = int(rng.integers(1, hi + 1))
        t = clamp_topleft(spec, apply_move(spec, start, d, mv))
        if tuple(t) not in seen:
            seen.add(tuple(t)); targets.append((tuple(t), d, mv))
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--P", type=int, default=5)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (out / "img").mkdir(exist_ok=True)
    spec = GridSpec()
    rng = np.random.default_rng(args.seed)

    inits = init_positions(spec, args.K)
    plan = []  # (start, target, direction, move)
    for start in inits:
        for (t, d, mv) in target_set(spec, start, args.P, rng):
            plan.append((start, t, d, mv))
    total = len(plan) * args.n
    print(f"K={args.K} inits={inits}")
    print(f"plan: {len(plan)} (start,target) pairs x n={args.n} = {total} generations")

    print("loading base-4B...")
    pipe = Flux2KleinPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    heat_dir = out / "attn"; heat_dir.mkdir(exist_ok=True)
    results = []
    gi = 0
    for (start, target, d, mv) in plan:
        src = render_state(spec, start).resize((512, 512))
        prompt = coord_prompt(start, target)
        # track 'triangle' + the TARGET coordinate tokens. Only count coord digits
        # that appear AFTER the move clause, so the start coords aren't tracked.
        tids = target_coord_token_ids(pipe, prompt, target)
        pe, _ = pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1)
        text_len = pe.shape[1]
        for run in range(args.n):
            gen = torch.Generator(device="cuda").manual_seed(args.seed + gi)
            cap = AttentionCapture(pipe.transformer, token_ids=tids)
            with cap.capture(text_len=text_len):
                img = pipe(image=src, prompt=prompt, num_inference_steps=args.steps,
                           guidance_scale=args.guidance, height=512, width=512,
                           generator=gen).images[0]
            fs = classify(spec, to_array(img.resize((spec.img_w, spec.img_h))), target)
            heat = cap.acc.heatmap(32, 32) if cap.acc else np.zeros((32, 32), np.float32)
            attn_on, attn_mean = attention_target_score(heat, spec, target)
            rec = dict(start=list(start), target=list(target), direction=d, move=mv,
                       run=run, attn_on_target=attn_on, attn_global_mean=attn_mean,
                       attn_focus_ratio=round(attn_on / attn_mean, 3) if attn_mean > 0 else 0.0,
                       **fs.as_dict())
            results.append(rec)
            # save heatmap as .npy (compact) for later visualisation
            np.save(heat_dir / f"s{start[0]}{start[1]}_t{target[0]}{target[1]}_r{run}.npy", heat)
            if args.save_images:
                img.save(out / "img" / f"s{start[0]}{start[1]}_t{target[0]}{target[1]}_r{run}.png")
            gi += 1
            if gi % 10 == 0 or gi == total:
                print(f"[{gi:3d}/{total}] start={start} target={target} run={run} "
                      f"-> {fs.primary}", flush=True)
        # incremental save
        (out / "results.json").write_text(json.dumps(results, indent=2))

    # -- summary: failure distribution overall and per init --
    def dist(recs):
        c = Counter(r["primary"] for r in recs)
        n = len(recs)
        return {k: {"count": c[k], "rate": round(c[k] / n, 3)} for k in
                ["SUCCESS", "POSITION_ERROR", "SHAPE_DESTROYED", "OBJECT_MISIDENTIFY", "MIXED"]}

    summary = dict(
        config=dict(K=args.K, P=args.P, n=args.n, total=total, prompt="terse(arm5)"),
        overall=dist(results),
        mean_mask_iou=round(float(np.mean([r["mask_iou_vs_target"] for r in results])), 3),
        mean_triangle_iou=round(float(np.mean([r["triangle_iou"] for r in results])), 3),
        mean_attn_focus_ratio=round(float(np.mean([r["attn_focus_ratio"] for r in results])), 3),
        attn_note=("focus_ratio = attention on target cell / global mean; "
                   ">1 means the model attends to the right cell, ~1 means it does not"),
        per_init={},
    )
    for start in inits:
        sub = [r for r in results if tuple(r["start"]) == tuple(start)]
        summary["per_init"][str(tuple(start))] = dict(n=len(sub), dist=dist(sub))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== FAILURE DISTRIBUTION (overall) ===")
    for k, v in summary["overall"].items():
        print(f"  {k:18} {v['count']:3d}  ({v['rate']*100:.0f}%)")
    print(f"  mean mask-IoU vs target: {summary['mean_mask_iou']}")
    print(f"-> {out}/summary.json")


if __name__ == "__main__":
    main()
