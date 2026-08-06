#!/usr/bin/env python
"""Localization probe for the FROZEN FLUX.2-klein-4B.

No training, no feedback reloop. For each dataset sample we give FLUX the input
board (triangle at the start block) + the instruction, ask it to render the
triangle at the commanded block ONE shot, then use the verifier to ask:

    did it put the triangle on the right grid block?

Reports, over N samples:
  - detect rate      : produced a single findable red triangle at all
  - success rate     : landed on the exact target block (the headline)
  - mean cell dist   : how far off (in cells) when detected
  - dir-correct rate : moved in (roughly) the commanded direction

Also dumps a side-by-side  [input | FLUX output | ground-truth target]  per
sample so you can eyeball whether it even preserved the grid.

Usage
-----
    python scripts/flux_probe.py --data data/center --n 8 --out runs/probe1 \
        --steps 4 --cpu-offload
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from spatial_intel.grid import GridSpec, to_array
from spatial_intel.reward import score_attempt, RewardConfig
from spatial_intel.agents.flux import FluxAgent


def side_by_side(paths_or_imgs, labels):
    imgs = [p if isinstance(p, Image.Image) else Image.open(p) for p in paths_or_imgs]
    imgs = [im.convert("RGB") for im in imgs]
    h = max(im.height for im in imgs)
    pad = 8
    w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h + 22), (255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(imgs, labels):
        canvas.paste(im, (x, 22))
        d.text((x + 4, 4), lab, fill=(0, 0, 0))
        x += im.width + pad
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/center")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/probe1")
    ap.add_argument("--steps", type=int, default=4, help="num_inference_steps")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--work-size", type=int, default=512)
    ap.add_argument("--cpu-offload", action="store_true",
                    help="enable_model_cpu_offload (use if VRAM is tight)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    g = manifest["grid"]
    spec = GridSpec(n=g["n"], object_size=g["object_size"], cell_px=g["cell_px"])
    samples = manifest["samples"][: args.n]

    out = Path(args.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)

    print(f"Loading FLUX.2-klein-4B (first run downloads weights)...", flush=True)
    agent = FluxAgent(spec, num_inference_steps=args.steps,
                      guidance_scale=args.guidance, work_size=args.work_size,
                      cpu_offload=args.cpu_offload, seed=args.seed)

    cfg = RewardConfig()
    rows, detects, successes, dists, dircorr = [], 0, 0, [], 0
    for i, s in enumerate(samples):
        in_path = data / "input" / f"{s['idx']:05d}.png"
        tgt_path = data / "output" / f"{s['idx']:05d}.png"
        obs = {"image": to_array(Image.open(in_path))}
        info = {"instruction": s["instruction"], "t": 0}

        gen = agent.act(obs, info)  # one-shot, no feedback
        sc = score_attempt(spec, gen,
                           start_topleft=tuple(s["start_topleft"]),
                           target_topleft=tuple(s["target_topleft"]),
                           direction=s["direction"], cfg=cfg)

        gen_img = Image.fromarray(gen)
        panel = side_by_side(
            [in_path, gen_img, tgt_path],
            [f"INPUT start={tuple(s['start_topleft'])}",
             f"FLUX -> det={sc.detected_topleft} {'OK' if sc.success else ''}",
             f"TARGET {tuple(s['target_topleft'])}"])
        panel.save(out / "panels" / f"{s['idx']:05d}.png")

        if sc.detected_topleft is not None:
            detects += 1
            dists.append(sc.cell_distance)
        if sc.success:
            successes += 1
        if sc.direction_correct:
            dircorr += 1
        rows.append({"idx": s["idx"], "instruction": s["instruction"],
                     **sc.as_dict()})
        print(f"[{i+1}/{len(samples)}] {s['direction']}x{s['steps']:<2} "
              f"target={tuple(s['target_topleft'])} det={sc.detected_topleft} "
              f"success={sc.success} dist={sc.cell_distance:.2f} conf={sc.confidence:.4f}",
              flush=True)

    n = len(samples)
    summary = {
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "data": str(data), "n": n, "steps": args.steps, "guidance": args.guidance,
        "detect_rate": detects / n,
        "success_rate": successes / n,
        "mean_cell_distance_when_detected": (statistics.mean(dists) if dists else None),
        "direction_correct_rate": dircorr / n,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n==== FLUX.2-klein-4B localization probe ====")
    print(f"  samples            : {n}")
    print(f"  detect rate        : {detects}/{n} = {detects/n:.1%}")
    print(f"  exact-block success: {successes}/{n} = {successes/n:.1%}")
    if dists:
        print(f"  mean cell distance : {statistics.mean(dists):.2f} (when detected)")
    print(f"  direction-correct  : {dircorr}/{n} = {dircorr/n:.1%}")
    print(f"  panels + summary   : {out}/")


if __name__ == "__main__":
    main()
