"""World-model benchmark: methods x moves, scored on speed + action-correctness +
VLM-judged quality. Research-grade comparison table.

Methods: cutpaste, hybrid (+ inpaint). Moves: right, left, down (up is off-board
from (1,1)). Accuracy = S detected on the expected cell (deterministic). Quality =
Qwen2.5-VL judge on scene_intact + quality; also VLM action_correct as a cross-check.

    python scripts/world_model_bench.py --out runs/wm_bench --judge
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.game.geometry import SceneGeom, apply_move
from spatial_intel.game import methods as M

SCENE = "game_ref/New Folder With Items/scene.png"


def detect_s_cell(img: np.ndarray, geom: SceneGeom):
    """Deterministic: the S badge is the LARGEST connected bright-red blob (it's a
    solid circle, ~1900px; presents/letters are smaller or multi-part). Return the
    cell of that blob's centroid."""
    from scipy import ndimage
    R, G, B = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lbl == biggest)
    return geom.pixel_to_cell(xs.mean(), ys.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/wm_bench")
    ap.add_argument("--moves", nargs="+", default=["right", "down", "left"])
    ap.add_argument("--s-cell", type=int, nargs=2, default=[1, 1])
    ap.add_argument("--judge", action="store_true", help="run VLM judge (loads Qwen)")
    ap.add_argument("--reps", type=int, default=3, help="speed timing reps")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    geom = SceneGeom()
    frame = Image.open(SCENE).convert("RGB")
    s_cell = tuple(args.s_cell)

    # load sd-turbo inpaint (for hybrid)
    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    from diffusers import AutoPipelineForInpainting
    print("loading sd-turbo inpaint (for hybrid)...")
    inpaint = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    inpaint.set_progress_bar_config(disable=True)

    method_fns = {
        "cutpaste": lambda f, sc, mv: M.cutpaste(f, sc, mv, geom),
        "hybrid":   lambda f, sc, mv: M.hybrid(f, sc, mv, geom, inpaint, steps=2),
    }

    judge = None
    if args.judge:
        from spatial_intel.game.vlm_judge import VLMJudge
        print("loading Qwen2.5-VL judge...")
        judge = VLMJudge()
        judge.load()

    results = []
    for mv in args.moves:
        expected = apply_move(s_cell, mv, geom)
        for mname, fn in method_fns.items():
            # timing (median of reps; warm up hybrid's model once)
            _ = fn(frame, s_cell, mv)  # warmup
            times = []
            for _ in range(args.reps):
                t0 = time.time(); nxt, info = fn(frame, s_cell, mv); times.append(time.time() - t0)
            med_t = float(np.median(times))
            nxt.save(out / f"{mname}_{mv}.png")

            det = detect_s_cell(np.array(nxt), geom)
            action_correct = (det == expected)
            rec = dict(method=mname, move=mv, expected_cell=list(expected),
                       detected_cell=list(det) if det else None,
                       action_correct=action_correct, time_s=round(med_t, 4))
            if judge is not None:
                js = judge.judge(frame, nxt, mv, expected)
                rec.update(vlm_action_correct=js.action_correct,
                           vlm_scene_intact=js.scene_intact, vlm_quality=js.quality)
            results.append(rec)
            extra = (f" | VLM intact={rec.get('vlm_scene_intact')} q={rec.get('vlm_quality')}"
                     if judge else "")
            print(f"[{mname:8} {mv:5}] det={det} exp={expected} correct={action_correct} "
                  f"{med_t*1000:.0f}ms{extra}", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    # summary table
    print("\n=== WORLD-MODEL BENCHMARK ===")
    print(f"{'method':10}{'accuracy':10}{'med_speed':12}{'VLM_intact':12}{'VLM_qual':10}")
    for mname in method_fns:
        sub = [r for r in results if r["method"] == mname]
        acc = np.mean([r["action_correct"] for r in sub])
        spd = np.median([r["time_s"] for r in sub]) * 1000
        si = np.mean([r.get("vlm_scene_intact", 0) for r in sub]) if judge else 0
        ql = np.mean([r.get("vlm_quality", 0) for r in sub]) if judge else 0
        print(f"{mname:10}{acc*100:6.0f}%   {spd:7.0f}ms   {si:9.1f}   {ql:7.1f}")
    print(f"-> {out}/results.json")


if __name__ == "__main__":
    main()
