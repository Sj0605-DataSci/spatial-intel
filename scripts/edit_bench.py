"""Multi-model FULL-DIFFUSION world-model benchmark: given (frame + action prompt),
each edit model produces the next frame; we score speed + action-correctness (S on
the right cell) + VLM quality.

    python scripts/edit_bench.py --models sd-turbo-i2i flux2-klein-4b --judge \
        --out runs/edit_bench

Compare against the deterministic cut-paste baseline (world_model_bench.py): the
question is whether ANY small edit model matches cut-paste's 100%/1ms, or how close.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from spatial_intel.game.geometry import SceneGeom, apply_move
from spatial_intel.game.editors import REGISTRY

SCENE = "game_ref/New Folder With Items/scene.png"

_DIRWORD = {"right": "right", "left": "left", "up": "up (toward the top)",
            "down": "down (toward the bottom)"}


def action_prompt(move, s_cell, geom):
    """COORDINATE-GROUNDED edit instruction: describe the grid coordinate system and
    the EXACT target cell + its pixel location, so the model knows precisely where to
    render S — instead of a vague relative 'move right'."""
    from spatial_intel.game.geometry import apply_move
    target = apply_move(s_cell, move, geom)
    tpx, tpy = geom.center(*target)
    spx, spy = geom.center(*s_cell)
    return (
        f"This is a top-down 8x8 grid game board (pixel-art christmas scene). The grid "
        f"cells are {int(geom.cell)} pixels wide; column 1 starts at the left, row 1 at "
        f"the top. The red circular 'S' badge is currently in column {s_cell[0]}, row "
        f"{s_cell[1]} (near pixel x={int(spx)}, y={int(spy)}). "
        f"Move ONLY the red 'S' badge {_DIRWORD[move]} by exactly one cell, so it is now "
        f"in column {target[0]}, row {target[1]} (centered at pixel x={int(tpx)}, "
        f"y={int(tpy)}). Remove it from its old cell (fill that cell with plain snow). "
        f"Keep every other object — houses, presents, trees, snow piles, HUD bar — "
        f"exactly unchanged in place."
    )


def detect_s_cell(img, geom):
    from scipy import ndimage
    a = np.asarray(img); R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0: return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
    return geom.pixel_to_cell(xs.mean(), ys.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(REGISTRY))
    ap.add_argument("--moves", nargs="+", default=["right", "down", "left"])
    ap.add_argument("--s-cell", type=int, nargs=2, default=[1, 1])
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default="runs/edit_bench")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    geom = SceneGeom(); frame = Image.open(SCENE).convert("RGB"); s_cell = tuple(args.s_cell)

    judge = None
    if args.judge:
        from spatial_intel.game.vlm_judge import VLMJudge
        judge = VLMJudge(); judge.load()

    results = []
    for mkey in args.models:
        ed = REGISTRY[mkey]()
        print(f"\n### {ed.name} — loading...")
        try:
            ed.load()
        except Exception as e:
            print(f"  FAILED to load: {type(e).__name__}: {str(e)[:120]}")
            results.append(dict(model=ed.name, load_error=str(e)[:200]))
            (out / "results.json").write_text(json.dumps(results, indent=2)); continue
        for mv in args.moves:
            expected = apply_move(s_cell, mv, geom)
            ap_txt = action_prompt(mv, s_cell, geom)
            r = ed.edit(frame, ap_txt, seed=1)  # warm
            r = ed.edit(frame, ap_txt, seed=2)  # timed
            r.image.save(out / f"{ed.name}_{mv}.png")
            det = detect_s_cell(r.image, geom)
            rec = dict(model=ed.name, move=mv, expected=list(expected),
                       detected=list(det) if det else None,
                       action_correct=(det == expected), seconds=round(r.seconds, 3))
            if judge:
                js = judge.judge(frame, r.image, mv, expected)
                rec.update(vlm_correct=js.action_correct, vlm_intact=js.scene_intact, vlm_quality=js.quality)
            results.append(rec)
            print(f"  [{mv:5}] det={det} exp={expected} {'✓' if rec['action_correct'] else '✗'} "
                  f"{r.seconds*1000:.0f}ms" + (f" q={rec.get('vlm_quality')}/5 intact={rec.get('vlm_intact')}/5" if judge else ""), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== EDIT-MODEL BENCHMARK ===")
    print(f"{'model':18}{'accuracy':10}{'med_ms':9}{'VLM_q':7}{'VLM_intact'}")
    for mkey in args.models:
        nm = REGISTRY[mkey]().name
        sub = [r for r in results if r.get("model") == nm and "move" in r]
        if not sub:
            print(f"{nm:18}{'(load failed)'}"); continue
        acc = np.mean([r["action_correct"] for r in sub]) * 100
        ms = np.median([r["seconds"] for r in sub]) * 1000
        q = np.mean([r.get("vlm_quality", 0) for r in sub]) if judge else 0
        si = np.mean([r.get("vlm_intact", 0) for r in sub]) if judge else 0
        print(f"{nm:18}{acc:5.0f}%   {ms:6.0f}   {q:5.1f}  {si:5.1f}")
    print(f"-> {out}/results.json")


if __name__ == "__main__":
    main()
