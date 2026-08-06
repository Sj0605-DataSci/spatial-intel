"""Build the supervised LoRA dataset with a strict coordinate train/test split.

Task: given an INPUT board (red triangle at a start cell) + a prompt, produce the
OUTPUT board (red triangle at the target cell). We have exact ground-truth renders
for both, so this is supervised img2img.

Split rule (per Aradhya): a set of cells is held out for TEST. A (start, target)
pair is TRAIN only if BOTH its start and target are in the train-cell set; it is a
candidate for TEST only if BOTH are in the held-out set. This guarantees no eval
coordinate ever appears — as input OR output — anywhere in training.

Two prompt styles per example (the two LoRA conditions):
  - natural_language : "Move the red triangle up-right by 2 cells."
  - coords           : "Red triangle at (sx,sy). Move it to (tx,ty)."

Emits:
  data/lora/manifest.json         : train + test example lists with both prompts
  data/lora/train/inp_*.png, tgt_*.png
  data/lora/test/...
  data/lora/split.json            : which cells are train vs held-out
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from spatial_intel.grid import (GridSpec, render_state, apply_move, clamp_topleft,
                                 max_steps_for, DIRECTION_NAMES, normalize_direction)
from spatial_intel.dataset import instruction_text


def all_cells(spec):
    m = spec.max_topleft
    return [(c, r) for c in range(m + 1) for r in range(m + 1)]


def nl_prompt(start, direction, steps):
    return "Move the red triangle " + instruction_text(direction, steps).split("Move the red triangle ")[-1] \
        if False else instruction_text(direction, steps)


def coord_prompt(start, target):
    return (f"Grid image. A red triangle is currently at cell ({start[0]},{start[1]}). "
            f"Move it to cell ({target[0]},{target[1]}). "
            "One red triangle, snapped to that cell.")


def enumerate_pairs(spec, cell_set):
    """All (start, dir, steps, target) where BOTH start and target are in cell_set."""
    cset = set(cell_set)
    pairs = []
    for start in cell_set:
        for d in DIRECTION_NAMES:
            hi = max_steps_for(spec, start, d)
            for steps in range(1, hi + 1):
                target = clamp_topleft(spec, apply_move(spec, start, d, steps))
                if target != start and tuple(target) in cset:
                    pairs.append((start, d, steps, tuple(target)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lora")
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--train-size", type=int, default=300, help="cap training examples")
    ap.add_argument("--test-size", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    rng = np.random.default_rng(args.seed)

    cells = all_cells(spec)
    n_hold = int(round(len(cells) * args.holdout_frac))
    hold_idx = rng.choice(len(cells), size=n_hold, replace=False)
    held = {cells[i] for i in hold_idx}
    train_cells = [c for c in cells if c not in held]
    test_cells = [c for c in cells if c in held]
    print(f"total cells {len(cells)} | train-cells {len(train_cells)} | held-out {len(test_cells)}")

    train_pairs = enumerate_pairs(spec, train_cells)
    test_pairs = enumerate_pairs(spec, test_cells)
    print(f"candidate pairs: train {len(train_pairs)}  test {len(test_pairs)}")

    # sanity: NO test cell appears in any train pair (as start or target)
    train_cells_used = set()
    for s, d, st, t in train_pairs:
        train_cells_used.add(tuple(s)); train_cells_used.add(tuple(t))
    leak = train_cells_used & held
    assert not leak, f"LEAK: held-out cells in train pairs: {leak}"
    print("split verified: zero held-out cells appear in training (as start or target)")

    def sample(pairs, k):
        idx = rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)
        return [pairs[i] for i in idx]

    train = sample(train_pairs, args.train_size)
    test = sample(test_pairs, args.test_size)

    def emit(pairs, split):
        recs = []
        for i, (s, d, st, t) in enumerate(pairs):
            inp = render_state(spec, tuple(s)).resize((args.img_size, args.img_size))
            tgt = render_state(spec, tuple(t)).resize((args.img_size, args.img_size))
            ip = out / split / f"inp_{i:04d}.png"
            tp = out / split / f"tgt_{i:04d}.png"
            inp.save(ip); tgt.save(tp)
            recs.append(dict(id=i, start=list(s), direction=d, steps=st, target=list(t),
                             input=str(ip.relative_to(out)), target_img=str(tp.relative_to(out)),
                             prompt_nl=nl_prompt(s, d, st),
                             prompt_coords=coord_prompt(s, t)))
        return recs

    manifest = dict(
        spec=dict(n=spec.n, object_size=spec.object_size, img_size=args.img_size),
        holdout_frac=args.holdout_frac,
        train=emit(train, "train"),
        test=emit(test, "test"),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "split.json").write_text(json.dumps(
        dict(train_cells=[list(c) for c in train_cells],
             held_out_cells=[list(c) for c in test_cells]), indent=2))
    print(f"wrote {len(manifest['train'])} train + {len(manifest['test'])} test examples -> {out}")
    print("example NL prompt:  ", manifest["train"][0]["prompt_nl"])
    print("example coord prompt:", manifest["train"][0]["prompt_coords"])


if __name__ == "__main__":
    main()
