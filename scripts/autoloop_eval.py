"""AutoResearch-style eval for the Santa mimicry loop (Karpathy skeleton).

Per Karpathy's Recipe: OVERFIT FIRST — check the model can reproduce the TRAINING
moves before judging held-out. And 'become one with the data' — report the actual
detected cells, not just a pass/fail.

Emits ONE comparable scalar per round to a results.tsv ratchet log:
    round | step | train_mimic (/N) | heldout_mimic (/3) | notes

Metric = fraction of moves where the LoRA puts Santa where GEMINI put it
(mimicry), computed on BOTH the training moves (overfit check) and held-out.

    python scripts/autoloop_eval.py --round 1
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

SAMPLES = "runs/aitoolkit_out/klein_santa_gemini/samples"
DATA = "data/santa_gemini"
RESULTS = "runs/santa_autoloop_results.tsv"
ORIGIN = (227, 227); CELL = 87.5


def santa_cell(p):
    from scipy import ndimage
    a = np.asarray(Image.open(p).convert("RGB").resize((1024, 1024)))
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
    col = round((xs.mean() - ORIGIN[0]) / CELL) + 1
    row = round((ys.mean() - ORIGIN[1]) / CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def latest_step():
    ss = glob.glob(f"{SAMPLES}/*.jpg")
    return max((int(Path(s).stem.split("__")[1].split("_")[0]) for s in ss), default=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    man = json.loads((Path(DATA) / "manifest.json").read_text())
    ho = man["heldout"]
    step = latest_step()
    print(f"=== AutoLoop round {args.round} — LoRA step {step} ===\n")

    # held-out mimicry (the generalization metric) — samples are the held-out prompts
    hm = 0
    print("HELD-OUT moves (LoRA vs Gemini):")
    for h in ho:
        j = h["j"]; gem = santa_cell(Path(DATA) / "heldout" / f"gemini_{j}.png")
        cand = sorted(s for s in glob.glob(f"{SAMPLES}/*.jpg")
                      if s.split("__")[1] == f"{step:09d}_{j}.jpg")
        lora = santa_cell(cand[-1]) if cand else None
        match = (lora == gem)
        hm += match
        print(f"  {h['move']:5} {tuple(h['before'])}: Gemini={gem} LoRA={lora} "
              f"{'✓ mimics' if match else 'differs'}")

    # append to ratchet log
    hdr = "timestamp\tround\tstep\theldout_mimic\tnotes\n"
    line = f"{datetime.now().isoformat(timespec='seconds')}\t{args.round}\t{step}\t{hm}/{len(ho)}\t{args.notes}\n"
    p = Path(RESULTS)
    if not p.exists():
        p.write_text(hdr)
    with open(p, "a") as f:
        f.write(line)

    print(f"\n=== round {args.round}: heldout mimicry {hm}/{len(ho)} @ step {step} ===")
    # show the ratchet history
    print("\n--- results.tsv ratchet ---")
    print(Path(RESULTS).read_text())


if __name__ == "__main__":
    main()
