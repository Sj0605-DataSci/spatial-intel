"""Compare the Santa LoRA's held-out samples against Gemini's actual outputs.
Answers: does the LoRA mimic Gemini's move?

For each held-out move, we have:
  - before frame (input)
  - Gemini's after (the target behavior)         data/santa_gemini/heldout/gemini_j.png
  - the LoRA's latest sample for that prompt      runs/.../samples/*_<idx>.jpg

We detect Santa's cell (largest bright-red 'S' badge blob) in each and report:
  gemini_cell vs lora_cell vs expected_cell   -> did the LoRA move S like Gemini?

    python scripts/compare_santa_gemini.py
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

SAMPLES = "runs/aitoolkit_out/klein_santa_gemini/samples"
HELDOUT = "data/santa_gemini"
GEOM_ORIGIN = (227, 227)   # from SceneGeom
GEOM_CELL = 87.5


def santa_cell(img_path_or_arr):
    """Largest bright-red blob -> grid cell (1-indexed), using the scene geometry.
    The 'S' badge is a solid bright-red circle; presents/letters are smaller/multi."""
    from scipy import ndimage
    a = np.asarray(Image.open(img_path_or_arr).convert("RGB").resize((1024, 1024))) \
        if isinstance(img_path_or_arr, (str, Path)) else img_path_or_arr
    R, G, B = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    red = (R > 195) & (R - G > 110) & (R - B > 110)
    lbl, n = ndimage.label(red)
    if n == 0:
        return None
    sizes = ndimage.sum(red, lbl, range(1, n + 1))
    ys, xs = np.where(lbl == int(np.argmax(sizes)) + 1)
    cx, cy = xs.mean(), ys.mean()
    col = round((cx - GEOM_ORIGIN[0]) / GEOM_CELL) + 1
    row = round((cy - GEOM_ORIGIN[1]) / GEOM_CELL) + 1
    return (max(1, min(8, col)), max(1, min(8, row)))


def main():
    ho = json.loads((Path(HELDOUT) / "manifest.json").read_text())["heldout"]
    # newest sample set (latest step)
    all_s = sorted(glob.glob(f"{SAMPLES}/*.jpg"))
    if not all_s:
        print("no samples yet"); return
    # samples named ..._<promptidx>.jpg ; group by step (the numeric middle)
    latest_step = max(int(Path(s).stem.split("__")[1].split("_")[0]) for s in all_s)
    print(f"=== comparing LoRA (step {latest_step}) vs Gemini on {len(ho)} held-out moves ===\n")
    mimick = 0
    for h in ho:
        j = h["j"]; move = h["move"]; b = tuple(h["before"]); exp = tuple(h["after"])
        gem = Path(HELDOUT) / "heldout" / f"gemini_{j}.png"
        gem_cell = santa_cell(gem)
        # find the LoRA sample for prompt j at the latest step
        cand = [s for s in all_s if s.split("__")[1] == f"{latest_step:09d}_{j}.jpg"]
        if not cand:
            cand = [s for s in all_s if s.endswith(f"_{j}.jpg") and f"{latest_step:09d}" in s]
        lora_cell = santa_cell(cand[-1]) if cand else None
        # did the LoRA put S where Gemini did?
        match_gem = (lora_cell == gem_cell)
        match_exp = (lora_cell == exp)
        if match_gem: mimick += 1
        print(f"move '{move}' from {b}: expected {exp}")
        print(f"   Gemini put S at:  {gem_cell}")
        print(f"   LoRA put S at:    {lora_cell}   -> {'MATCHES Gemini ✓' if match_gem else 'differs'}"
              f"{' (and correct)' if match_exp else ''}")
        print()
    print(f"=== LoRA mimics Gemini on {mimick}/{len(ho)} held-out moves (step {latest_step}) ===")


if __name__ == "__main__":
    main()
