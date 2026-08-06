#!/usr/bin/env python
"""Generate the synthetic spatial-move dataset (input/output image pairs + manifest).

Examples
--------
    python scripts/generate_dataset.py --out data/v1 --n 128
    python scripts/generate_dataset.py --out data/center --n 64 --center-start
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_intel.grid import GridSpec
from spatial_intel.dataset import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--object-size", type=int, default=2)
    ap.add_argument("--cell-px", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center-start", action="store_true",
                    help="always start from the centre block (isolates direction generalisation)")
    args = ap.parse_args()

    spec = GridSpec(n=args.grid, object_size=args.object_size, cell_px=args.cell_px)
    samples = generate_dataset(args.out, n_samples=args.n, spec=spec,
                               seed=args.seed, center_start=args.center_start)
    print(f"Wrote {len(samples)} samples to {args.out}/")
    print(f"  {args.out}/input/*.png   {args.out}/output/*.png   {args.out}/manifest.json")
    for s in samples[:3]:
        print(f"  e.g. start={s.start_topleft} --{s.direction}x{s.steps}--> target={s.target_topleft}")


if __name__ == "__main__":
    main()
