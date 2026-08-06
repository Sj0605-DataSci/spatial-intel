"""Synthetic dataset generation for the spatial-move task.

Each sample is a triple:
    input image   -> board with the triangle at the START block + a coord overlay
    instruction   -> natural-language command ("move the triangle to the top-right by 2")
    output image  -> board with the triangle at the TARGET block (ground truth)

plus a metadata dict capturing start/target/direction/steps so the env and the
reward function can score arbitrary attempts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .grid import (
    GridSpec, DIRECTION_NAMES, DIRECTION_ID, apply_move, clamp_topleft,
    max_steps_for, render_state,
)

PHRASE = {
    "right": "to the right", "left": "to the left",
    "up": "up", "down": "down",
    "up_right": "to the top-right", "up_left": "to the top-left",
    "down_right": "to the bottom-right", "down_left": "to the bottom-left",
}


def instruction_text(direction: str, steps: int) -> str:
    cells = "cell" if steps == 1 else "cells"
    return (f"Move the red triangle {PHRASE[direction]} by {steps} {cells}, "
            f"keeping it aligned to the grid.")


@dataclass
class Sample:
    idx: int
    grid_n: int
    object_size: int
    start_topleft: Tuple[int, int]
    direction: str
    steps: int
    target_topleft: Tuple[int, int]
    instruction: str

    def meta(self) -> dict:
        d = asdict(self)
        d["direction_id"] = DIRECTION_ID[self.direction]
        return d


def make_sample(spec: GridSpec, rng: np.random.Generator, idx: int,
                *, fixed_start: Optional[Tuple[int, int]] = None,
                fixed_direction: Optional[str] = None,
                fixed_steps: Optional[int] = None) -> Sample:
    """Sample a solvable (start, direction, steps) so the target stays on board."""
    for _ in range(200):
        start = fixed_start if fixed_start is not None else (
            int(rng.integers(0, spec.max_topleft + 1)),
            int(rng.integers(0, spec.max_topleft + 1)),
        )
        direction = fixed_direction or DIRECTION_NAMES[int(rng.integers(len(DIRECTION_NAMES)))]
        max_s = max_steps_for(spec, start, direction)
        if max_s < 1:
            if fixed_start or fixed_direction:
                # caller forced an infeasible combo; clamp to 0 and bail
                break
            continue
        steps = fixed_steps if fixed_steps is not None else int(rng.integers(1, max_s + 1))
        steps = min(steps, max_s)
        target = clamp_topleft(spec, apply_move(spec, start, direction, steps))
        return Sample(idx, spec.n, spec.object_size, tuple(start), direction,
                      int(steps), tuple(target), instruction_text(direction, steps))
    # fallback: a trivially valid no-op-ish sample
    start = (spec.max_topleft // 2, spec.max_topleft // 2)
    direction = "right"
    steps = max(1, max_steps_for(spec, start, direction))
    target = clamp_topleft(spec, apply_move(spec, start, direction, steps))
    return Sample(idx, spec.n, spec.object_size, start, direction, steps,
                  target, instruction_text(direction, steps))


def generate_dataset(out_dir: str | Path, n_samples: int = 64,
                     spec: Optional[GridSpec] = None, seed: int = 0,
                     center_start: bool = False) -> List[Sample]:
    """Write input/output PNG pairs + a manifest to `out_dir`.

    center_start: if True, always start from the centre block (the (3,3) example
                  from the brief) so direction generalisation is isolated.
    """
    spec = spec or GridSpec()
    out = Path(out_dir)
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "output").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    center = (spec.max_topleft // 2, spec.max_topleft // 2)
    samples: List[Sample] = []
    for i in range(n_samples):
        s = make_sample(spec, rng, i,
                        fixed_start=center if center_start else None)
        in_img = render_state(spec, s.start_topleft,
                              caption=f"START {s.start_topleft}  |  {s.instruction}")
        out_img = render_state(spec, s.target_topleft,
                               caption=f"TARGET {s.target_topleft}")
        in_img.save(out / "input" / f"{i:05d}.png")
        out_img.save(out / "output" / f"{i:05d}.png")
        samples.append(s)

    manifest = {
        "grid": {"n": spec.n, "object_size": spec.object_size,
                 "cell_px": spec.cell_px, "img_w": spec.img_w, "img_h": spec.img_h},
        "n_samples": n_samples,
        "seed": seed,
        "center_start": center_start,
        "samples": [s.meta() for s in samples],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return samples
