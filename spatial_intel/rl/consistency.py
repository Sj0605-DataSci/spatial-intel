"""Image-consistency reward: how close is the model's output to the ground-truth
target render — not just 'is the red blob in the right block'.

The placement reward (reward.py) is blind to shape and grid quality: a fat red
square on the right block scores the same as a crisp triangle. To push toward
'draw the EXACT image', we add terms that compare the output against the target
PNG we can render for free:

  - grid_consistency : the non-red structure (grid lines, axes, labels) should be
    preserved. Scored as 1 - mean abs diff on the grayscale background where the
    target has NO red (so the triangle's own pixels don't penalise).
  - shape_match      : IoU of the red region between output and target. Rewards
    the triangle being the right size/shape AND in the right place at once.

Both are in [0, 1]. Kept separate from RewardConfig so the placement reward stays
a clean, self-contained verifiable signal; the trainer combines them with weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import GridSpec, render_state, to_array


def _red_mask(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    b = img[..., 2].astype(int)
    return (r > 110) & (r - g > 35) & (r - b > 35)


@dataclass
class ConsistencyScore:
    grid_consistency: float    # background/grid preservation [0,1]
    shape_match: float         # red-region IoU vs target [0,1]
    combined: float            # weighted blend [0,1]

    def as_dict(self):
        return {"grid_consistency": round(self.grid_consistency, 4),
                "shape_match": round(self.shape_match, 4),
                "combined": round(self.combined, 4)}


def consistency_reward(spec: GridSpec, output_img: np.ndarray, target_topleft,
                       *, grid_weight: float = 0.4, shape_weight: float = 0.6
                       ) -> ConsistencyScore:
    """Compare output against the ground-truth target render.

    `output_img` must be at the env-native resolution (spec.img_h, spec.img_w);
    resize before calling. The target is rendered on the fly from target_topleft.
    """
    target_img = to_array(render_state(spec, target_topleft))
    if output_img.shape != target_img.shape:
        raise ValueError(f"output {output_img.shape} != target {target_img.shape}; "
                         "resize output to env resolution first")

    out_red = _red_mask(output_img)
    tgt_red = _red_mask(target_img)

    # shape_match: IoU of the red regions (size + shape + position in one)
    inter = np.logical_and(out_red, tgt_red).sum()
    union = np.logical_or(out_red, tgt_red).sum()
    shape_match = float(inter / union) if union > 0 else 0.0

    # grid_consistency: on the background (where target has no red), how close is
    # the grayscale? penalises mangled grid lines / axes / labels.
    bg = ~tgt_red
    if bg.sum() == 0:
        grid_consistency = 0.0
    else:
        out_gray = output_img.astype(np.float32).mean(-1)
        tgt_gray = target_img.astype(np.float32).mean(-1)
        diff = np.abs(out_gray - tgt_gray)[bg].mean() / 255.0
        grid_consistency = float(max(0.0, 1.0 - diff))

    combined = grid_weight * grid_consistency + shape_weight * shape_match
    return ConsistencyScore(grid_consistency, shape_match, combined)
