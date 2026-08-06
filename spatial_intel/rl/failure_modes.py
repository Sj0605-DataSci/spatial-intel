"""Failure-mode taxonomy scorer for a generated board image.

Inspired by GridVQA-X (Belsare et al., CVPR 2026 TRUE-V workshop, arXiv:2606.14740):
controlled grid synthesis with ground-truth cell annotations, scored with
IoU-against-ground-truth + (separately) cross-modal attention alignment.

Given a generated image and the commanded target cell, we classify the outcome
into one of these mutually-informative failure modes (a run can flag several;
`primary` picks the dominant one):

  SUCCESS          : a single clean red triangle on the exact target cell.
  POSITION_ERROR   : a clean red triangle, correct shape, but on the WRONG cell.
  SHAPE_DESTROYED  : red object present near target but not triangle-shaped
                     (blob/square/fragmented) — detected via mask shape metrics.
  OBJECT_MISIDENTIFY: no proper red object at all (wrong colour / no red / the
                     red mask is tiny or absent) — the model drew the wrong thing.
  MIXED            : wrong place AND wrong shape (both position + shape fail).

Method (all from a binary red mask):
  1. threshold red -> binary mask; connected components.
  2. identity  : is there a single dominant red blob of plausible size?
  3. shape     : triangularity = how well the blob matches a triangle template
                 (solidity vs convex hull + a filled-triangle IoU at the blob's
                 bounding box). High solidity + good triangle-IoU => clean triangle.
  4. position  : map blob centroid -> grid cell; compare to target.
  5. IoU       : red-mask IoU vs the ground-truth target render (the GridVQA-X
                 spatial-faithfulness primitive).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

from ..grid import GridSpec, render_state, to_array

# classification thresholds (tuned on the clean renders; see _self_test)
_MIN_RED_FRAC = 0.004      # min red area (frac of image) to count as "an object present"
_TRIANGLE_IOU_OK = 0.45    # blob-vs-triangle-template IoU above which shape is "clean"
_SOLIDITY_TRI = 0.62       # a filled triangle has solidity ~0.5-0.7; a square ~1.0


def _red_mask(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(int); g = img[..., 1].astype(int); b = img[..., 2].astype(int)
    return (r > 110) & (r - g > 35) & (r - b > 35)


def _largest_component(mask: np.ndarray):
    """Return (component_mask, (rows, cols) bbox, area) of the largest 4-conn blob.

    Pure-numpy flood fill (no scipy dependency)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    ys, xs = np.where(mask)
    idx = set(zip(ys.tolist(), xs.tolist()))
    for sy, sx in list(idx):
        if seen[sy, sx]:
            continue
        # BFS
        stack = [(sy, sx)]; comp = []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        if best is None or len(comp) > len(best):
            best = comp
    if best is None:
        return None, None, 0
    arr = np.array(best)
    cmask = np.zeros_like(mask, dtype=bool)
    cmask[arr[:, 0], arr[:, 1]] = True
    bbox = (arr[:, 0].min(), arr[:, 0].max(), arr[:, 1].min(), arr[:, 1].max())
    return cmask, bbox, len(best)


def _triangle_template(h: int, w: int) -> np.ndarray:
    """Upward solid triangle filling an h x w box (apex top-centre, base bottom)."""
    t = np.zeros((h, w), dtype=bool)
    for yy in range(h):
        frac = yy / max(h - 1, 1)            # 0 at top -> 1 at bottom
        half = frac * (w / 2)
        cx = w / 2
        lo = int(round(cx - half)); hi = int(round(cx + half))
        t[yy, max(0, lo):min(w, hi + 1)] = True
    return t


def _shape_metrics(comp_mask, bbox):
    """Return (triangle_iou, solidity) for the blob."""
    r0, r1, c0, c1 = bbox
    sub = comp_mask[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    if h < 3 or w < 3:
        return 0.0, 1.0
    tri = _triangle_template(h, w)
    inter = np.logical_and(sub, tri).sum()
    union = np.logical_or(sub, tri).sum()
    tri_iou = float(inter / union) if union else 0.0
    # solidity = filled area / bounding-box area (triangle ~0.5, square ~1.0)
    solidity = float(sub.sum() / (h * w))
    return tri_iou, solidity


@dataclass
class FailureScore:
    primary: str               # one of SUCCESS / POSITION_ERROR / SHAPE_DESTROYED / OBJECT_MISIDENTIFY / MIXED
    on_target: bool
    shape_ok: bool
    object_present: bool
    detected_cell: Optional[Tuple[int, int]]
    target_cell: Tuple[int, int]
    triangle_iou: float        # blob vs triangle template
    solidity: float
    mask_iou_vs_target: float  # red-mask IoU vs ground-truth render (GridVQA-X primitive)
    red_fraction: float

    def as_dict(self):
        d = asdict(self)
        d["detected_cell"] = list(self.detected_cell) if self.detected_cell else None
        d["target_cell"] = list(self.target_cell)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def classify(spec: GridSpec, img: np.ndarray, target_topleft: Tuple[int, int]) -> FailureScore:
    if img.shape[:2] != (spec.img_h, spec.img_w):
        from ..grid import from_array
        img = to_array(from_array(img).resize((spec.img_w, spec.img_h)))

    mask = _red_mask(img)
    red_frac = float(mask.mean())
    target = tuple(target_topleft)

    # ground-truth mask-IoU (GridVQA-X spatial faithfulness)
    tgt_mask = _red_mask(to_array(render_state(spec, target)))
    inter = np.logical_and(mask, tgt_mask).sum()
    union = np.logical_or(mask, tgt_mask).sum()
    mask_iou = float(inter / union) if union else 0.0

    object_present = red_frac >= _MIN_RED_FRAC
    if not object_present:
        return FailureScore("OBJECT_MISIDENTIFY", False, False, False, None, target,
                            0.0, 0.0, mask_iou, red_frac)

    comp, bbox, area = _largest_component(mask)
    tri_iou, solidity = _shape_metrics(comp, bbox)
    shape_ok = (tri_iou >= _TRIANGLE_IOU_OK) and (solidity <= 0.9)

    # detected cell from blob centroid
    ys, xs = np.where(comp)
    cy, cx = ys.mean(), xs.mean()
    cell_w = spec.img_w / spec.n
    det_col = int(cx // cell_w); det_row = int(cy // cell_w)
    # snap to object top-left convention (object is object_size cells)
    det = (max(0, min(spec.max_topleft, det_col - spec.object_size // 2)),
           max(0, min(spec.max_topleft, det_row - spec.object_size // 2)))
    on_target = det == target

    # also: is the red region big/scattered enough to be "not even a coherent object"?
    coherent = area / max(mask.sum(), 1) > 0.5   # largest blob is most of the red

    if not coherent or (tri_iou < 0.2 and solidity > 0.95):
        primary = "OBJECT_MISIDENTIFY"
    elif on_target and shape_ok:
        primary = "SUCCESS"
    elif not on_target and not shape_ok:
        primary = "MIXED"
    elif not shape_ok:
        primary = "SHAPE_DESTROYED"
    else:                                          # shape ok, wrong place
        primary = "POSITION_ERROR"

    return FailureScore(primary, on_target, shape_ok, object_present, det, target,
                        tri_iou, solidity, mask_iou, red_frac)
