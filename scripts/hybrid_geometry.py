"""Per-game cell<->pixel geometry, calibrated from real data (no hardcoding).

For each game: use transitions where the player moved, and the states.jsonl px,py
ground truth, to least-squares fit  pixel_center = origin + (cell-1)*cell_size
on each axis. Handles y-flip automatically (fit finds the sign). The ARRIVAL cell
pixel is isolated by taking, within the changed region, the centroid of pixels
that match the NEXT frame (the sprite that appeared), not the vacated one.

Caches to game_datasets_bundle/<game>/geometry.json.
"""
import json
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"


def calibrate(game, max_samples=120):
    g = BUNDLE / game
    trans = [json.loads(l) for l in open(g / "transitions.jsonl")]
    pts = []
    for t in trans:
        e = t.get("effects", {})
        mp = e.get("moved_player")
        if not mp:
            continue
        a = np.asarray(Image.open(g / t["obs_t"]).convert("RGB")).astype(int)
        b = np.asarray(Image.open(g / t["obs_next"]).convert("RGB")).astype(int)
        d = np.abs(a - b).sum(-1)
        ys, xs = np.where(d > 40)
        if len(xs) < 20:
            continue
        # isolate ARRIVAL: among changed pixels, those where b differs a lot from a
        # AND b is "sprite-like" (not background) — approximate by the half of the
        # changed pixels closest to the 'to' cell via k-means on 2 clusters, pick
        # the cluster whose mean, mapped later, is most consistent. Simpler & robust:
        # take the changed pixels and split by which frame's local variance is higher.
        to = mp["to"]
        # cluster changed coords into 2 (vacated vs arrived); keep both as candidates,
        # we disambiguate globally after the fit by residual.
        pts.append((to[0], to[1], xs.mean(), ys.mean(), xs, ys))
        if len(pts) >= max_samples:
            break

    # First fit using centroid (coarse), then refine by choosing, per sample, the
    # sub-blob (via 2-means) that best matches the coarse model.
    P = np.array([(p[0], p[1], p[2], p[3]) for p in pts], float)

    def fit(P):
        Ax = np.c_[P[:, 0] - 1, np.ones(len(P))]
        cw, ox = np.linalg.lstsq(Ax, P[:, 2], rcond=None)[0]
        Ay = np.c_[P[:, 1] - 1, np.ones(len(P))]
        ch, oy = np.linalg.lstsq(Ay, P[:, 3], rcond=None)[0]
        return ox, oy, cw, ch

    ox, oy, cw, ch = fit(P)
    # refine: for each sample, 2-means the changed pixels, pick the sub-centroid
    # closest to the model-predicted (to) pixel.
    refined = []
    for (px, py, _, _, xs, ys) in pts:
        pred = (ox + (px - 1) * cw, oy + (py - 1) * ch)
        coords = np.c_[xs, ys].astype(float)
        if len(coords) >= 4:
            # simple 2-means
            c0, c1 = coords[coords[:, 0].argmin()], coords[coords[:, 0].argmax()]
            for _ in range(6):
                d0 = ((coords - c0) ** 2).sum(1); d1 = ((coords - c1) ** 2).sum(1)
                m0 = d0 < d1
                if m0.any(): c0 = coords[m0].mean(0)
                if (~m0).any(): c1 = coords[~m0].mean(0)
            cand = c0 if ((c0 - pred) ** 2).sum() < ((c1 - pred) ** 2).sum() else c1
        else:
            cand = coords.mean(0)
        refined.append((px, py, cand[0], cand[1]))
    P = np.array(refined, float)
    ox, oy, cw, ch = fit(P)

    # residual
    predx = ox + (P[:, 0] - 1) * cw; predy = oy + (P[:, 1] - 1) * ch
    res = float(np.sqrt(((predx - P[:, 2]) ** 2 + (predy - P[:, 3]) ** 2).mean()))
    geom = dict(origin=[round(ox, 1), round(oy, 1)],
                cell=[round(cw, 1), round(ch, 1)],
                grid=json.load(open(g / "meta.json"))["grid"],
                residual_px=round(res, 1), n=len(P))
    (g / "geometry.json").write_text(json.dumps(geom, indent=2))
    return geom


def cell_center(geom, px, py):
    ox, oy = geom["origin"]; cw, ch = geom["cell"]
    return (ox + (px - 1) * cw, oy + (py - 1) * ch)


def pixel_to_cell(geom, x, y):
    ox, oy = geom["origin"]; cw, ch = geom["cell"]
    gx, gy = geom["grid"]
    px = round((x - ox) / cw) + 1
    py = round((y - oy) / ch) + 1
    return (max(1, min(gx, px)), max(1, min(gy, py)))


if __name__ == "__main__":
    games = sys.argv[1:] or [p.name for p in BUNDLE.iterdir()
                             if p.is_dir() and (p / "transitions.jsonl").exists()]
    for g in games:
        geom = calibrate(g)
        print(f"{g:22} origin={geom['origin']} cell={geom['cell']} "
              f"grid={geom['grid']} residual={geom['residual_px']}px (n={geom['n']})")
