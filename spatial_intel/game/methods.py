"""Next-frame generation METHODS for the world-model benchmark.

Each method takes (frame, s_cell, move, geom) and returns (next_frame_image, info).
The benchmark times each and scores accuracy (S on correct cell) + VLM quality.

Methods:
  cutpaste : detect S sprite, move it, fill hole from a clean snow cell. Exact
             placement, rest pixel-identical, but fill may seam. ~instant.
  inpaint  : sd-turbo full-image inpaint of old+new S cells. Clean fill but
             re-renders (corrupts) the whole frame; may fail to draw S.
  hybrid   : cutpaste the sprite (exact, rest untouched) THEN inpaint ONLY the
             vacated hole, composited so only that hole changes. Best of both.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

from .geometry import SceneGeom, apply_move

RED = lambda a: ((a[..., 0].astype(int) > 180) & (a[..., 0].astype(int) - a[..., 1].astype(int) > 90)
                 & (a[..., 0].astype(int) - a[..., 2].astype(int) > 90))


def _find_s(img: np.ndarray, geom: SceneGeom, s_cell, pad=6):
    l, t, r, b = [int(v) for v in geom.cell_bbox(*s_cell)]
    crop = img[t:b, l:r]
    red = RED(crop)
    ys, xs = np.where(red)
    if len(xs) == 0:
        return None
    return (l + xs.min() - pad, t + ys.min() - pad, l + xs.max() + pad, t + ys.max() + pad)


def _clean_fill(img, bbox, geom, fill_cell=(7, 1)):
    """Return a background patch (same size as bbox) sampled from a clean snow cell."""
    bl, bt, br, bb = bbox
    ph, pw = bb - bt, br - bl
    fpx, fpy = geom.center(*fill_cell)
    fl, ft = int(fpx - pw / 2), int(fpy - ph / 2)
    return img[ft:ft + ph, fl:fl + pw].copy()


def cutpaste(frame: Image.Image, s_cell, move, geom: SceneGeom):
    t0 = time.time()
    img = np.array(frame.convert("RGB"))
    new_cell = apply_move(s_cell, move, geom)
    bbox = _find_s(img, geom, s_cell)
    out = img.copy()
    if bbox is not None:
        bl, bt, br, bb = [int(v) for v in bbox]
        sprite = img[bt:bb, bl:br].copy()
        out[bt:bb, bl:br] = _clean_fill(img, (bl, bt, br, bb), geom)
        dx = int((new_cell[0] - s_cell[0]) * geom.cell)
        dy = int((new_cell[1] - s_cell[1]) * geom.cell)
        R, G, B = sprite[..., 0].astype(int), sprite[..., 1].astype(int), sprite[..., 2].astype(int)
        m = ((R > 150) & (R - G > 60) & (R - B > 60)) | ((R > 200) & (G > 200) & (B > 200))
        reg = out[bt + dy:bb + dy, bl + dx:br + dx]
        reg[m] = sprite[m]
        out[bt + dy:bb + dy, bl + dx:br + dx] = reg
    return Image.fromarray(out), dict(new_cell=new_cell, time=time.time() - t0)


def hybrid(frame: Image.Image, s_cell, move, geom: SceneGeom, inpaint_pipe, steps=2):
    """Cutpaste the sprite, then diffusion-inpaint ONLY the vacated hole, and
    composite so ONLY the hole region is taken from the diffusion output."""
    t0 = time.time()
    img = np.array(frame.convert("RGB"))
    new_cell = apply_move(s_cell, move, geom)
    bbox = _find_s(img, geom, s_cell)
    pasted, _ = cutpaste(frame, s_cell, move, geom)
    if bbox is None or inpaint_pipe is None:
        return pasted, dict(new_cell=new_cell, time=time.time() - t0)
    import torch
    # inpaint only the OLD cell hole on the pasted frame
    l, t, r, b = [int(v) for v in geom.cell_bbox(*s_cell)]
    size = frame.size[0]
    work = 512; sc = work / size
    fr = pasted.resize((work, work))
    mask = Image.new("L", (work, work), 0)
    ImageDraw.Draw(mask).rectangle([int(l*sc)-4, int(t*sc)-4, int(r*sc)+4, int(b*sc)+4], fill=255)
    res = inpaint_pipe(prompt="clean fresh snow, top-down christmas game grid",
                       image=fr, mask_image=mask, num_inference_steps=steps,
                       guidance_scale=0.0, strength=0.99,
                       generator=torch.Generator("cuda").manual_seed(0)).images[0]
    res = np.array(res.resize((size, size)))
    # composite: take ONLY the old-cell hole from the diffusion result
    comp = np.array(pasted).copy()
    comp[t:b, l:r] = res[t:b, l:r]
    return Image.fromarray(comp), dict(new_cell=new_cell, time=time.time() - t0)
