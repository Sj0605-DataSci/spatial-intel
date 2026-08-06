"""Hospital frame compositor: build a clean game frame from sprites at MDP-known
positions, so we can render ANY state on demand (drift-free inference anchor +
exhaustive training data).

Layers (bottom->top):
  themed_ground (or base background)  ->  static objects at RDDL cells  ->  actor
  ->  HUD overlay (counters).

Uses the calibrated geometry (cell->pixel) + RDDL object cells + assets/sprites.
Handles effects: hidden objects removed, HUD counters drawn from state ints.

    python scripts/composite_frame.py --verify   # composite state 0, diff vs real
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"


def load_geom(game):
    return json.loads((BUNDLE / game / "geometry.json").read_text())


def cell_center(geom, col, row):
    ox, oy = geom["origin"]; cw, ch = geom["cell"]
    return ox + (col - 1) * cw, oy + (row - 1) * ch


def rddl_positions(game):
    """{obj: (col,row)} from the instance RDDL."""
    inst = next((BUNDLE / game / "source").glob("*instance*.rddl")).read_text()
    pos = {}
    for m in re.finditer(r"(\w+?)-([xy])\((\w+),\s*[xy](\d+)\)", inst):
        _, axis, obj, idx = m.group(1), m.group(2), m.group(3), int(m.group(4))
        pos.setdefault(obj, {})[axis] = idx
    return {o: (xy["x"], xy["y"]) for o, xy in pos.items() if "x" in xy and "y" in xy}


def paste_sprite(canvas, sprite_path, cx, cy, cell_px):
    """Paste an RGBA sprite centered at (cx,cy), scaled so its width ~= cell_px."""
    sp = Image.open(sprite_path).convert("RGBA")
    scale = cell_px / sp.width
    w, h = int(sp.width * scale), int(sp.height * scale)
    sp = sp.resize((w, h), Image.LANCZOS)
    # anchor: sprite's bottom-center sits slightly below the cell center (stands on cell)
    x = int(cx - w / 2); y = int(cy - h * 0.6)
    canvas.alpha_composite(sp, (x, y))


def _hud_state_index(game):
    """map ints-tuple -> a state id whose real frame has that HUD (for HUD strip reuse)."""
    idx = {}
    for l in open(BUNDLE / game / "states.jsonl"):
        s = json.loads(l)
        idx.setdefault(tuple(s["ints"]), s["id"])
    return idx


def composite(game, player_cell, ints=None, hidden=(), bg="themed_ground",
              hud_h=82, hud_index=None):
    """Build a frame: background + objects (minus hidden) + player, at MDP positions.
    If `ints` given, overlay the real HUD strip from a state with matching counters."""
    g = BUNDLE / game
    geom = load_geom(game)
    pos = rddl_positions(game)
    cell_px = int(abs(geom["cell"][0]) * 0.62)     # sprite ~ 0.62 of a cell width (matches real)

    # CLEAN background (no baked-in actor): HUD strip (from a real state) on top +
    # themed_ground (actor-free board) below. Real frame is 512x594, HUD ~82px,
    # board 512x512 = themed_ground size. This avoids the double-actor bug.
    ref = Image.open(g / "states/state_00000.png").convert("RGBA")
    W, H = ref.size
    ground = Image.open(g / "themed_ground.png").convert("RGBA")
    canvas = Image.new("RGBA", (W, H))
    canvas.paste(ref.crop((0, 0, W, hud_h)), (0, 0))     # HUD strip (placeholder, real HUD set later)
    canvas.paste(ground.resize((W, H - hud_h)), (0, hud_h))  # clean actor-free board

    # static objects (skip player + hidden)
    player_obj = next((o for o in pos if "player" in o), None)
    for obj, (col, row) in pos.items():
        if obj == player_obj or obj in hidden:
            continue
        sp = g / "assets/sprites" / f"{obj}.png"
        if sp.exists():
            cx, cy = cell_center(geom, col, row)
            paste_sprite(canvas, sp, cx, cy, cell_px)

    # player/actor at the requested cell
    if player_obj:
        sp = g / "assets/sprites" / f"{player_obj}.png"
        if sp.exists():
            cx, cy = cell_center(geom, *player_cell)
            paste_sprite(canvas, sp, cx, cy, cell_px)

    frame = canvas.convert("RGB")

    # HUD: overlay the real HUD strip from a state whose counters match `ints`
    # (avoids hand-rendering the game font/positions; perfect HUD numbers).
    if ints is not None:
        hud_index = hud_index or _hud_state_index(game)
        sid = hud_index.get(tuple(ints))
        if sid is not None:
            real = Image.open(g / f"states/state_{sid:05d}.png").convert("RGB")
            strip = real.crop((0, 0, W, hud_h))
            frame.paste(strip, (0, 0))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="hospital")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--base-set", action="store_true", help="generate actor-at-each-cell frames")
    ap.add_argument("--out", default="runs/composite_test")
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.base_set:
        geom = load_geom(args.game)
        cols, rows = geom["grid"]
        grid = Image.new("RGB", (0, 0))
        tiles = []
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                f = composite(args.game, (col, row), ints=[0, 0, 0, 0])
                f.save(out / f"cell_{col}_{row}.png")
                tiles.append((col, row, f))
        # contact sheet
        w0, h0 = tiles[0][2].size
        tw, th = w0 // 3, h0 // 3
        sheet = Image.new("RGB", (tw * cols, th * rows), (17, 24, 33))
        for col, row, f in tiles:
            sheet.paste(f.resize((tw, th)), ((col - 1) * tw, (row - 1) * th))
        sheet.save(out / "base_set_grid.png")
        print(f"generated {len(tiles)} base frames (actor at each cell)")
        print(f"-> {out}/base_set_grid.png")
        return

    if args.verify:
        # composite the initial state (player at 1,1) and compare to real state 0
        states = {json.loads(l)["id"]: json.loads(l)
                  for l in open(BUNDLE / args.game / "states.jsonl")}
        s0 = states[0]
        comp = composite(args.game, (s0["px"], s0["py"]))
        real = Image.open(BUNDLE / args.game / "states/state_00000.png").convert("RGB")
        comp.save(out / "composited.png"); real.save(out / "real.png")
        a = np.asarray(comp.resize(real.size)).astype(int)
        b = np.asarray(real).astype(int)
        mae = np.abs(a - b).mean()
        # side-by-side
        sb = Image.new("RGB", (real.width * 2 + 8, real.height), (17, 24, 33))
        sb.paste(comp.resize(real.size), (0, 0)); sb.paste(real, (real.width + 8, 0))
        sb.save(out / "compare.png")
        print(f"composited vs real state 0: MAE {mae:.1f}/255")
        print(f"-> {out}/compare.png (left=composited, right=real)")


if __name__ == "__main__":
    main()
