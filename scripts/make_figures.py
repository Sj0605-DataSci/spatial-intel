"""Render publication figures from the 150-run failure grid for the Notion doc.

Produces into runs/fgrid_main/figs/:
  fig_failure_dist.png      : failure-mode distribution bar chart
  fig_attn_by_outcome.png   : mean attention focus-ratio per outcome (the key result)
  fig_per_init.png          : failure distribution per initial position
  fig_heatmap_<mode>.png    : example attention heatmap overlaid on the generated
                              image, one per failure mode (SUCCESS / POSITION_ERROR / ...)
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RUN = Path("runs/fgrid_main")
FIGS = RUN / "figs"; FIGS.mkdir(exist_ok=True)
results = json.loads((RUN / "results.json").read_text())
summary = json.loads((RUN / "summary.json").read_text())

MODES = ["SUCCESS", "POSITION_ERROR", "SHAPE_DESTROYED", "OBJECT_MISIDENTIFY", "MIXED"]
COLORS = {"SUCCESS": "#2ca02c", "POSITION_ERROR": "#ff7f0e", "SHAPE_DESTROYED": "#d62728",
          "OBJECT_MISIDENTIFY": "#9467bd", "MIXED": "#8c564b"}

# -- fig 1: failure distribution --
counts = Counter(r["primary"] for r in results)
fig, ax = plt.subplots(figsize=(7, 4))
vals = [counts.get(m, 0) for m in MODES]
bars = ax.bar(MODES, vals, color=[COLORS[m] for m in MODES])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v}\n({v/len(results)*100:.0f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("count (of 150 runs)")
ax.set_title("Failure-mode distribution — base-4B, terse prompt, K=3×P=5×n=10")
plt.xticks(rotation=20, ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "fig_failure_dist.png", dpi=130); plt.close()

# -- fig 2: attention focus by outcome (THE key result) --
fig, ax = plt.subplots(figsize=(7, 4))
order = ["SUCCESS", "POSITION_ERROR", "MIXED", "OBJECT_MISIDENTIFY"]
means = []
for m in order:
    fr = [r["attn_focus_ratio"] for r in results if r["primary"] == m]
    means.append(np.mean(fr) if fr else 0)
bars = ax.bar(order, means, color=[COLORS[m] for m in order])
ax.axhline(1.0, ls="--", c="gray", lw=1, label="no localization (ratio=1)")
for b, v in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("mean attention focus-ratio\n(attn on target cell ÷ global mean)")
ax.set_title("Latent reasoning: attention focus PREDICTS success")
ax.legend(fontsize=8)
plt.xticks(rotation=15, ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "fig_attn_by_outcome.png", dpi=130); plt.close()

# -- fig 3: per-init breakdown --
inits = list(summary["per_init"].keys())
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(inits))
for m in MODES:
    vals = [summary["per_init"][p]["dist"][m]["count"] for p in inits]
    ax.bar(inits, vals, bottom=bottom, label=m, color=COLORS[m])
    bottom += np.array(vals)
ax.set_ylabel("count (of 50 runs each)")
ax.set_title("Failure distribution per initial position (uniform → not a position artifact)")
ax.legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig(FIGS / "fig_per_init.png", dpi=130); plt.close()

# -- fig 4: INTERPRETABLE attention viz, one per mode --
# Three panels: (1) FLUX output, (2) attention overlaid on the INPUT board (grid
# + labels visible) with the target cell boxed, (3) per-CELL attention heatmap
# (one value per grid cell) with the target cell outlined. Panels 2&3 fix the
# earlier bug of overlaying patch-attention on the noisy output.
from matplotlib.patches import Rectangle
from spatial_intel.grid import GridSpec, render_state, to_array
spec = GridSpec()
N = spec.n


def per_cell(heat):
    """Reduce the 32x32 patch heatmap to an NxN per-cell grid (mean per cell)."""
    side = heat.shape[0]
    cw = side / N
    grid = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            r0, r1 = int(r*cw), int((r+1)*cw)
            c0, c1 = int(c*cw), int((c+1)*cw)
            grid[r, c] = heat[r0:r1, c0:c1].mean()
    return grid


def viz_for(mode):
    cand = [r for r in results if r["primary"] == mode]
    if not cand:
        return None
    r = max(cand, key=lambda x: x["attn_focus_ratio"]) if mode == "SUCCESS" else cand[len(cand)//2]
    s = r["start"]; t = r["target"]; run = r["run"]
    stem = f"s{s[0]}{s[1]}_t{t[0]}{t[1]}_r{run}"
    img_p = RUN / "img" / f"{stem}.png"
    heat_p = RUN / "attn" / f"{stem}.npy"
    if not img_p.exists() or not heat_p.exists():
        return None
    out_img = np.array(Image.open(img_p).resize((256, 256)))
    heat = np.load(heat_p)
    board = to_array(render_state(spec, tuple(s)))           # INPUT board (grid visible)
    board = np.array(Image.fromarray(board).resize((256, 256)))
    heat_up = np.array(Image.fromarray((heat*255).astype(np.uint8)).resize((256, 256)))
    cell_w = 256 / N

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    # panel 1: output
    axs[0].imshow(out_img); axs[0].axis("off")
    axs[0].set_title(f"{mode}\nFLUX output", fontsize=10)
    # panel 2: attention over the INPUT board + target box
    axs[1].imshow(board); axs[1].imshow(heat_up, cmap="jet", alpha=0.5)
    axs[1].add_patch(Rectangle((t[0]*cell_w, t[1]*cell_w), spec.object_size*cell_w,
                               spec.object_size*cell_w, fill=False, edgecolor="lime", lw=2.5))
    axs[1].axis("off")
    axs[1].set_title(f"attention on INPUT board\n(green box = target cell {tuple(t)})", fontsize=10)
    # panel 3: per-cell grid heatmap + target outline
    grid = per_cell(heat)
    im = axs[2].imshow(grid, cmap="hot")
    axs[2].add_patch(Rectangle((t[0]-0.5, t[1]-0.5), spec.object_size, spec.object_size,
                               fill=False, edgecolor="lime", lw=2.5))
    axs[2].set_xticks(range(N)); axs[2].set_yticks(range(N))
    axs[2].set_title(f"per-cell attention\nfocus-ratio={r['attn_focus_ratio']:.2f}", fontsize=10)
    plt.tight_layout(); out = FIGS / f"fig_heatmap_{mode}.png"
    plt.savefig(out, dpi=130); plt.close()
    return out.name

made = []
for m in ["SUCCESS", "POSITION_ERROR", "OBJECT_MISIDENTIFY", "MIXED"]:
    r = viz_for(m)
    if r: made.append(r)

print("figures written to", FIGS)
for f in sorted(FIGS.glob("*.png")):
    print("  ", f.name)
