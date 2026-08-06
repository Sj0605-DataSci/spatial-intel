"""Generate a detailed markdown report of the ablation: every arm, every command,
the exact prompt sent to FLUX, the VLM reasoning (arms 4), the input/target/output
images, and per-command + aggregate results.

    python scripts/make_report.py --out runs/REPORT.md

Reads all runs/abl_*/ablation.json. Images are linked relative to the report so
they render in the IDE / on GitHub.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_intel.grid import GridSpec, render_state, to_array
from spatial_intel.dataset import instruction_text
from scripts.ablation import build_prompt, FLUX
from spatial_intel.rl.vlm_reasoner import QWEN_ID, VIGORL_ID

ARM_DESC = {
    1: "text only (baseline)",
    2: "text + explicit target coords",
    3: "text + chain-of-thought + coords",
    4: "text + small-VLM grounded reasoning + coords",
    5: "non-verbose terse localised instruction",
    6: "verbose localised instruction",
}

# arm dir name -> (arm number, vlm label)
ARM_DIRS = [
    ("abl_arm1", 1, None),
    ("abl_arm2", 2, None),
    ("abl_arm3", 3, None),
    ("abl_arm4_qwen", 4, "qwen"),
    ("abl_arm4_vigorl", 4, "vigorl"),
    ("abl_arm5", 5, None),
    ("abl_arm6", 6, None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/REPORT.md")
    args = ap.parse_args()

    runs = Path(args.runs)
    spec = GridSpec()
    L = []  # lines

    L.append("# Spatial-Intel — Reasoning-Conditioning Ablation Report\n")
    L.append("Auto-generated. The image task is held trivial (move one red triangle "
             "from cell A to cell B); the variable is **how much reasoning "
             "scaffolding conditions the diffusion model**.\n")

    # -- models used --
    L.append("## Models used\n")
    L.append(f"- **Image generator:** `{FLUX}` (FLUX.2-klein-base-4B, flow-matching, "
             "img2img, 28 steps, guidance 4.0)")
    L.append(f"- **VLM reasoner (arm 4, qwen):** `{QWEN_ID}`")
    L.append(f"- **VLM reasoner (arm 4, vigorl):** `{VIGORL_ID}` (RL-trained for spatial grounding)")
    L.append(f"- **Board / scorer:** {spec.n}x{spec.n} grid, object {spec.object_size}x"
             f"{spec.object_size}; success = triangle centroid on the exact target cell.\n")

    # -- aggregate table --
    L.append("## Aggregate results\n")
    L.append("| Arm | Conditioning | n | Success rate | Mean cell-dist |")
    L.append("|-----|--------------|---|--------------|----------------|")
    arms_data = []
    for dirname, arm, vlm in ARM_DIRS:
        j = runs / dirname / "ablation.json"
        if not j.exists():
            L.append(f"| {arm}{'-'+vlm if vlm else ''} | {ARM_DESC[arm]} | — | *(not run)* | — |")
            continue
        d = json.loads(j.read_text())
        a = d["agg"]
        label = f"{arm}" + (f"-{vlm}" if vlm else "")
        L.append(f"| {label} | {ARM_DESC[arm]} | {a['n']} | "
                 f"{a['success_rate']*100:.1f}% | {a['mean_cell_distance']} |")
        arms_data.append((dirname, arm, vlm, d))
    L.append("")

    # -- per-arm detail --
    for dirname, arm, vlm, d in arms_data:
        label = f"Arm {arm}" + (f" ({vlm})" if vlm else "")
        L.append(f"\n---\n\n## {label} — {ARM_DESC[arm]}\n")
        for r in d["results"]:
            start = tuple(r["start"]); target = tuple(r["target"])
            d_, mv = r["direction"], r["move"]
            instr = instruction_text(d_, mv)
            prompt = build_prompt(arm, spec, start, d_, mv, target,
                                  vlm_reasoning=r.get("vlm_reasoning"))
            ci = r["cmd"]
            L.append(f"### cmd {ci:02d}: {d_} ×{mv}  {start} → {target}\n")
            L.append(f"- **instruction:** {instr}")
            if r.get("vlm_reasoning") is not None:
                ok = "✓" if str(tuple(target)).replace(" ", "") in r["vlm_reasoning"].replace(" ", "") else "✗ (wrong)"
                L.append(f"- **VLM reasoning:** `{r['vlm_reasoning']}`  → {ok}")
            L.append(f"- **result:** detected cell `{r['detected']}`, target `{target}` "
                     f"— **{'SUCCESS' if r['success'] else 'miss'}** "
                     f"(cell-distance {r['cell_distance']})")
            L.append(f"\n  **full prompt sent to FLUX:**\n\n  > {prompt}\n")
            # images
            img_in = runs / dirname / f"_in_cmd{ci:02d}.png"
            img_tgt = runs / dirname / f"_tgt_cmd{ci:02d}.png"
            # render input + target on the fly for the report (cheap, no model)
            from PIL import Image
            Image.fromarray(to_array(render_state(spec, start))).save(img_in)
            Image.fromarray(to_array(render_state(spec, target))).save(img_tgt)
            out_img = runs / dirname / f"cmd{ci:02d}.png"
            row = (f"| input | target | FLUX output |\n|---|---|---|\n"
                   f"| ![in]({img_in.name}) | ![tgt]({img_tgt.name}) | "
                   + (f"![out]({out_img.name})" if out_img.exists() else "*(no image)*")
                   + " |")
            L.append(row + "\n")

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out} ({len(arms_data)} arms documented)")


if __name__ == "__main__":
    main()
