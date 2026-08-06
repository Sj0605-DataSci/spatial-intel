"""Evaluate a trained LoRA across the generalization matrix (Aradhya's design).

Axes:
  1. COORD split : SEEN cells (train pairs) vs UNSEEN cells (held-out test pairs).
                   Unseen coords never appeared in training as start OR target.
  2. APPEARANCE  : red_triangle (trained) | blue_triangle | red_square | blue_square.
                   Same coords, changed look → tests attribute generalization.
  3. IMAGE       : the standard grid (--cross-image renders an alternate board style
                   to test generalization beyond the training image).

Metric: success = the object (detected by its color) lands on the exact target cell.
We reuse the coord split from data/lora/manifest.json so SEEN/UNSEEN are exactly the
train/test coordinate sets.

    python scripts/eval_lora_matrix.py --lora runs/lora_coords/lora_final \
        --prompt-key prompt_coords --n 3 --out runs/eval_lora_coords
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import GridSpec
from spatial_intel.variants import Appearance, render_variant, detect_variant_topleft, to_array
from spatial_intel.dataset import instruction_text

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
APPEARANCES = [Appearance("red", "triangle"), Appearance("blue", "triangle"),
               Appearance("red", "square"), Appearance("blue", "square")]


def make_prompt(prompt_key, start, direction, steps, target, app):
    """Build the eval prompt. For non-red-triangle appearances we swap the object
    words so the instruction matches what we're asking it to draw."""
    obj = f"{app.color} {app.shape}"
    if prompt_key == "prompt_coords":
        return (f"Grid image. A {obj} is currently at cell ({start[0]},{start[1]}). "
                f"Move it to cell ({target[0]},{target[1]}). "
                f"One {obj}, snapped to that cell.")
    # natural language
    base = instruction_text(direction, steps).replace("red triangle", obj)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompt-key", choices=["prompt_nl", "prompt_coords"], required=True)
    ap.add_argument("--data", default="data/lora")
    ap.add_argument("--n", type=int, default=3, help="runs per (pair, appearance)")
    ap.add_argument("--max-pairs", type=int, default=12, help="cap pairs per split")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="store_true", help="eval base model (no LoRA)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())

    def pairs_from(split):
        seen = set(); ps = []
        for ex in manifest[split]:
            k = (tuple(ex["start"]), tuple(ex["target"]))
            if k in seen:
                continue
            seen.add(k)
            ps.append((tuple(ex["start"]), ex["direction"], ex["steps"], tuple(ex["target"])))
        return ps[:args.max_pairs]

    splits = {"SEEN": pairs_from("train"), "UNSEEN": pairs_from("test")}

    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if not args.baseline:
        # save_lora_adapter (PEFT) writes pytorch_lora_weights.safetensors; load
        # the state dict explicitly to avoid the weight-name auto-guess failing.
        from safetensors.torch import load_file
        wpath = Path(args.lora)
        if wpath.is_dir():
            wpath = wpath / "pytorch_lora_weights.safetensors"
        sd = load_file(str(wpath))
        # keys are transformer_blocks.*; prefix=None so they match the transformer
        pipe.transformer.load_lora_adapter(sd, prefix=None)
        print("loaded LoRA:", wpath, "| active:", pipe.transformer.active_adapters())
    else:
        print("BASELINE (no LoRA)")

    results = []
    gi = 0
    for split, ps in splits.items():
        for (start, d, st, target) in ps:
            for app in APPEARANCES:
                prompt = make_prompt(args.prompt_key, start, d, st, target, app)
                src = render_variant(spec, start, app).resize((args.img_size, args.img_size))
                succ = 0
                dists = []
                for run in range(args.n):
                    gen = torch.Generator("cuda").manual_seed(1000 * gi + run)
                    img = pipe(image=src, prompt=prompt, num_inference_steps=args.steps,
                               guidance_scale=args.guidance, height=args.img_size,
                               width=args.img_size, generator=gen).images[0]
                    arr = to_array(img.resize((spec.img_w, spec.img_h)))
                    det, conf = detect_variant_topleft(arr, spec, app)
                    if det == target:
                        succ += 1
                    dists.append(99 if det is None else
                                 float(np.hypot(det[0]-target[0], det[1]-target[1])))
                results.append(dict(split=split, start=list(start), target=list(target),
                                    appearance=app.name, n=args.n, success=succ,
                                    success_rate=round(succ/args.n, 3),
                                    mean_dist=round(float(np.mean(dists)), 2)))
                gi += 1
                print(f"[{split:6}] {app.name:14} {start}->{target}: "
                      f"{succ}/{args.n} succ", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2))

    # aggregate: success rate per (split x appearance)
    agg = {}
    for split in splits:
        agg[split] = {}
        for app in APPEARANCES:
            sub = [r for r in results if r["split"] == split and r["appearance"] == app.name]
            tot = sum(r["success"] for r in sub); n = sum(r["n"] for r in sub)
            agg[split][app.name] = dict(success=tot, n=n,
                                        rate=round(tot / n, 3) if n else 0.0)
    (out / "summary.json").write_text(json.dumps(dict(lora=args.lora,
        prompt_key=args.prompt_key, baseline=args.baseline, agg=agg), indent=2))

    print("\n=== SUCCESS RATE (split × appearance) ===")
    hdr = "split ".ljust(8) + "".join(a.name.ljust(15) for a in APPEARANCES)
    print(hdr)
    for split in splits:
        row = split.ljust(8) + "".join(f"{agg[split][a.name]['rate']*100:4.0f}%".ljust(15) for a in APPEARANCES)
        print(row)
    print(f"-> {out}/summary.json")


if __name__ == "__main__":
    main()
