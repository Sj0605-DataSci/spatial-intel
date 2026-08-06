"""Eval a weighted-loss LoRA checkpoint on REAL edits: load the LoRA, feed an input
board + coordinate prompt, check where the triangle lands. The honest test (samples,
not the loss curve).

    python scripts/eval_weighted_ckpt.py --ckpt runs/lora_weighted2/ckpt_500 --n 6 --out runs/eval_weighted
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import GridSpec, render_state, to_array, apply_move, clamp_topleft
from spatial_intel.variants import detect_variant_topleft, Appearance

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir with pytorch_lora_weights.safetensors")
    ap.add_argument("--data", default="data/lora")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--out", default="runs/eval_weighted")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    # use TEST examples (unseen coords) — the real generalization check
    exs = manifest["test"][:args.n]

    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    wpath = Path(args.ckpt)
    if wpath.is_dir():
        wpath = wpath / "pytorch_lora_weights.safetensors"
    sd = load_file(str(wpath))
    pipe.transformer.load_lora_adapter(sd, prefix=None)
    print(f"loaded LoRA: {wpath} | active: {pipe.transformer.active_adapters()}")

    app = Appearance("red", "triangle")
    results = []
    for i, ex in enumerate(exs):
        start = tuple(ex["start"]); target = tuple(ex["target"])
        prompt = ex["prompt_coords"]
        src = render_state(spec, start).resize((512, 512))
        gen = torch.Generator("cuda").manual_seed(i)
        img = pipe(image=src, prompt=prompt, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, height=512, width=512, generator=gen).images[0]
        det, _ = detect_variant_topleft(to_array(img.resize((spec.img_w, spec.img_h))), spec, app)
        hit = (det == target)
        dist = 99 if det is None else float(np.hypot(det[0]-target[0], det[1]-target[1]))
        img.save(out / f"eval_{i}_start{start[0]}{start[1]}_tgt{target[0]}{target[1]}.png")
        results.append(dict(i=i, start=list(start), target=list(target),
                            detected=list(det) if det else None, hit=hit, dist=round(dist, 2)))
        print(f"[{i}] {start}->{target}: detected {det} {'✓ HIT' if hit else '✗ miss'} (dist {dist:.1f})", flush=True)

    n_hit = sum(r["hit"] for r in results)
    mean_dist = np.mean([r["dist"] for r in results])
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== WEIGHTED-LORA EVAL: {n_hit}/{len(results)} correct, mean dist {mean_dist:.2f} ===")
    print(f"-> {out}/  (open eval_*.png to see if triangle moved)")


if __name__ == "__main__":
    main()
