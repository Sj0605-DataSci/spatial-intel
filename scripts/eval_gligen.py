"""Evaluate the GLIGEN coord-grounded LoRA across the generalization matrix.

Same matrix as eval_lora_matrix (SEEN/UNSEEN coords x 4 appearances) but at
generation time we APPEND the grounding token (from the trained GroundingEncoder,
built from the target x,y) to the prompt embeddings — matching training.

We monkeypatch pipe.encode_prompt so the standard pipe(...) path uses the grounded
embeddings.

    python scripts/eval_gligen.py --lora runs/lora_gligen --n 2 --max-pairs 6 --out runs/eval_gligen
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from safetensors.torch import load_file
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import GridSpec, render_state, to_array
from spatial_intel.variants import Appearance, render_variant, detect_variant_topleft
from spatial_intel.dataset import instruction_text
from spatial_intel.rl.gligen import GroundingEncoder, append_grounding

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
APPEARANCES = [Appearance("red", "triangle"), Appearance("blue", "triangle"),
               Appearance("red", "square"), Appearance("blue", "square")]


def coord_prompt(start, target, app):
    obj = f"{app.color} {app.shape}"
    return (f"Grid image. A {obj} is currently at cell ({start[0]},{start[1]}). "
            f"Move it to cell ({target[0]},{target[1]}). One {obj}, snapped to that cell.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="dir with lora_final + grounder.pt")
    ap.add_argument("--data", default="data/lora")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())

    def pairs_from(split):
        seen, ps = set(), []
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
    lora_dir = Path(args.lora)
    sd = load_file(str(lora_dir / "lora_final" / "pytorch_lora_weights.safetensors"))
    pipe.transformer.load_lora_adapter(sd, prefix=None)
    edim = pipe.transformer.config["joint_attention_dim"]
    grounder = GroundingEncoder(embed_dim=edim).to("cuda", pipe.transformer.dtype)
    grounder.load_state_dict(torch.load(lora_dir / "grounder.pt"))
    grounder.eval()
    print("loaded LoRA + grounder | gate:", float(torch.tanh(grounder.gate)))

    # monkeypatch encode_prompt to append the grounding token for the current target
    orig_encode = pipe.encode_prompt
    cur = {"target": None}

    def grounded_encode(*a, **k):
        pe, tids = orig_encode(*a, **k)
        t = cur["target"]
        if t is not None:
            xy = torch.tensor([[t[0]/spec.max_topleft, t[1]/spec.max_topleft]],
                              device=pe.device, dtype=torch.float32)
            with torch.no_grad():
                g = grounder(xy)
            pe, tids = append_grounding(pe, tids, g)
        return pe, tids
    pipe.encode_prompt = grounded_encode

    results = []
    gi = 0
    for split, ps in splits.items():
        for (start, d, st, target) in ps:
            for app in APPEARANCES:
                cur["target"] = target
                prompt = coord_prompt(start, target, app)
                src = render_variant(spec, start, app).resize((args.img_size, args.img_size))
                succ, dists = 0, []
                for run in range(args.n):
                    gen = torch.Generator("cuda").manual_seed(1000*gi + run)
                    img = pipe(image=src, prompt=prompt, num_inference_steps=args.steps,
                               guidance_scale=args.guidance, height=args.img_size,
                               width=args.img_size, generator=gen).images[0]
                    det, _ = detect_variant_topleft(to_array(img.resize((spec.img_w, spec.img_h))), spec, app)
                    if det == target:
                        succ += 1
                    dists.append(99 if det is None else float(np.hypot(det[0]-target[0], det[1]-target[1])))
                results.append(dict(split=split, start=list(start), target=list(target),
                                    appearance=app.name, n=args.n, success=succ,
                                    success_rate=round(succ/args.n, 3),
                                    mean_dist=round(float(np.mean(dists)), 2)))
                gi += 1
                print(f"[{split:6}] {app.name:14} {start}->{target}: {succ}/{args.n}", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2))

    agg = {}
    for split in splits:
        agg[split] = {}
        for app in APPEARANCES:
            sub = [r for r in results if r["split"] == split and r["appearance"] == app.name]
            tot = sum(r["success"] for r in sub); n = sum(r["n"] for r in sub)
            agg[split][app.name] = dict(success=tot, n=n, rate=round(tot/n, 3) if n else 0.0)
    (out / "summary.json").write_text(json.dumps(dict(lora=str(lora_dir), agg=agg), indent=2))
    print("\n=== GLIGEN SUCCESS RATE (split × appearance) ===")
    for split in splits:
        print(split.ljust(8) + "".join(f"{agg[split][a.name]['rate']*100:4.0f}%".ljust(15) for a in APPEARANCES))
    print(f"-> {out}/summary.json")


if __name__ == "__main__":
    main()
