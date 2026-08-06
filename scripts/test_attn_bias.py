"""Test the training-free attention-bias intervention: does forcing the coord-token
attention onto the (VLM-predicted) target cell fix placement?

For each command: generate WITHOUT bias and WITH bias (same seed), compare where
the triangle lands. Uses the VLM to predict the target cell (the 'thinking'), then
biases FLUX's attention toward it.

    python scripts/test_attn_bias.py --n-cmds 6 --strength 6 --out runs/attn_bias
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import (GridSpec, render_state, to_array, apply_move,
                                 clamp_topleft, max_steps_for, DIRECTION_NAMES)
from spatial_intel.variants import Appearance, detect_variant_topleft
from spatial_intel.dataset import instruction_text
from spatial_intel.rl.attn_bias import (BiasSpec, install_bias, remove_bias,
                                        target_cell_patches)
from spatial_intel.rl.attention import find_token_ids

FLUX = "black-forest-labs/FLUX.2-klein-base-4B"
SIDE = 32  # generated-image patch grid side for 512px


def coord_prompt(start, target):
    return (f"Grid image. A red triangle is currently at cell ({start[0]},{start[1]}). "
            f"Move it to cell ({target[0]},{target[1]}). "
            "One red triangle, snapped to that cell.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cmds", type=int, default=6)
    ap.add_argument("--strength", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--use-vlm", action="store_true",
                    help="use the VLM to predict target (else use ground-truth target)")
    ap.add_argument("--out", default="runs/attn_bias")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    rng = np.random.default_rng(args.seed)

    # sample a few commands
    cmds = []
    while len(cmds) < args.n_cmds:
        start = (int(rng.integers(0, spec.max_topleft + 1)),
                 int(rng.integers(0, spec.max_topleft + 1)))
        d = DIRECTION_NAMES[int(rng.integers(len(DIRECTION_NAMES)))]
        hi = max_steps_for(spec, start, d)
        if hi < 1:
            continue
        mv = int(rng.integers(1, hi + 1))
        target = clamp_topleft(spec, apply_move(spec, start, d, mv))
        if target != start:
            cmds.append((start, d, mv, target))

    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    pipe = Flux2KleinPipeline.from_pretrained(FLUX, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    reasoner = None
    if args.use_vlm:
        from spatial_intel.rl.vlm_reasoner import VLMReasoner, VLMConfig, QWEN_ID
        reasoner = VLMReasoner(VLMConfig(model_id=QWEN_ID, max_new_tokens=64))
        reasoner.load()

    results = []
    app = Appearance("red", "triangle")
    for i, (start, d, mv, target) in enumerate(cmds):
        prompt = coord_prompt(start, target)
        src = render_state(spec, start).resize((512, 512))
        # tokens to bias: triangle + target coord digits
        tids = find_token_ids(pipe, prompt, ["triangle", str(target[0]), str(target[1])])
        pe, _ = pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1)
        text_len = pe.shape[1]

        # which cell to steer toward: VLM prediction or ground truth
        steer = target
        vlm_txt = None
        if reasoner is not None:
            vlm_txt = reasoner.reason(render_state(spec, start), instruction_text(d, mv))
            import re
            m = re.search(r"\((\d)\s*,\s*(\d)\)", vlm_txt)
            if m:
                steer = (int(m.group(1)), int(m.group(2)))

        patches = target_cell_patches(steer, spec, SIDE, spec.object_size)
        bspec = BiasSpec(token_rows=tids, text_len=text_len, n_gen_patches=SIDE*SIDE,
                         side=SIDE, target_cols=patches, strength=args.strength, enabled=True)

        def gen():
            g = torch.Generator("cuda").manual_seed(args.seed + i)
            return pipe(image=src, prompt=prompt, num_inference_steps=args.steps,
                        guidance_scale=args.guidance, height=512, width=512, generator=g).images[0]

        # OFF
        bspec.enabled = False
        off = gen()
        d_off, _ = detect_variant_topleft(to_array(off.resize((spec.img_w, spec.img_h))), spec, app)
        # ON
        orig, procs = install_bias(pipe.transformer, bspec)
        bspec.enabled = True
        on = gen()
        remove_bias(pipe.transformer, orig)
        d_on, _ = detect_variant_topleft(to_array(on.resize((spec.img_w, spec.img_h))), spec, app)

        off.save(out / f"cmd{i}_off.png"); on.save(out / f"cmd{i}_on.png")
        rec = dict(cmd=i, start=list(start), target=list(target), steer=list(steer),
                   vlm=vlm_txt, det_off=d_off, det_on=d_on,
                   hit_off=(d_off == target), hit_on=(d_on == target),
                   dist_off=None if d_off is None else float(np.hypot(d_off[0]-target[0], d_off[1]-target[1])),
                   dist_on=None if d_on is None else float(np.hypot(d_on[0]-target[0], d_on[1]-target[1])))
        results.append(rec)
        print(f"cmd{i} {start}->{target} steer{steer}: OFF={d_off}(hit={rec['hit_off']}) "
              f"ON={d_on}(hit={rec['hit_on']})", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    n = len(results)
    print(f"\n=== ATTENTION-BIAS EFFECT (strength={args.strength}, n={n}) ===")
    print(f"  hit rate OFF: {sum(r['hit_off'] for r in results)}/{n}")
    print(f"  hit rate ON:  {sum(r['hit_on'] for r in results)}/{n}")
    moved = sum(1 for r in results if r['det_off'] != r['det_on'])
    print(f"  bias CHANGED placement in {moved}/{n} cases")
    closer = sum(1 for r in results if r['dist_on'] is not None and r['dist_off'] is not None
                 and r['dist_on'] < r['dist_off'])
    print(f"  bias got CLOSER to target in {closer}/{n} cases")


if __name__ == "__main__":
    main()
