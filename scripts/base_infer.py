"""Plain base-4B img2img inference: input board + prompt -> output image.

No RL, no LoRA — just see what FLUX.2-klein-BASE-4B does with one edit command,
so we have a baseline before the RLVR feedback loop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

from spatial_intel.grid import (GridSpec, render_state, to_array, apply_move,
                                 clamp_topleft)
from spatial_intel.dataset import instruction_text

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
OUT = Path("runs/base_infer"); OUT.mkdir(parents=True, exist_ok=True)

spec = GridSpec()
# the brief's example: center start, move right by 1
start = (spec.max_topleft // 2, spec.max_topleft // 2)
direction, steps = "right", 1
target = clamp_topleft(spec, apply_move(spec, start, direction, steps))
instr = instruction_text(direction, steps)

# input board (triangle at start) + ground-truth target, for comparison
input_img = render_state(spec, start)
to_array(input_img)
Image.fromarray(to_array(input_img)).save(OUT / "input.png")
Image.fromarray(to_array(render_state(spec, target))).save(OUT / "target.png")

prompt = (
    "A square grid with numbered columns and rows and a single solid red "
    "triangle occupying one square block. " + instr +
    " Keep the grid, axes and labels identical; move ONLY the red triangle "
    "to its new grid block; output one solid red triangle snapped to the grid."
)
print("instruction:", instr)
print("start:", start, "-> target:", target)

print("loading base-4B...")
pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
pipe.set_progress_bar_config(disable=True)

print("running img2img inference...")
out = pipe(
    image=input_img.resize((512, 512)),
    prompt=prompt,
    num_inference_steps=28,        # base wants ~28-50 steps (not the 4-step distilled)
    guidance_scale=4.0,
    height=512, width=512,
    generator=torch.Generator(device="cuda").manual_seed(0),
).images[0]
out.save(OUT / "output.png")
print("DONE — open runs/base_infer/{input,target,output}.png")
