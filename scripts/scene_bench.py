"""Speed + quality benchmark for small/fast diffusion models on the Santa grid
scene task (prompt-only, no training).

Two axes (per the reframed goal):
  1. SPEED   — wall-clock inference time per image (the primary target: fast!)
  2. ACCURACY— all-object placement, VLM-scored (separate step)
  + QUALITY  — the generated image, for visual comparison

Because the target models span short-context CLIP (SD1.5/SDXL, 77 tokens) to long
(FLUX), we build a COMPACT scene prompt from the YAML rather than dumping the whole
spec. The grid layout / coordinates are described concisely.

    python scripts/scene_bench.py --model sdxl-turbo --n 3 --out runs/scene_bench
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# Model registry: id -> (hf repo, pipeline loader kind, default steps, guidance)
MODELS = {
    "sd-turbo":        ("stabilityai/sd-turbo",              "sd",   1,  0.0),
    "sdxl-turbo":      ("stabilityai/sdxl-turbo",            "sdxl", 1,  0.0),
    "sd15-lcm":        ("Lykon/dreamshaper-7",               "lcm",  4,  1.5),
    "sdxl-lightning":  ("ByteDance/SDXL-Lightning",          "light",4,  0.0),
    "hyper-sd":        ("ByteDance/Hyper-SD",                "hyper",1,  0.0),
    "flux2-klein-4b":  ("black-forest-labs/FLUX.2-klein-4B", "flux", 4,  1.0),
}

# compact prompt built from the YAML spec (fits CLIP 77-token where possible)
SCENE_PROMPT = (
    "A charming pixel-art Christmas game scene on an 8x8 grid in a snowy neighborhood. "
    "Santa 'S' in a red badge top-left; four wrapped presents P1 red, P2 green, P3 blue, "
    "P4 purple placed on the upper grid; four cozy decorated houses A B C D in a row along "
    "the bottom; a reindeer stable 'R' in the bottom-left corner; snow piles and small "
    "Christmas trees scattered on the grid. Top-down view, festive, clean grid lines."
)


def load_pipe(model_key, device="cuda"):
    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    repo, kind, steps, guid = MODELS[model_key]
    import torch
    if kind == "sd":
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16")
    elif kind == "sdxl":
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16")
    elif kind == "lcm":
        from diffusers import AutoPipelineForText2Image, LCMScheduler
        pipe = AutoPipelineForText2Image.from_pretrained(repo, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    elif kind == "light":
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, "sdxl_lightning_4step_unet.safetensors"), device=device))
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif kind == "hyper":
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DDIMScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, "Hyper-SDXL-1step-Unet.safetensors"), device=device))
        pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif kind == "flux":
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, steps, guid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n", type=int, default=3, help="images to time")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="runs/scene_bench")
    args = ap.parse_args()

    out = Path(args.out) / args.model; out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model} ({MODELS[args.model][0]})...")
    t0 = time.time()
    pipe, steps, guid = load_pipe(args.model)
    load_s = time.time() - t0
    size = args.size if MODELS[args.model][1] != "sd" else 512  # sd1.5 is 512

    # warmup (first call compiles/allocates; not counted)
    _ = pipe(SCENE_PROMPT, num_inference_steps=steps, guidance_scale=guid,
             height=size, width=size, generator=torch.Generator("cuda").manual_seed(0)).images[0]
    torch.cuda.synchronize()

    times = []
    for i in range(args.n):
        torch.cuda.synchronize(); s = time.time()
        img = pipe(SCENE_PROMPT, num_inference_steps=steps, guidance_scale=guid,
                   height=size, width=size, generator=torch.Generator("cuda").manual_seed(i+1)).images[0]
        torch.cuda.synchronize()
        dt = time.time() - s
        times.append(dt)
        img.save(out / f"gen_{i}.png")
        print(f"  gen {i}: {dt:.2f}s", flush=True)

    import numpy as np
    res = dict(model=args.model, repo=MODELS[args.model][0], steps=steps, size=size,
               load_s=round(load_s, 1),
               mean_s=round(float(np.mean(times)), 3),
               median_s=round(float(np.median(times)), 3),
               imgs_per_s=round(1.0 / np.mean(times), 2))
    (out / "speed.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== {args.model}: {res['median_s']}s/img ({res['imgs_per_s']} img/s), "
          f"{steps} steps, {size}px, load {res['load_s']}s ===")


if __name__ == "__main__":
    main()
