"""Stage 2 of exhaustive synthesis: erase the movable sprite -> clean background plate.

Takes the frame + the sprite's SAM mask (from extract_sprite.py), dilates the mask
a little (kill any red halo), and inpaints the hole with SD-1.5-inpainting so the
cell looks like empty board. The result is the 'empty plate' we composite sprites
onto to synthesize every (cell, action) transition.

    python scripts/inpaint_plate.py \
        --frame runs/diamond_dev/playthrough_demo/step00_start.png \
        --sprite-meta runs/sprite_santa/sprite_meta.json \
        --prompt "empty snowy game board cell, snow-covered ground, seamless" \
        --out runs/plate_santa
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.rl.policy import _load_hf_token_from_env_file


def rebuild_mask(frame_path, sprite_meta, device):
    """Re-run SAM2 to get the full-res boolean mask (meta only stored the bbox)."""
    from transformers import Sam2Model, Sam2Processor
    meta = json.loads(Path(sprite_meta).read_text())
    bbox = meta["bbox"]
    mid = "facebook/sam2-hiera-large"
    proc = Sam2Processor.from_pretrained(mid)
    model = Sam2Model.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
    img = Image.open(frame_path).convert("RGB")
    inputs = proc(images=img, input_boxes=[[bbox]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    mask = proc.post_process_masks(out.pred_masks, inputs["original_sizes"])[0][0, 0].bool().cpu().numpy()
    del model; torch.cuda.empty_cache()
    return img, mask


def dilate(mask, px):
    from scipy import ndimage
    return ndimage.binary_dilation(mask, iterations=px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--sprite-meta", required=True)
    ap.add_argument("--prompt", default="empty snowy ground, seamless winter game board, no objects")
    ap.add_argument("--dilate", type=int, default=6, help="mask dilation px (kill halo)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="runs/plate_out")
    args = ap.parse_args()

    _load_hf_token_from_env_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("SAM2: rebuilding sprite mask...")
    img, mask = rebuild_mask(args.frame, args.sprite_meta, device)
    mask_d = dilate(mask, args.dilate)
    print(f"  mask {int(mask.sum())}px -> dilated {int(mask_d.sum())}px")

    # SD-1.5 inpainting works at 512; we inpaint a CROP around the hole for quality,
    # then paste back into the full-res frame.
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16,
        safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    W, H = img.size
    ys, xs = np.where(mask_d)
    cx, cy = int(xs.mean()), int(ys.mean())
    # crop a padded square around the hole (context for the inpainter)
    half = max(96, int(1.6 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    crop = img.crop((x0, y0, x1, y1)).resize((512, 512))
    mcrop = Image.fromarray((mask_d[y0:y1, x0:x1] * 255).astype(np.uint8)).resize((512, 512))

    print(f"inpainting hole (crop {x1-x0}x{y1-y0} @ {cx},{cy})...")
    gen = torch.Generator(device).manual_seed(0)
    result = pipe(prompt=args.prompt, image=crop, mask_image=mcrop,
                  num_inference_steps=args.steps, guidance_scale=7.5, generator=gen).images[0]
    result = result.resize((x1 - x0, y1 - y0))

    # paste inpainted crop back; only where the mask was (feathered)
    plate = np.asarray(img).copy()
    region = np.asarray(result)
    m_region = mask_d[y0:y1, x0:x1]
    plate[y0:y1, x0:x1][m_region] = region[m_region]
    Image.fromarray(plate).save(out / "plate.png")
    # also save the mask (full-res) for the compositor's occupancy checks
    Image.fromarray((mask_d * 255).astype(np.uint8)).save(out / "sprite_mask.png")
    print(f"-> {out}/plate.png  (empty board, sprite erased)")
    print(f"-> {out}/sprite_mask.png")


if __name__ == "__main__":
    main()
