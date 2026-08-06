"""Stage 1 of exhaustive synthesis: extract the movable sprite CLEANLY.

  Qwen2.5-VL  -> precise bbox of the movable character ("coordinates via a model")
  SAM2        -> pixel-perfect mask of the sprite inside that bbox
  -> save  sprite RGBA (cut-out with alpha)  +  a mask preview overlay

This replaces the brittle color-threshold cutpaste. Background inpainting (the
hole fill) is a separate stage (inpaint_plate.py).

    python scripts/extract_sprite.py \
        --frame runs/diamond_dev/playthrough_demo/step00_start.png \
        --query "the red Santa 'S' badge (a solid red circle with a white S)" \
        --out runs/sprite_santa
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.rl.policy import _load_hf_token_from_env_file


def vlm_bbox(frame_path, query, device):
    """Qwen2.5-VL: return [x0,y0,x1,y1] pixel bbox of `query` in the frame."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map=device)
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    prompt = (f"Locate {query} in this image. Respond ONLY with its bounding box as "
              f'JSON: {{"bbox_2d": [x0, y0, x1, y1]}} in pixel coordinates.')
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    reply = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", reply)
    if not m:
        raise RuntimeError(f"VLM gave no bbox. Reply: {reply!r}")
    box = [float(x) for x in m.groups()]
    # Qwen sometimes normalizes to 0-1000; rescale if so
    if max(box) <= 1000 and (box[2] <= 1000 and box[3] <= 1000) and W > 1000:
        box = [box[0] / 1000 * W, box[1] / 1000 * H, box[2] / 1000 * W, box[3] / 1000 * H]
    del model
    torch.cuda.empty_cache()
    return [int(round(v)) for v in box], reply


def sam2_mask(frame_path, bbox, device):
    """SAM2 prompted with the bbox -> boolean mask (H,W)."""
    from transformers import Sam2Model, Sam2Processor
    mid = "facebook/sam2-hiera-large"
    proc = Sam2Processor.from_pretrained(mid)
    model = Sam2Model.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
    img = Image.open(frame_path).convert("RGB")
    inputs = proc(images=img, input_boxes=[[bbox]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks, inputs["original_sizes"])[0]
    mask = masks[0, 0].bool().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--query", required=True, help="natural-language description of the sprite")
    ap.add_argument("--out", default="runs/sprite_out")
    args = ap.parse_args()

    _load_hf_token_from_env_file()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("Qwen2.5-VL: locating sprite bbox...")
    bbox, reply = vlm_bbox(args.frame, args.query, device)
    print(f"  bbox = {bbox}   (raw: {reply.strip()[:80]})")

    print("SAM2: segmenting sprite mask...")
    mask = sam2_mask(args.frame, bbox, device)
    print(f"  mask covers {int(mask.sum())} px")

    img = np.asarray(Image.open(args.frame).convert("RGB"))
    # tight-crop the sprite RGBA to its mask bbox
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = img[y0:y1, x0:x1]
    rgba[..., 3] = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    Image.fromarray(rgba).save(out / "sprite_rgba.png")

    # overlay preview: mask edge on the frame
    prev = img.copy()
    edge = mask ^ np.pad(mask, 1)[2:, 2:]
    prev[edge] = [0, 255, 255]
    Image.fromarray(prev).save(out / "mask_overlay.png")
    (out / "sprite_meta.json").write_text(json.dumps(
        {"frame": args.frame, "query": args.query, "bbox": bbox,
         "mask_bbox": [int(x0), int(y0), int(x1), int(y1)]}, indent=2))
    print(f"-> {out}/sprite_rgba.png  (cut-out)")
    print(f"-> {out}/mask_overlay.png (verify the mask is tight on the sprite)")


if __name__ == "__main__":
    main()
