"""VQA / latent round-trip diagnostic for the LoRA training data path.

Hypothesis: the training loss oscillated and the model never learned placement
because the DATA PATH is buggy — if VAE encode->decode doesn't faithfully
reproduce the target (triangle on the right cell), the model was trained against
corrupted latents.

Test, using the EXACT encode/pack path from train_lora_sft.py:
  target image -> _encode_vae_image -> _pack_latents         (what training sees as x0)
              -> _unpack -> bn-denorm -> _unpatchify -> vae.decode   (invert it)
  Then detect the triangle cell on BOTH original target and round-tripped image.
  If they differ, the latent path is broken -> explains the training failure.

Also runs a VLM (Qwen2.5-VL) as an objective judge on the round-trip.

    python scripts/vqa_latent_check.py --n 5 --out runs/vqa_check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from spatial_intel.grid import GridSpec
from spatial_intel.variants import detect_variant_topleft, Appearance


def roundtrip(pipe, img_pil, size=512, dtype=torch.bfloat16, device="cuda"):
    """encode->pack (as in training) then invert back to pixels."""
    a = torch.from_numpy(np.asarray(img_pil.resize((size, size)))).float() / 127.5 - 1.0
    t = a.permute(2, 0, 1).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        x0 = pipe._encode_vae_image(t, torch.Generator(device).manual_seed(0))  # (1,128,32,32)
        packed = pipe._pack_latents(x0)                                          # (1,1024,128)
        # --- invert (mirror of FluxPolicy._decode) ---
        lh = lw = 2 * (size // (pipe.vae_scale_factor * 2))
        # rebuild the latent ids so _unpack_latents_with_ids works
        num_ch = pipe.transformer.config.in_channels // 4
        _, latent_ids = pipe.prepare_latents(1, num_ch, size, size, dtype, device,
                                             torch.Generator(device).manual_seed(0))
        lat = pipe._unpack_latents_with_ids(packed, latent_ids, lh // 2, lw // 2)
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        lat = lat * std + mean
        lat = pipe._unpatchify_latents(lat)
        dec = pipe.vae.decode(lat.to(dtype), return_dict=False)[0]
        out = pipe.image_processor.postprocess(dec, output_type="np")[0]
    return Image.fromarray((out * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lora")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="runs/vqa_check")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    data_dir = Path(args.data)

    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    from diffusers import Flux2KleinPipeline
    print("loading FLUX.2-klein-base-4B (for its VAE)...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16).to("cuda")

    judge = None
    if args.judge:
        from spatial_intel.game.vlm_judge import VLMJudge
        judge = VLMJudge(); judge.load()

    app = Appearance("red", "triangle")
    results = []
    for i in range(args.n):
        ex = manifest["train"][i]
        tgt_path = data_dir / ex["target_img"]
        tgt = Image.open(tgt_path).convert("RGB")
        rt = roundtrip(pipe, tgt)
        # detect triangle cell on original target vs round-tripped
        det_orig, _ = detect_variant_topleft(np.asarray(tgt.resize((spec.img_w, spec.img_h))), spec, app)
        det_rt, _ = detect_variant_topleft(np.asarray(rt.resize((spec.img_w, spec.img_h))), spec, app)
        # pixel-level fidelity
        diff = float(np.abs(np.asarray(tgt.resize((512, 512))).astype(int)
                            - np.asarray(rt.resize((512, 512))).astype(int)).mean())
        faithful = (det_orig == det_rt)
        tgt.resize((512, 512)).save(out / f"{i}_target.png")
        rt.save(out / f"{i}_roundtrip.png")
        rec = dict(i=i, target_cell=ex["target"], det_target=list(det_orig) if det_orig else None,
                   det_roundtrip=list(det_rt) if det_rt else None,
                   cell_faithful=faithful, mean_pixel_diff=round(diff, 2))
        if judge:
            js = judge.judge(tgt.resize((512, 512)), rt, "identity (encode-decode)", tuple(ex["target"]))
            rec.update(vlm_scene_intact=js.scene_intact, vlm_quality=js.quality)
        results.append(rec)
        print(f"[{i}] target cell {tuple(ex['target'])}: orig detects {det_orig}, "
              f"roundtrip detects {det_rt} -> {'FAITHFUL' if faithful else 'CORRUPTED'} "
              f"(pixdiff {diff:.1f})", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    n_faithful = sum(r["cell_faithful"] for r in results)
    print(f"\n=== VAE ROUND-TRIP FIDELITY: {n_faithful}/{len(results)} faithful ===")
    print(f"mean pixel diff: {np.mean([r['mean_pixel_diff'] for r in results]):.1f}")
    if n_faithful < len(results):
        print(">>> DATA PATH IS BROKEN: training targets were corrupted by the "
              "encode/decode latent code. This explains the loss oscillation and "
              "the failure to learn placement.")
    else:
        print(">>> Data path is FAITHFUL. The training-failure cause is elsewhere "
              "(the model/loss, not the latent path).")
    print(f"-> {out}/results.json  (compare {out}/*_target.png vs *_roundtrip.png)")


if __name__ == "__main__":
    main()
