"""Supervised FLUX.2-klein edit LoRA with the TWO fixes we diagnosed:

  FIX 1 — timestep loss weighting (kills the spikes / oscillation):
    flow-matching loss is naturally huge at high t. We weight each sample by the
    SD3-style factor so high-t steps don't dominate -> stable, readable loss.

  FIX 2 — REGION-weighted loss (makes the move actually learnable):
    the moved object is only ~4-19% of the image, so plain whole-image MSE drowns
    the move signal in background-rendering loss. We upweight the latent patches
    covering the START cell (must be cleared) and TARGET cell (must be filled), so
    the placement error dominates the gradient instead of being a rounding error.

    loss = mean( w_region * w_timestep * (v_pred - v_target)^2 )

Uses the same conditioning path as train_lora_sft.py (input board -> img_lat).

    python scripts/train_lora_weighted.py --data data/lora --prompt-key prompt_coords \
        --region-weight 20 --steps 2000 --out runs/lora_weighted
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

from spatial_intel.grid import GridSpec
from spatial_intel.rl.attn_bias import target_cell_patches

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
SIDE = 32
spec = GridSpec()


def load_pipe(rank, alpha, device="cuda"):
    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    for p in pipe.transformer.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig
    cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
                     target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj",
                                     "add_k_proj", "add_v_proj"])
    pipe.transformer.add_adapter(cfg)
    return pipe


def img_to_tensor(path, size, device, dtype):
    im = Image.open(path).convert("RGB").resize((size, size))
    a = torch.from_numpy(np.asarray(im).copy()).float() / 127.5 - 1.0
    return a.permute(2, 0, 1).unsqueeze(0).to(device, dtype)


def region_weight_map(start_cell, target_cell, region_w, device):
    """Per-token (1024) weight: `region_w` on patches covering start|target cells,
    1.0 elsewhere. Latent is 32x32 patches packed row-major -> matches target_cell_patches."""
    w = torch.ones(SIDE * SIDE, device=device)
    patches = set(target_cell_patches(tuple(start_cell), spec, SIDE, spec.object_size))
    patches |= set(target_cell_patches(tuple(target_cell), spec, SIDE, spec.object_size))
    idx = torch.tensor([p for p in patches if p < SIDE * SIDE], device=device)
    if len(idx):
        w[idx] = region_w
    return w  # (1024,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lora")
    ap.add_argument("--prompt-key", choices=["prompt_nl", "prompt_coords"], default="prompt_coords")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--region-weight", type=float, default=20.0,
                    help="how much to upweight the start+target cell patches")
    ap.add_argument("--flow-shift", type=float, default=3.16)
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlflow", action="store_true", help="log to local MLflow")
    ap.add_argument("--mlflow-uri", default="sqlite:///runs/mlflow.db",
                    help="MLflow tracking backend (local sqlite, no cloud)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"; torch.manual_seed(args.seed)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    train = manifest["train"]
    data_dir = Path(args.data)

    ml = None
    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)      # local file store, no cloud
        mlflow.set_experiment("klein-weighted")
        ml = mlflow.start_run(run_name=Path(args.out).name)
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})

    pipe = load_pipe(args.lora_rank, args.lora_alpha, device)
    dtype = pipe.transformer.dtype
    params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"trainable LoRA tensors: {len(params)} | region_weight={args.region_weight}")

    rng = np.random.default_rng(args.seed)
    log = []
    for step in range(args.steps):
        ex = train[int(rng.integers(len(train)))]
        prompt = ex[args.prompt_key]
        with torch.no_grad():
            pe, text_ids = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
            tgt = img_to_tensor(data_dir / ex["target_img"], args.img_size, device, dtype)
            x0 = pipe._encode_vae_image(tgt, torch.Generator(device).manual_seed(step))
            x0 = pipe._pack_latents(x0)
            inp = img_to_tensor(data_dir / ex["input"], args.img_size, device, dtype)
            img_lat, img_ids = pipe.prepare_image_latents([inp], 1,
                                torch.Generator(device).manual_seed(step), device, dtype)
            num_ch = pipe.transformer.config.in_channels // 4
            _, latent_ids = pipe.prepare_latents(1, num_ch, args.img_size, args.img_size,
                                dtype, device, torch.Generator(device).manual_seed(step))

        x0 = x0.float()
        noise = torch.randn_like(x0)
        # --- FIX 1: logit-normal timestep + weighting ---
        u = torch.randn(1, device=device) * args.logit_std + args.logit_mean
        t = torch.sigmoid(u)
        shift = args.flow_shift
        t = (shift * t) / (1 + (shift - 1) * t)
        xt = (1 - t) * x0 + t * noise
        v_target = noise - x0
        # Timestep weighting: logit-normal SAMPLING already biases toward the
        # informative mid-trajectory, so the loss weight should be mild — NOT the
        # 1/(1-t)^2 blow-up (that AMPLIFIES high-t and creates the spikes we saw).
        # Use a gentle SNR-style weight that DOWN-weights the noisy end: (1-t).
        t_w = (1.0 - t).clamp(min=0.1, max=1.0)

        model_in = torch.cat([xt.to(dtype), img_lat], dim=1)
        ids_in = torch.cat([latent_ids, img_ids], dim=1)
        timestep = t.expand(1).to(dtype)
        v_pred = pipe.transformer(hidden_states=model_in, timestep=timestep, guidance=None,
                                  encoder_hidden_states=pe, txt_ids=text_ids, img_ids=ids_in,
                                  return_dict=False)[0]
        v_pred = v_pred[:, : xt.size(1)].float()               # (1, 1024, 128)

        # --- FIX 2: region-weighted per-token MSE ---
        se = (v_pred - v_target) ** 2                          # (1,1024,128)
        rw = region_weight_map(ex["start"], ex["target"], args.region_weight, device)  # (1024,)
        per_token = se.mean(-1)[0] * rw                        # (1024,)
        loss = (per_token.mean() * t_w).squeeze()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        lval = float(loss.detach())
        raw_mse = float(se.mean().detach())
        log.append(dict(step=step, loss=lval, raw_mse=raw_mse, t=float(t)))
        if ml:
            import mlflow
            mlflow.log_metrics({"loss": lval, "raw_mse": raw_mse, "t": float(t),
                                "t_weight": float(t_w)}, step=step)
        if step % 20 == 0 or step == args.steps - 1:
            med = float(np.median([r["raw_mse"] for r in log[-30:]]))
            print(f"[{step:4d}/{args.steps}] loss={lval:.4f} raw_mse={raw_mse:.4f} (med30={med:.4f})", flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        if args.save_every and step and step % args.save_every == 0:
            pipe.transformer.save_lora_adapter(str(out / f"ckpt_{step}"))

    pipe.transformer.save_lora_adapter(str(out / "lora_final"))
    if ml:
        import mlflow
        mlflow.end_run()
    print(f"done -> {out}/lora_final")


if __name__ == "__main__":
    main()
