"""Supervised flow-matching LoRA fine-tune of FLUX.2-klein-base-4B for the
spatial-move task. Teaches PLACEMENT the base model lacks.

Task: (input board with red triangle at start) + prompt -> (output board with red
triangle at target). Standard rectified-flow SFT:

    x0   = VAE-encode(target image)                 (clean latent)
    noise~ N(0,1)
    t    ~ U(0,1)
    x_t  = (1-t)*x0 + t*noise                        (flow interpolation)
    v*   = noise - x0                                (target velocity)
    v^   = transformer(x_t | prompt, input-image-latents, t)
    loss = MSE(v^, v*)                               (only LoRA params trained)

Two prompt conditions selected by --prompt-key {prompt_nl, prompt_coords} → two
separate LoRAs (natural-language vs coordinates).

    python scripts/train_lora_sft.py --prompt-key prompt_coords --steps 400 --out runs/lora_coords
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
from spatial_intel.rl.gligen import GroundingEncoder, append_grounding

MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
SIDE = 32                                   # generated-image patch grid side (512px)
spec = GridSpec()


def load_pipe(lora_rank, lora_alpha, device="cuda"):
    from spatial_intel.rl.policy import _load_hf_token_from_env_file
    _load_hf_token_from_env_file()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    for p in pipe.transformer.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig
    cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
                     target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj",
                                     "add_k_proj", "add_v_proj"])
    pipe.transformer.add_adapter(cfg)
    return pipe


def img_to_tensor(path, size, device, dtype):
    im = Image.open(path).convert("RGB").resize((size, size))
    a = torch.from_numpy(np.asarray(im)).float() / 127.5 - 1.0     # [-1,1]
    return a.permute(2, 0, 1).unsqueeze(0).to(device, dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lora")
    ap.add_argument("--prompt-key", choices=["prompt_nl", "prompt_coords"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-logdir", default="runs/tb", help="TensorBoard log dir (shared)")
    # flow-matching timestep sampling (SD3/FLUX standard, replaces uniform)
    ap.add_argument("--logit-mean", type=float, default=0.0)
    ap.add_argument("--logit-std", type=float, default=1.0)
    ap.add_argument("--flow-shift", type=float, default=3.16,
                    help="discrete flow shift (kohya-recommended ~3.16)")
    ap.add_argument("--align-lambda", type=float, default=0.0,
                    help="weight of the attention-alignment aux loss (0 = plain SFT)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)

    # TensorBoard: local live dashboard. All runs log under one dir so you can
    # compare NL vs coords side by side:  tensorboard --logdir runs/tb
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(Path(args.tb_logdir) / Path(args.out).name))
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    train = manifest["train"]
    print(f"train examples: {len(train)} | prompt key: {args.prompt_key}")

    pipe = load_pipe(args.lora_rank, args.lora_alpha, device)
    dtype = pipe.transformer.dtype
    # GLIGEN grounding encoder: Fourier(target x,y) -> grounding token (trained jointly)
    edim = pipe.transformer.config["joint_attention_dim"]     # 7680
    grounder = GroundingEncoder(embed_dim=edim).to(device, dtype)
    params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    params += list(grounder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"trainable LoRA tensors: {len(params)} (incl. grounding encoder)")

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data)
    log = []
    for step in range(args.steps):
        ex = train[int(rng.integers(len(train)))]
        prompt = ex[args.prompt_key]
        # encode prompt (frozen text encoder)
        with torch.no_grad():
            pe, text_ids = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1)
            # clean target latent x0
            tgt = img_to_tensor(data_dir / ex["target_img"], args.img_size, device, dtype)
            x0 = pipe._encode_vae_image(tgt, torch.Generator(device).manual_seed(step))
            x0 = pipe._pack_latents(x0)                        # (1, 1024, 128)
            # conditioning: the INPUT board latents + ids
            inp = img_to_tensor(data_dir / ex["input"], args.img_size, device, dtype)
            img_lat, img_ids = pipe.prepare_image_latents([inp], 1,
                                torch.Generator(device).manual_seed(step), device, dtype)
            # latent ids for the generated stream
            num_ch = pipe.transformer.config.in_channels // 4
            _, latent_ids = pipe.prepare_latents(1, num_ch, args.img_size, args.img_size,
                                dtype, device, torch.Generator(device).manual_seed(step))

        x0 = x0.float()
        noise = torch.randn_like(x0)
        # LOGIT-NORMAL timestep sampling (SD3/FLUX standard) instead of uniform:
        # concentrates mass in the mid-trajectory where training signal is richest,
        # then apply a resolution/shift factor. Uniform wastes steps on low-info t.
        u = torch.randn(1, device=device) * args.logit_std + args.logit_mean
        t = torch.sigmoid(u)
        shift = args.flow_shift
        t = (shift * t) / (1 + (shift - 1) * t)               # flow-matching shift
        xt = (1 - t) * x0 + t * noise
        v_target = noise - x0                                  # rectified-flow target

        # GLIGEN: build grounding token from target (x,y) and APPEND to text stream
        tx, ty = ex["target"]
        target_xy = torch.tensor([[tx / spec.max_topleft, ty / spec.max_topleft]],
                                 device=device, dtype=torch.float32)
        gtok = grounder(target_xy)                             # (1,1,edim) with grad
        pe_g, tids_g = append_grounding(pe, text_ids, gtok)

        # transformer forward (LoRA + grounding active) — concat gen + conditioning tokens
        model_in = torch.cat([xt.to(dtype), img_lat], dim=1)
        ids_in = torch.cat([latent_ids, img_ids], dim=1)
        timestep = t.expand(1).to(dtype)
        v_pred = pipe.transformer(hidden_states=model_in, timestep=timestep, guidance=None,
                                  encoder_hidden_states=pe_g, txt_ids=tids_g, img_ids=ids_in,
                                  return_dict=False)[0]
        v_pred = v_pred[:, : xt.size(1)].float()               # keep generated stream

        mse = torch.mean((v_pred - v_target) ** 2)
        loss = mse
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        lval = float(loss.detach()); mval = float(mse.detach())
        gate = float(torch.tanh(grounder.gate).detach())
        log.append(dict(step=step, loss=lval, mse=mval, gate=gate))
        med = float(np.median([r["mse"] for r in log[-30:]]))
        tb.add_scalar("mse", mval, step)
        tb.add_scalar("gate", gate, step)             # grounding gate opening 0 -> nonzero
        tb.add_scalar("mse_median30", med, step)
        tb.flush()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"[{step:4d}/{args.steps}] mse={mval:.4f} gate={gate:.3f}", flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        if args.save_every and step and step % args.save_every == 0:
            pipe.transformer.save_lora_adapter(str(out / f"ckpt_{step}"))
            torch.save(grounder.state_dict(), out / f"ckpt_{step}_grounder.pt")

    pipe.transformer.save_lora_adapter(str(out / "lora_final"))
    torch.save(grounder.state_dict(), out / "grounder.pt")
    tb.close()
    print(f"done -> {out}/lora_final (+ grounder.pt)")


if __name__ == "__main__":
    main()
