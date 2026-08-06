"""Shared latent-world-model utilities: FLUX VAE encode/decode + a latent-space
DIAMOND denoiser factory. The denoiser runs in the VAE's 128x128x32 latent space
(1024px frames, 8x downscale) so attention is cheap AND output is crisp — the
fix for the pixel-space blur/pixelation.

Latents are normalized to ~unit std (LATENT_STD) so DIAMOND's EDM preconditioning
(sigma_data=0.5) behaves; we un-normalize before VAE.decode.
"""
import sys
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

import torch.nn.functional as F
from models.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

ACTIONS = {"up": 0, "down": 1, "left": 2, "right": 3}


class ChangeWeightedDenoiser(Denoiser):
    """DIAMOND's Denoiser, unchanged EDM math — we ONLY override the loss reduction
    in forward() to upweight the region that actually changes between frames.

    Motivation: per-transition |next-cur| is tiny (~1% of the latent), so plain MSE
    is dominated by the static 99% and the model can leave the moved region blurry
    at near-zero loss. We weight each latent element by (1 + K*|next-cur|) so
    predicting the move is what drives the gradient. Everything else (noise,
    preconditioning, targets) is inherited verbatim from DIAMOND.
    """
    change_weight: float = 20.0

    def forward(self, batch):
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n
        all_obs = batch.obs.clone()
        loss = 0
        for i in range(seq_length):
            obs = all_obs[:, i:n + i]
            next_obs = all_obs[:, n + i]
            act = batch.act[:, i:n + i]
            mask = batch.mask_padding[:, n + i]
            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)
            cs = self.compute_conditioners(sigma)
            model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out

            # --- the ONLY change vs DIAMOND: change-weighted MSE ---
            cur = obs[:, -c:]                                    # last conditioning frame
            w_change = 1.0 + self.change_weight * (next_obs - cur).abs()
            se = (model_output - target) ** 2 * w_change
            loss += se[mask].mean()

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs[:, n + i] = denoised
        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}


class PerceptualDenoiser(ChangeWeightedDenoiser):
    """Change-weighted EDM loss + a NOISE-GATED LPIPS perceptual term (papers: add
    LPIPS only at LOW-noise timesteps -> sharper textures, no high-noise instability).

    The perceptual term decodes the predicted clean latent through the (frozen) VAE
    and compares to the real next frame in RGB with LPIPS. Set net.vae, net.lpips,
    net.decode_fn before training. lpips_weight & sigma_gate control it.
    """
    lpips_weight: float = 0.5
    sigma_gate: float = 1.0      # apply LPIPS only when sigma < gate (low noise)
    perc_max: int = 1            # max samples to VAE-decode+backprop for LPIPS (memory)
    gt_next_rgb = None           # (B,3,H,W) real next frame in [-1,1], set per batch

    def set_perceptual(self, vae, lpips_fn, decode_fn):
        # store as plain attrs bypassing nn.Module submodule registration (we do NOT
        # want VAE/LPIPS in the denoiser's parameters); a class attr `lpips=None`
        # would shadow assignment, so use object.__setattr__ into __dict__.
        object.__setattr__(self, "_vae", vae)
        object.__setattr__(self, "_lpips", lpips_fn)
        object.__setattr__(self, "_decode_fn", decode_fn)

    def forward(self, batch):
        n = self.cfg.inner_model.num_steps_conditioning
        obs = batch.obs[:, :n].reshape(batch.obs.size(0), -1, *batch.obs.shape[-2:])
        next_obs = batch.obs[:, n]
        act = batch.act[:, :n]
        b, c = next_obs.size(0), next_obs.size(1)

        sigma = self.sample_sigma_training(b, self.device)
        noisy = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy, obs, act, cs)
        target = (next_obs - cs.c_skip * noisy) / cs.c_out
        cur = obs[:, -c:]
        w = 1.0 + self.change_weight * (next_obs - cur).abs()
        mse = ((model_output - target) ** 2 * w).mean()

        logs = {"loss_denoising": mse.detach()}
        perc = torch.zeros((), device=self.device)
        # noise-gate: only add LPIPS on the low-noise samples in the batch
        low = (sigma.flatten() < self.sigma_gate)
        if getattr(self, "_lpips", None) is not None and low.any():
            # predicted CLEAN latent (differentiable path, not the quantized wrap)
            pred_clean = cs.c_skip * noisy + cs.c_out * model_output
            idx = low.nonzero(as_tuple=True)[0]
            # CAP the differentiable VAE-decode backward to at most `perc_max` samples
            # (decoding 1024px through the VAE for backprop is the memory bottleneck).
            if idx.numel() > self.perc_max:
                idx = idx[:self.perc_max]
            pred_rgb = self._decode_fn(pred_clean[idx])         # (k,3,H,W) in [-1,1]
            gt_rgb = self.gt_next_rgb[idx]
            perc = self._lpips(pred_rgb, gt_rgb).mean()
            logs["loss_lpips"] = perc.detach()
        loss = mse + self.lpips_weight * perc
        return loss, logs


# Swappable VAE backend. Qwen-Image VAE has a 16-ch latent (vs FLUX's 32) —
# HALF the prediction burden for the denoiser, which is data-starved (9 frames),
# so it averages less -> sharper predicted latents. Both round-trip near-lossless.
VAE_BACKEND = "qwen"          # "qwen" (16ch) or "flux" (32ch)

_FLUX = "black-forest-labs/FLUX.2-klein-base-4B"
_FLUX_STD = 1.76              # single-scalar norm (FLUX ships no per-channel stats)

# LATENT_CH is set by load_vae() based on backend; default matches Qwen.
LATENT_CH = 16 if VAE_BACKEND == "qwen" else 32


def load_vae(device="cuda", dtype=torch.bfloat16, backend=None):
    """Returns a VAE wrapper with .encode/.decode and .latent_ch / normalization baked in.
    Loads the BUNDLED Qwen VAE from ./model/qwen_vae (self-contained, no download)."""
    backend = backend or VAE_BACKEND
    if backend == "qwen":
        import os
        from pathlib import Path
        from diffusers import AutoencoderKLQwenImage
        # bundled VAE path (next to this file); override with QWEN_VAE_PATH env
        local = os.environ.get("QWEN_VAE_PATH",
                               str(Path(__file__).resolve().parent / "qwen_vae"))
        src = local if Path(local).exists() else "Qwen/Qwen-Image-Edit"
        kw = {} if Path(local).exists() else {"subfolder": "vae"}
        vae = AutoencoderKLQwenImage.from_pretrained(
            src, torch_dtype=dtype, **kw).to(device).eval()
        # per-channel normalization (Qwen ships these) -> ~unit std per channel
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        vae._norm = (mean, std, True)      # (mean, std, is_video_vae)
        vae.latent_ch = 16
    else:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(_FLUX, torch_dtype=dtype)
        vae = pipe.vae.to(device).eval(); del pipe
        vae._norm = (0.0, _FLUX_STD, False)
        vae.latent_ch = 32
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, pil_img, res, device, dtype=torch.bfloat16):
    """PIL -> normalized latent (1, C, res/8, res/8) float32."""
    im = pil_img.convert("RGB").resize((res, res), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im).copy()).float().div(127.5).sub(1)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device, dtype)
    mean, std, is_video = vae._norm
    if is_video:
        x = x.unsqueeze(2)                 # (B,C,T=1,H,W)
    enc = vae.encode(x)
    lat = enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.latents
    lat = lat.float()
    if is_video:
        lat = (lat - mean) / std           # (1,16,1,128,128)
        lat = lat.squeeze(2)               # -> (1,16,128,128)
    else:
        lat = lat / std
    return lat


@torch.no_grad()
def decode(vae, lat_norm, dtype=torch.bfloat16):
    """normalized latent (1,C,128,128) -> PIL 1024px."""
    mean, std, is_video = vae._norm
    if is_video:
        lat = lat_norm.unsqueeze(2).float() * std + mean   # (1,16,1,128,128)
        lat = lat.to(dtype)
    else:
        lat = (lat_norm * std).to(dtype)
    img = vae.decode(lat).sample
    if img.dim() == 5:
        img = img.squeeze(2)
    a = img.float().clamp(-1, 1).add(1).div(2).mul(255).byte()[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def decode_tensor(vae, lat_norm, out_size=256):
    """DIFFERENTIABLE decode: normalized latent (B,C,128,128) -> RGB tensor (B,3,S,S)
    in [-1,1], downsampled to out_size for LPIPS (keeps memory bounded; gradients
    flow through the frozen VAE back to the denoiser). No no_grad here."""
    mean, std, is_video = vae._norm
    dtype = next(vae.parameters()).dtype
    if is_video:
        lat = lat_norm.unsqueeze(2) * std + mean
        lat = lat.to(dtype)
        img = vae.decode(lat).sample
        img = img.squeeze(2) if img.dim() == 5 else img
    else:
        img = vae.decode((lat_norm * std).to(dtype)).sample
    img = img.float().clamp(-1, 1)
    if out_size and img.shape[-1] != out_size:
        img = torch.nn.functional.interpolate(img, size=out_size, mode="bilinear",
                                              align_corners=False)
    return img


def make_latent_denoiser(device="cuda", num_actions=len(ACTIONS),
                         change_weighted=False, change_weight=20.0, cond_channels=256,
                         sigma_data=0.55, sigma_offset_noise=0.1, channels=None):
    """DIAMOND denoiser operating on latents (128x128 grid from 1024px).

    CRITICAL: attention cost is O(tokens^2). At the top 128x128 level that is
    16384^2 -> ~1.5 TB, which OOM-kills the whole (unified-memory) machine. So we
    use 4 U-Net levels (128->64->32->16) and enable self-attention ONLY at the two
    DEEPEST/smallest levels (32x32 and 16x16), exactly like Stable Diffusion /
    GameNGen latent U-Nets. The bottleneck block also runs attention at 16x16.

    change_weighted=True builds ChangeWeightedDenoiser (DIAMOND's EDM math verbatim,
    only the loss reduction upweights the moved region). cond_channels controls the
    STRENGTH of action conditioning — DIAMOND concats the action embedding into the
    cond vector fed to every UNet block, so a larger cond_channels = a stronger,
    harder-to-ignore action signal.
    """
    inner = InnerModelConfig(
        img_channels=LATENT_CH,                 # 16 (Qwen) / 32 (FLUX) latent ch
        num_steps_conditioning=1,
        cond_channels=cond_channels,            # bigger -> stronger action conditioning
        depths=[2, 2, 2, 2],                    # 4 levels: 128 -> 64 -> 32 -> 16
        channels=channels or [64, 128, 256, 512],  # widen for capacity (Axis 3)
        attn_depths=[False, False, True, True], # attention ONLY at 32x32 and 16x16
        num_actions=num_actions,
    )
    # DATA-DERIVED EDM config. sigma_data = measured latent std (pooled 0.55, or
    # PER-GAME for specialist models, Axis 1). sigma_offset_noise small because
    # per-transition |next-cur| delta is tiny.
    cfg = DenoiserConfig(inner_model=inner, sigma_data=sigma_data,
                         sigma_offset_noise=sigma_offset_noise)
    cls = ChangeWeightedDenoiser if change_weighted else Denoiser
    net = cls(cfg).to(device)
    if change_weighted:
        net.change_weight = change_weight
    net.setup_training(SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20))
    return net


def make_perceptual_denoiser(device="cuda", num_actions=len(ACTIONS), change_weight=20.0,
                             cond_channels=512, sigma_data=0.55, channels=None,
                             lpips_weight=0.5, sigma_gate=1.0):
    """PerceptualDenoiser: change-weighted MSE + noise-gated LPIPS. Same DIAMOND
    config as make_latent_denoiser, but the perceptual class. Attach net.vae/lpips
    /decode_fn/gt_next_rgb per batch in the trainer."""
    inner = InnerModelConfig(
        img_channels=LATENT_CH, num_steps_conditioning=1, cond_channels=cond_channels,
        depths=[2, 2, 2, 2], channels=channels or [64, 128, 256, 512],
        attn_depths=[False, False, True, True], num_actions=num_actions)
    cfg = DenoiserConfig(inner_model=inner, sigma_data=sigma_data, sigma_offset_noise=0.1)
    net = PerceptualDenoiser(cfg).to(device)
    net.change_weight = change_weight
    net.lpips_weight = lpips_weight
    net.sigma_gate = sigma_gate
    net.setup_training(SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20))
    return net
