"""FluxPolicy — FLUX.2-klein-base-4B + LoRA as an RL policy.

Wraps the diffusers pipeline but replaces the opaque `pipe(...)` call with a
manual SDE denoising loop (see `sde.py`) that RECORDS, per step:
  - the latent x_t going in,
  - the velocity v the transformer predicted,
  - the sampled x_{t-1},
  - log pi(x_{t-1} | x_t),
  - the (sigma, sigma_next) pair,
so the rollout is differentiable-by-replay: the REINFORCE loss (reinforce.py)
replays the recorded latents through the transformer to recompute the log-probs
WITH gradient, then weights them by (reward - baseline).

We train on the BASE model (50-step, the correct LoRA training target); the
adapter loads on the distilled 4-step model for inference afterwards.

This is a scaffold: `MODEL_ID`, the transformer-call signature, and latent
(un)packing are the parts most likely to need a tweak against the exact diffusers
build — they are isolated in `_predict_velocity` and `_decode` for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .sde import sde_step

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"


def _load_hf_token_from_env_file() -> None:
    """If a (git-ignored) .env holds an HF token, export it as HF_TOKEN.

    huggingface_hub reads HF_TOKEN from the environment automatically, so this
    keeps the token out of source. The FLUX.2 weights are public/ungated, so a
    token is only needed if you later PUSH trained LoRAs to the Hub. We never log
    or return the value.
    """
    import os
    if os.environ.get("HF_TOKEN"):
        return
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip().lower() in ("hf_token", "huggingface_token", "hf_hub_token"):
            os.environ["HF_TOKEN"] = val.strip().strip('"').strip("'")
            break


@dataclass
class Trajectory:
    """One recorded denoising rollout (one image)."""
    latents: List[torch.Tensor] = field(default_factory=list)      # x_t per step
    next_latents: List[torch.Tensor] = field(default_factory=list)  # x_{t-1}
    velocities: List[torch.Tensor] = field(default_factory=list)    # v_t (detached)
    log_probs: List[torch.Tensor] = field(default_factory=list)     # old log pi
    sigmas: List[float] = field(default_factory=list)
    sigmas_next: List[float] = field(default_factory=list)
    timesteps_raw: List[float] = field(default_factory=list)  # raw scheduler t (0-1000)
    image: Optional[np.ndarray] = None    # decoded RGB uint8, for scoring
    img_ids: Optional[torch.Tensor] = None   # latent image ids (constant per rollout)
    prompt: str = ""


class FluxPolicy:
    def __init__(self, *, model_id: str = MODEL_ID, device: str = "cuda",
                 num_steps: int = 20, guidance_scale: float = 1.0,
                 eta: float = 0.1, work_size: int = 512,
                 lora_rank: int = 16, lora_alpha: int = 16,
                 dtype: torch.dtype = torch.bfloat16):
        self.model_id = model_id
        self.device = device
        self.num_steps = num_steps          # Flow-GRPO Denoising Reduction: few train steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.work_size = work_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.dtype = dtype
        self._pipe = None

    # -- setup -----------------------------------------------------------
    def load(self):
        if self._pipe is not None:
            return
        _load_hf_token_from_env_file()
        import diffusers
        cls = getattr(diffusers, "Flux2KleinPipeline", None) or diffusers.Flux2Pipeline
        pipe = cls.from_pretrained(self.model_id, torch_dtype=self.dtype).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        # Freeze everything; LoRA adapters on the transformer are the only trainables.
        for p in pipe.transformer.parameters():
            p.requires_grad_(False)
        from peft import LoraConfig
        # target the attention projections — the standard, safe choice
        cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        pipe.transformer.add_adapter(cfg)
        self._pipe = pipe

    @property
    def trainable_parameters(self):
        return [p for p in self._pipe.transformer.parameters() if p.requires_grad]

    # -- prompt conditioning cache (so the REINFORCE replay doesn't re-encode) --
    def _cond_for(self, prompt: str):
        """Return (prompt_embeds, text_ids) for `prompt`, cached.

        Encoding is under no_grad — the text encoders are frozen; only the
        transformer LoRA is trained, so the conditioning is a constant input.
        FLUX.2's encode_prompt returns (prompt_embeds, text_ids); there is no
        pooled projection (guidance_embeds is False on klein).
        """
        cache = getattr(self, "_cond_cache", None)
        if cache is None:
            cache = self._cond_cache = {}
        if prompt not in cache:
            with torch.no_grad():
                pe, text_ids = self._pipe.encode_prompt(
                    prompt=prompt, device=self.device, num_images_per_prompt=1)
            cache[prompt] = (pe, text_ids)
        return cache[prompt]

    # -- the model seam: call the transformer exactly as the pipeline does ----
    def _predict_velocity(self, latents, timestep, prompt_embeds, text_ids, img_ids):
        """Velocity prediction v. `timestep` is the raw scheduler t (0-1000);
        the transformer wants t/1000, matching Flux2KleinPipeline.__call__."""
        out = self._pipe.transformer(
            hidden_states=latents.to(self._pipe.transformer.dtype),
            timestep=timestep / 1000.0,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        return out[:, : latents.size(1)]      # drop any ref tokens, keep image seq

    def _decode(self, latents, img_ids) -> np.ndarray:
        """VAE-decode packed latents to an HxWx3 uint8 array.

        FLUX.2's VAE denormalises with a batch-norm (running_mean/var), not a
        scaling/shift factor, and unpacks via _unpack_latents_with_ids — copied
        from Flux2KleinPipeline.__call__.
        """
        pipe = self._pipe
        lh = 2 * (int(self.work_size) // (pipe.vae_scale_factor * 2))
        lw = 2 * (int(self.work_size) // (pipe.vae_scale_factor * 2))
        lat = pipe._unpack_latents_with_ids(latents, img_ids, lh // 2, lw // 2)
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            lat.device, lat.dtype)
        lat = lat * std + mean
        lat = pipe._unpatchify_latents(lat)      # 128ch packed -> 32ch for the VAE
        img = pipe.vae.decode(lat.to(self.dtype), return_dict=False)[0]
        img = pipe.image_processor.postprocess(img, output_type="np")[0]
        return (img * 255).round().astype(np.uint8)

    # -- rollout ---------------------------------------------------------
    @torch.no_grad()
    def rollout(self, prompt: str, *, src_image=None,
                generator: torch.Generator | None = None) -> Trajectory:
        """Sample ONE image via recorded SDE denoising."""
        self.load()
        pipe = self._pipe
        prompt_embeds, text_ids = self._cond_for(prompt)
        latents, img_ids = self._init_latents(generator)
        timesteps, sigmas = self._schedule(image_seq_len=latents.shape[1])

        traj = Trajectory(prompt=prompt)
        for i in range(self.num_steps):
            sigma = torch.tensor(sigmas[i], device=self.device, dtype=torch.float32)
            sigma_next = torch.tensor(sigmas[i + 1], device=self.device, dtype=torch.float32)
            t = timesteps[i].expand(latents.shape[0]).to(self.dtype)
            v = self._predict_velocity(latents, t, prompt_embeds, text_ids, img_ids).float()
            step = sde_step(latents.float(), v, sigma, sigma_next, self.eta,
                            generator=generator)
            traj.latents.append(latents.detach().cpu())
            traj.velocities.append(v.detach().cpu())
            traj.next_latents.append(step.x_next.detach().cpu())
            traj.log_probs.append(step.log_prob.detach().cpu())
            traj.sigmas.append(float(sigma)); traj.sigmas_next.append(float(sigma_next))
            traj.timesteps_raw = getattr(traj, "timesteps_raw", [])
            traj.timesteps_raw.append(float(timesteps[i]))
            latents = step.x_next.to(self.dtype)

        traj.image = self._decode(latents, img_ids)
        traj.img_ids = img_ids.detach().cpu()
        return traj

    # -- pipeline-helper wrappers (now using the REAL klein API) ---------
    def _schedule(self, image_seq_len: int):
        """Return (timesteps, sigmas) using klein's dynamic-shift schedule.

        FLUX.2 uses dynamic shifting, so set_timesteps needs `mu` computed from
        the image sequence length (compute_empirical_mu) — exactly as the
        pipeline's __call__ does. sigmas length is num_steps+1, runs 1 -> 0.
        """
        import numpy as np
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu, retrieve_timesteps)
        sigmas = np.linspace(1.0, 1.0 / self.num_steps, self.num_steps)
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.num_steps)
        timesteps, _ = retrieve_timesteps(
            self._pipe.scheduler, self.num_steps, self.device, sigmas=sigmas, mu=mu)
        sig = self._pipe.scheduler.sigmas.float().cpu().tolist()
        return timesteps, sig

    def _init_latents(self, generator):
        """Build the initial packed latent + image ids via the pipeline helper."""
        pipe = self._pipe
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=self.work_size,
            width=self.work_size,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        return latents, latent_ids
