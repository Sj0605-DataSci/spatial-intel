"""FLUX.2-klein-4B image-editing agent (scaffold).

The agent treats the board as an image-editing problem: given the current board
(triangle at the start block) and a textual instruction + feedback, it asks
FLUX to *edit* the image so the triangle lands on the commanded block, then we
read back where the triangle ended up and reward it.

This is intentionally a thin, lazy-loaded wrapper:
  - diffusers / the 4B weights are only imported/downloaded when you actually
    instantiate FluxAgent (so the rest of the project runs with no GPU download).
  - the pipeline class name is resolved defensively (the model card lists
    `Flux2KleinPipeline`; we fall back to `Flux2Pipeline` / auto img2img) so a
    minor naming drift in diffusers doesn't break the scaffold.

Requires (into 3dgit/.venv):
    pip install "git+https://github.com/huggingface/diffusers.git" \
                transformers accelerate safetensors sentencepiece
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..grid import GridSpec, from_array, to_array
from .base import Agent

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def _resolve_pipeline_cls():
    import diffusers
    for cls_name in ("Flux2KleinPipeline", "Flux2Pipeline", "FluxKontextPipeline"):
        cls = getattr(diffusers, cls_name, None)
        if cls is not None:
            return cls, cls_name
    # last resort: auto pipeline for image-to-image
    from diffusers import AutoPipelineForImage2Image
    return AutoPipelineForImage2Image, "AutoPipelineForImage2Image"


class FluxAgent(Agent):
    name = "flux2-klein-4b"

    def __init__(self, spec: GridSpec, *,
                 model_id: str = MODEL_ID,
                 device: str = "cuda",
                 num_inference_steps: int = 4,
                 guidance_scale: float = 1.0,
                 work_size: int = 512,
                 cpu_offload: bool = False,
                 seed: int = 0):
        self.spec = spec
        self.action_mode = "image"  # FLUX always produces images
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.work_size = work_size  # multiple of 16; FLUX render resolution
        self.seed = seed
        self._pipe = None
        self._model_id = model_id
        self._cpu_offload = cpu_offload

    # -- lazy model load -------------------------------------------------
    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        cls, name = _resolve_pipeline_cls()
        kwargs = dict(torch_dtype=torch.bfloat16)
        pipe = cls.from_pretrained(self._model_id, **kwargs)
        if self._cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        try:  # speed up repeat calls on Blackwell if torch.compile is available
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        self._pipe = pipe
        self._pipe_name = name

    # -- prompt construction --------------------------------------------
    def _build_prompt(self, info: dict) -> str:
        instr = info.get("instruction", "")
        fb = info.get("feedback")
        base = (
            "This is a square grid with numbered columns and rows and a solid "
            "red triangle occupying one square block. " + instr +
            " Keep the grid, axes and coordinate labels identical; move ONLY the "
            "red triangle to its new grid block; output a single solid red "
            "triangle snapped to the grid."
        )
        if fb:
            base += " Correction from the previous attempt: " + fb
        return base

    # -- act -------------------------------------------------------------
    def act(self, obs: dict, info: dict) -> np.ndarray:
        self._ensure()
        import torch
        src = from_array(obs["image"]).resize((self.work_size, self.work_size))
        prompt = self._build_prompt(info)
        gen = torch.Generator(device=self.device).manual_seed(self.seed + info.get("t", 0))
        result = self._pipe(
            image=src,
            prompt=prompt,
            height=self.work_size,
            width=self.work_size,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=gen,
        )
        out = result.images[0]
        # detection runs at the env's native resolution, so resize back.
        out = out.resize((self.spec.img_w, self.spec.img_h))
        return to_array(out)
