"""Pluggable image-EDIT model backends for the world-model quality benchmark.

Each editor implements: load() and edit(frame, prompt, **kw) -> PIL.Image.
The benchmark feeds (current frame + an instruction like "move the red S badge one
cell to the right") and measures speed + whether S landed on the right cell + VLM
quality. This is the FULL-diffusion path (the model regenerates the frame), in
contrast to the deterministic cut-paste baseline.

Backends are added as models are verified downloadable. API-only ones (fal) get a
separate FalEditor. Keep each backend self-contained so a missing model doesn't
break the others.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


@dataclass
class EditResult:
    image: Image.Image
    seconds: float
    ok: bool = True
    error: str = ""


class Editor:
    name = "base"
    def load(self): ...
    def edit(self, frame: Image.Image, prompt: str, seed: int = 0) -> EditResult:
        raise NotImplementedError


# ---- SD / SDXL turbo img2img (local, cached) ----------------------------
class SDTurboImg2Img(Editor):
    name = "sd-turbo-i2i"
    def __init__(self, repo="stabilityai/sd-turbo", size=512, steps=2, strength=0.6):
        self.repo, self.size, self.steps, self.strength = repo, size, steps, strength
        self._p = None
    def load(self):
        if self._p: return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        from diffusers import AutoPipelineForImage2Image
        self._p = AutoPipelineForImage2Image.from_pretrained(
            self.repo, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self._p.set_progress_bar_config(disable=True)
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(prompt=prompt, image=src, num_inference_steps=self.steps,
                      strength=self.strength, guidance_scale=0.0, generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


# ---- InstructPix2Pix (local, the classic instruction editor) ------------
class InstructPix2Pix(Editor):
    name = "instruct-pix2pix"
    def __init__(self, repo="timbrooks/instruct-pix2pix", size=512, steps=10):
        self.repo, self.size, self.steps = repo, size, steps
        self._p = None
    def load(self):
        if self._p: return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self._p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.repo, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self._p.set_progress_bar_config(disable=True)
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(prompt, image=src, num_inference_steps=self.steps,
                      image_guidance_scale=1.5, guidance_scale=7.0, generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


# ---- FLUX.2-klein edit (local, cached; the model we already used) -------
class Flux2KleinEdit(Editor):
    name = "flux2-klein-4b"
    def __init__(self, repo="black-forest-labs/FLUX.2-klein-4B", size=512, steps=4):
        self.repo, self.size, self.steps = repo, size, steps
        self._p = None
    def load(self):
        if self._p: return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        from diffusers import Flux2KleinPipeline
        self._p = Flux2KleinPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16).to("cuda")
        self._p.set_progress_bar_config(disable=True)
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(image=src, prompt=prompt, num_inference_steps=self.steps,
                      guidance_scale=1.0, height=self.size, width=self.size, generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


# ---- FLUX.1-Kontext-dev (instruction edit, best frame->frame consistency) ----
class FluxKontextEdit(Editor):
    name = "flux-kontext-dev"
    def __init__(self, repo="AlekseyCalvin/flux1-kontext-dev_fp8_e4m3fn_diffusers", size=512, steps=20):
        self.repo, self.size, self.steps = repo, size, steps      # fp8 diffusers build (~27GB, ungated)
        self._p = None
    def load(self):
        if self._p: return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        from diffusers import FluxKontextPipeline
        self._p = FluxKontextPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16).to("cuda")
        self._p.set_progress_bar_config(disable=True)
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(image=src, prompt=prompt, num_inference_steps=self.steps,
                      guidance_scale=2.5, height=self.size, width=self.size, generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


# ---- Qwen-Image-Edit (purpose-built instruction edit, 20B) --------------
class QwenImageEdit(Editor):
    name = "qwen-image-edit"
    def __init__(self, repo="Qwen/Qwen-Image-Edit", size=512, steps=8):
        self.repo, self.size, self.steps = repo, size, steps      # full unquantized 20B (~54GB, fits 120GB on GPU)
        self._p = None
    def load(self):
        if self._p:
            return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        import diffusers
        # full model: plain load, fully on GPU (faster than 4-bit+offload which was 56s/edit)
        Pipe = getattr(diffusers, "QwenImageEditPlusPipeline", None) or diffusers.QwenImageEditPipeline
        self._p = Pipe.from_pretrained(self.repo, torch_dtype=torch.bfloat16).to("cuda")
        self._p.set_progress_bar_config(disable=True)
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(image=src, prompt=prompt, num_inference_steps=self.steps,
                      true_cfg_scale=4.0, negative_prompt=" ", generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


class QwenImageEditFull(Editor):
    name = "qwen-image-edit-full"
    def __init__(self, repo="Qwen/Qwen-Image-Edit", size=512, steps=50):
        self.repo, self.size, self.steps = repo, size, steps
        self._p = None
    def load(self):
        if self._p:
            return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        from diffusers import QwenImageEditPipeline
        self._p = QwenImageEditPipeline.from_pretrained(
            self.repo, torch_dtype=torch.bfloat16)
        self._p.enable_model_cpu_offload()
        self._p.set_progress_bar_config(disable=True)
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(image=src, prompt=prompt, num_inference_steps=self.steps,
                      true_cfg_scale=4.0, negative_prompt=" ", generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


# ---- FLUX.2-dev NVFP4 (32B flagship edit, Blackwell-native 4-bit) --------
class Flux2DevNVFP4(Editor):
    name = "flux2-dev-nvfp4"
    def __init__(self, repo="black-forest-labs/FLUX.2-dev-NVFP4", size=1024, steps=28):
        self.repo, self.size, self.steps = repo, size, steps
        self._p = None
    def load(self):
        if self._p: return
        from spatial_intel.rl.policy import _load_hf_token_from_env_file
        _load_hf_token_from_env_file()
        from diffusers import DiffusionPipeline
        # loads per the HF model card: NVFP4 transformer + base config/encoder resolved by HF
        self._p = DiffusionPipeline.from_pretrained(
            self.repo, dtype=torch.bfloat16, device_map="cuda")
        try: self._p.set_progress_bar_config(disable=True)
        except Exception: pass
    def edit(self, frame, prompt, seed=0):
        self.load()
        src = frame.resize((self.size, self.size))
        g = torch.Generator("cuda").manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        out = self._p(image=src, prompt=prompt, num_inference_steps=self.steps,
                      generator=g).images[0]
        torch.cuda.synchronize()
        return EditResult(out.resize(frame.size), time.time() - t0)


# registry — extended as models get verified/downloaded
REGISTRY = {
    "sd-turbo-i2i": SDTurboImg2Img,
    "instruct-pix2pix": InstructPix2Pix,
    "flux2-klein-4b": Flux2KleinEdit,
    "flux-kontext-dev": FluxKontextEdit,
    "qwen-image-edit": QwenImageEdit,
    "qwen-image-edit-full": QwenImageEditFull,
    "flux2-dev-nvfp4": Flux2DevNVFP4,
}
