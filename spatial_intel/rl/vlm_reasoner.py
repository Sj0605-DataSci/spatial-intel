"""Small-VLM reasoner for the conditioning ablation.

Given the input board image + the move instruction, a small VLM (Qwen2.5-VL-3B or
ViGoRL-3b-Spatial) produces a short grounded reasoning string about WHERE the
triangle should end up. That text becomes the extra conditioning fed to the
diffusion model in arm 4 of the ablation.

Both models share the Qwen2.5-VL architecture/processor, so one wrapper serves
both — only the checkpoint id differs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VIGORL_ID = "gsarch/ViGoRL-3b-Spatial"


@dataclass
class VLMConfig:
    model_id: str = QWEN_ID
    max_new_tokens: int = 128
    device: str = "cuda"


class VLMReasoner:
    def __init__(self, cfg: VLMConfig | None = None):
        self.cfg = cfg or VLMConfig()
        self._model = None
        self._proc = None

    def load(self):
        if self._model is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.model_id, torch_dtype=torch.bfloat16, device_map=self.cfg.device)
        self._model.eval()
        # ViGoRL ships its own processor config; fall back to the Qwen base if absent
        try:
            self._proc = AutoProcessor.from_pretrained(self.cfg.model_id)
        except Exception:
            self._proc = AutoProcessor.from_pretrained(QWEN_ID)

    @torch.no_grad()
    def reason(self, image: Image.Image, instruction: str) -> str:
        """Return a short grounded reasoning string for the move command."""
        self.load()
        prompt = (
            "You are looking at a square grid with numbered columns (x) and rows "
            "(y), origin at top-left, and one solid red triangle in a cell. "
            f"Instruction: {instruction} "
            "Reason briefly about which exact grid cell the red triangle should "
            "end up in, then state the target as 'TARGET=(col,row)'. Be concise."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._proc.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._proc(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt").to(self.cfg.device)
        out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens,
                                   do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._proc.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0].strip()
