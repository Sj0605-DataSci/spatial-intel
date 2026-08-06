"""VLM-as-judge for world-model frames. Scores a generated next-frame on:
  - action_correct : did S move to the expected cell? (VLM reads the board)
  - scene_intact   : is the rest of the scene unchanged / coherent (no corruption)?
  - quality        : overall image quality (1-5)

Pluggable judge backend — Qwen2.5-VL-3B first; InternVL / others drop in by
subclassing or swapping model_id. The judge is given the BEFORE and AFTER frames
plus the intended action, and returns structured scores.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch
from PIL import Image

QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass
class JudgeScores:
    action_correct: bool
    scene_intact: int      # 1-5
    quality: int           # 1-5
    raw: str


class VLMJudge:
    def __init__(self, model_id: str = QWEN_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._proc = None

    def load(self):
        if self._model is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device).eval()
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def judge(self, before: Image.Image, after: Image.Image, action: str,
              expected_cell) -> JudgeScores:
        self.load()
        q = (
            "You are judging a grid-based Christmas game world model. The FIRST image is "
            "the game frame BEFORE an action; the SECOND is the predicted frame AFTER the "
            f"action '{action}'. The player is the red 'S' badge. After '{action}', S should "
            f"be at grid cell {tuple(expected_cell)} (col,row, 1-indexed, 8x8 grid, top-left=1,1).\n"
            "Answer STRICTLY as JSON with keys:\n"
            '  "action_correct": true/false (is S now at the expected cell?),\n'
            '  "scene_intact": 1-5 (is everything ELSE — houses, presents, HUD, trees — '
            'unchanged and uncorrupted? 5=identical, 1=badly corrupted),\n'
            '  "quality": 1-5 (overall image quality/coherence).\n'
            "Return ONLY the JSON."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": before},
            {"type": "image", "image": after},
            {"type": "text", "text": q},
        ]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        imgs, vids = process_vision_info(messages)
        inp = self._proc(text=[text], images=imgs, videos=vids, padding=True,
                         return_tensors="pt").to(self.device)
        out = self._model.generate(**inp, max_new_tokens=128, do_sample=False)
        gen = out[:, inp.input_ids.shape[1]:]
        raw = self._proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> JudgeScores:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        ac, si, ql = False, 3, 3
        if m:
            try:
                d = json.loads(m.group(0))
                ac = bool(d.get("action_correct", False))
                si = int(d.get("scene_intact", 3))
                ql = int(d.get("quality", 3))
            except Exception:
                pass
        return JudgeScores(action_correct=ac, scene_intact=si, quality=ql, raw=raw)
