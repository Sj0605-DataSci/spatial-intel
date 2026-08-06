"""Cross-attention capture for FLUX.2 (mech-interp on latent spatial reasoning).

FLUX uses MM-DiT joint attention: text (encoder) tokens are concatenated BEFORE
image tokens, so in the joint sequence positions [0:T] are text and [T:] are the
image patches. The fused attention kernel doesn't expose weights, so we hook each
Flux2Attention's to_q/to_k (and add_q/add_k for the text stream) projections,
recompute softmax(QK^T) ONLY for the text rows we care about (the 'triangle' and
target-coordinate tokens) against the image columns, and accumulate a running
mean heatmap per run.

The output is, per run, a single H_patch x W_patch heatmap: "where in the image
do the triangle/coordinate text tokens attend" — averaged over heads, the chosen
layers, and denoising steps. That directly probes whether the model attends to
the correct grid cell (latent reasoning) regardless of what it finally renders.

To keep memory bounded across 150 runs x 28 steps x N layers we never store raw
Q/K — we fold each layer/step straight into the accumulator and free it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class AttnAccumulator:
    """Running-mean text->image attention heatmap for one generation."""
    token_ids: List[int]                 # which text-token rows to track
    text_len: int                        # number of text tokens (T)
    n_image: int                         # number of image patches
    heat: Optional[torch.Tensor] = None  # [n_image] running sum
    count: int = 0

    def add(self, attn_text_to_img: torch.Tensor):
        # attn_text_to_img: [len(token_ids), n_image] (already head-averaged)
        v = attn_text_to_img.mean(dim=0).float()      # avg over the tracked tokens
        if self.heat is None:
            self.heat = torch.zeros_like(v)
        self.heat += v
        self.count += 1

    def heatmap(self, h_patch: int, w_patch: int) -> np.ndarray:
        if self.heat is None or self.count == 0:
            return np.zeros((h_patch, w_patch), np.float32)
        m = (self.heat / self.count).cpu().numpy()
        # img2img concatenates [generated_latent, conditioning_image] -> only the
        # FIRST h*w patches are the image being generated; the rest is the cond input.
        m = m[: h_patch * w_patch].reshape(h_patch, w_patch)
        # normalise to [0,1] for visualisation
        if m.max() > m.min():
            m = (m - m.min()) / (m.max() - m.min())
        return m.astype(np.float32)


class AttentionCapture:
    """Context-manager that hooks the transformer and accumulates a heatmap.

    Usage:
        cap = AttentionCapture(pipe.transformer, token_ids=[...], layers=range(0,8))
        with cap.capture(text_len=T):
            pipe(...)
        heat = cap.acc.heatmap(h_patch, w_patch)
    """

    def __init__(self, transformer, token_ids, layers=None, heads_avg=True):
        self.t = transformer
        self.token_ids = list(token_ids)
        # default: first 8 double-stream blocks (most semantic per FLUX interp work)
        blocks = transformer.transformer_blocks
        # the double-stream blocks are the ones with text<->image joint attention
        # (they have add_q_proj). FLUX.2-klein-4B has 5 of them.
        self.layers = list(layers) if layers is not None else list(range(len(blocks)))
        self.heads_avg = heads_avg
        self._hooks = []
        self.acc: Optional[AttnAccumulator] = None
        self._text_len = None

    def _make_hook(self, attn_module):
        def hook(module, inputs, kwargs, output):
            # We need q,k for this block's joint attention. Recompute from the
            # module's projections using the captured hidden states.
            hs = kwargs.get("hidden_states")
            ehs = kwargs.get("encoder_hidden_states")
            if hs is None or ehs is None or self._text_len is None:
                return
            with torch.no_grad():
                q_img = module.to_q(hs)
                k_img = module.to_k(hs)
                q_txt = module.add_q_proj(ehs)
                k_txt = module.add_k_proj(ehs)
                # shape into heads
                B = hs.shape[0]
                heads = module.heads
                def reshape(x):
                    return x.view(B, x.shape[1], heads, -1).transpose(1, 2)  # B,H,S,d
                q = torch.cat([reshape(q_txt), reshape(q_img)], dim=2)
                k = torch.cat([reshape(k_txt), reshape(k_img)], dim=2)
                T = ehs.shape[1]
                d = q.shape[-1]
                # only the tracked text rows vs the image columns
                rows = [i for i in self.token_ids if i < T]
                if not rows:
                    return
                qr = q[:, :, rows, :]                          # B,H,len(rows),d
                ki = k[:, :, T:, :]                            # B,H,n_image,d
                scores = torch.matmul(qr, ki.transpose(-1, -2)) / (d ** 0.5)
                attn = torch.softmax(scores.float(), dim=-1)   # B,H,rows,n_image
                a = attn.mean(dim=1)[0]                        # avg heads -> rows,n_image
                if self.acc is None:
                    self.acc = AttnAccumulator(self.token_ids, T, a.shape[-1])
                self.acc.add(a)
        return hook

    def capture(self, text_len: int):
        self._text_len = text_len
        return self

    def __enter__(self):
        blocks = self.t.transformer_blocks
        for li in self.layers:
            attn = blocks[li].attn
            h = attn.register_forward_hook(self._make_hook(attn), with_kwargs=True)
            self._hooks.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._text_len = None
        return False


def find_token_ids(processor, prompt: str, keywords) -> List[int]:
    """Return token positions in `prompt` whose decoded text matches any keyword
    (e.g. 'triangle', the coordinate digits). Best-effort substring match."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids[0]
    hits = []
    for i, tid in enumerate(ids.tolist()):
        piece = tok.decode([tid]).strip().lower()
        if piece and any(kw in piece or piece in kw for kw in keywords):
            hits.append(i)
    return hits
