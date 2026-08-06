"""Attention-alignment auxiliary loss for supervised LoRA training.

Insight (Experiment 2): FLUX fails placement because coordinate/triangle text
tokens don't attend to the target cell. Plain SFT never supervises this. Here we
add a loss that, during training, makes those tokens attend to the target-cell
patches — so the model LEARNS the attention→placement link instead of being forced
per-step at inference (which we found too weak).

Total training loss = MSE(v_pred, v_target)  +  lambda * attn_align_loss

attn_align_loss = mean over tracked layers of  -log( sum_{p in target} A[token, p] )
i.e. maximise the attention mass the coord/triangle tokens put on the target-cell
patches. A is softmax(QK^T) restricted to [tracked text rows] x [generated patches],
computed with gradients (manual, not the fused kernel).

We register forward hooks on the double-stream blocks that recompute this small
attention slice from the block's captured hidden states. To keep it differentiable
and cheap we only compute the tracked rows vs generated-image columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class AlignState:
    token_rows: List[int]
    text_len: int
    n_gen: int                      # generated image patches (1024)
    target_cols: List[int]          # patch indices of the target cell
    losses: list = field(default_factory=list)   # per-layer align losses (with grad)


class AttnAlignHooks:
    """Hooks the transformer's double-stream attention modules and, each forward,
    computes the coord-token->target-patch attention-alignment loss with grad."""

    def __init__(self, transformer, state: AlignState):
        self.t = transformer
        self.st = state
        self._handles = []

    def _hook(self, module):
        def fn(mod, inputs, kwargs, output):
            hs = kwargs.get("hidden_states")
            ehs = kwargs.get("encoder_hidden_states")
            if hs is None or ehs is None:
                return
            st = self.st
            rows = [r for r in st.token_rows if r < ehs.shape[1]]
            cols = [c for c in st.target_cols if c < st.n_gen]
            if not rows or not cols:
                return
            # recompute q for the tracked text rows, k for the generated patches.
            # text (encoder) stream uses add_q_proj/add_k_proj; image uses to_q/to_k.
            q_txt = mod.add_q_proj(ehs)            # (B, T, H*d)
            k_img = mod.to_k(hs)                   # (B, 2048, H*d) [gen+cond]
            B = ehs.shape[0]; H = mod.heads
            def shape(x):
                return x.view(B, x.shape[1], H, -1).transpose(1, 2)   # B,H,S,d
            q = shape(q_txt)[:, :, rows, :]        # B,H,|rows|,d
            k = shape(k_img)[:, :, :st.n_gen, :]   # B,H,n_gen,d (generated only)
            d = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)  # B,H,rows,n_gen
            attn = torch.softmax(scores.float(), dim=-1)               # over patches
            col_idx = torch.tensor(cols, device=attn.device)
            on_target = attn[..., col_idx].sum(-1)                     # B,H,rows: mass on target
            # maximise on-target mass -> minimise -log(mass)
            loss = -(on_target.clamp_min(1e-6).log()).mean()
            st.losses.append(loss)
        return fn

    def __enter__(self):
        self.st.losses = []
        for blk in self.t.transformer_blocks:
            h = blk.attn.register_forward_hook(self._hook(blk.attn), with_kwargs=True)
            self._handles.append(h)
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def align_loss(self):
        if not self.st.losses:
            return torch.tensor(0.0)
        return torch.stack(self.st.losses).mean()
