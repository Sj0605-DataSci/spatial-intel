"""Training-free attention-BIAS intervention for FLUX.2 (VLM-thinking → attention).

Hypothesis (from Experiment 2): FLUX places the triangle wrong because the
coordinate/triangle text tokens attend to image CORNERS (sinks), not the target
cell (success focus-ratio 2.8 vs failure 0.9). Here we test the fix directly,
with NO training: a VLM predicts the target cell, we turn it into a patch mask,
and we ADD a positive bias to the attention logits from the coordinate tokens to
the target-cell patches at every denoising step — forcing the model to "look"
where the VLM says to.

Mechanism: Flux2AttnProcessor calls dispatch_attention_fn(..., attn_mask=...),
which is an ADDITIVE bias on the pre-softmax logits. So we inject a bias tensor
of shape [1, 1, S, S] that is `+strength` on rows = coordinate/triangle text
tokens, cols = target-cell image patches, and 0 elsewhere. This nudges softmax
mass onto the target without reimplementing attention.

The joint sequence is [text (T tokens) | generated image (P patches) | (img2img)
conditioning image (P patches)]. We bias the generated-image columns only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor, _get_qkv_projections, apply_rotary_emb, dispatch_attention_fn)


@dataclass
class BiasSpec:
    token_rows: List[int]        # text-token indices to bias (coord/triangle tokens)
    text_len: int                # T
    n_gen_patches: int           # P (generated image patches, = side*side)
    side: int                    # patch grid side (e.g. 32)
    target_cols: List[int] = field(default_factory=list)  # flat patch idxs of target cell
    strength: float = 4.0        # additive logit bias
    enabled: bool = True


class BiasedFlux2AttnProcessor(Flux2AttnProcessor):
    """Flux2AttnProcessor + an additive attention bias toward target patches."""

    def __init__(self, spec: BiasSpec):
        super().__init__()
        self.spec = spec

    def _build_mask(self, S: int, device, dtype):
        sp = self.spec
        m = torch.zeros((1, 1, S, S), device=device, dtype=dtype)
        if not sp.enabled or not sp.target_cols or not sp.token_rows:
            return None
        # generated image patches occupy [T : T+P] in the joint sequence
        base = sp.text_len
        cols = [base + c for c in sp.target_cols if base + c < S]
        rows = [r for r in sp.token_rows if r < S]
        if not cols or not rows:
            return None
        r_idx = torch.tensor(rows, device=device)
        c_idx = torch.tensor(cols, device=device)
        m[0, 0, r_idx[:, None], c_idx[None, :]] = sp.strength
        return m

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        query, key, value, eq, ek, ev = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query); key = attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            eq = eq.unflatten(-1, (attn.heads, -1)); ek = ek.unflatten(-1, (attn.heads, -1))
            ev = ev.unflatten(-1, (attn.heads, -1))
            eq = attn.norm_added_q(eq); ek = attn.norm_added_k(ek)
            query = torch.cat([eq, query], dim=1)
            key = torch.cat([ek, key], dim=1)
            value = torch.cat([ev, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        S = query.shape[1]
        bias = self._build_mask(S, query.device, query.dtype)
        if attention_mask is not None and bias is not None:
            bias = bias + attention_mask
        elif bias is None:
            bias = attention_mask

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=bias,
            backend=self._attention_backend, parallel_config=self._parallel_config)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def target_cell_patches(target_topleft, spec, side: int, object_size: int) -> List[int]:
    """Flat patch indices (row-major over side×side) covering the target cell block."""
    cw = side / spec.n
    tc, tr = target_topleft
    cols = []
    for dr in range(object_size):
        for dc in range(object_size):
            c0 = int((tc + dc) * cw); c1 = int((tc + dc + 1) * cw)
            r0 = int((tr + dr) * cw); r1 = int((tr + dr + 1) * cw)
            for pr in range(r0, min(r1 + 1, side)):
                for pc in range(c0, min(c1 + 1, side)):
                    cols.append(pr * side + pc)
    return sorted(set(cols))


def install_bias(transformer, spec: BiasSpec):
    """Swap all double-stream blocks' attention processors for the biased one.
    Returns a handle to toggle/remove."""
    originals = {}
    procs = []
    for i, blk in enumerate(transformer.transformer_blocks):
        originals[i] = blk.attn.processor
        p = BiasedFlux2AttnProcessor(spec)
        blk.attn.processor = p
        procs.append(p)
    return originals, procs


def remove_bias(transformer, originals):
    for i, blk in enumerate(transformer.transformer_blocks):
        if i in originals:
            blk.attn.processor = originals[i]
