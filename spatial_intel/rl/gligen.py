"""GLIGEN-style coordinate grounding for FLUX.2 (structured position injection).

Why: attention-based grounding failed (Exp: attention perfect, placement ~0%).
GLIGEN/SpatialLock/SOTA instead inject the target position as a STRUCTURED signal
into the generation path — a grounding token = MLP(Fourier(x,y)) — that the image
tokens attend to in the joint attention. Position enters as an embedding, not text
digits.

Adaptation to FLUX.2 (MM-DiT joint attention): we build one grounding token of the
text embedding dim (7680) from the target cell's normalized (x,y), and APPEND it to
the encoder_hidden_states (text) stream (+ a matching positional id). The image
patches then attend to it through the existing joint attention. Trained via LoRA +
this small grounding encoder; a zero-init output gate keeps the pretrained model
unchanged at step 0 (GLIGEN's gated-injection idea).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def fourier_embed(coords: torch.Tensor, n_freqs: int = 16) -> torch.Tensor:
    """coords: (B, 2) in [0,1]. Returns (B, 2*2*n_freqs) Fourier features."""
    freqs = torch.arange(n_freqs, device=coords.device, dtype=torch.float32)
    freqs = (2.0 ** freqs) * torch.pi                       # (n_freqs,)
    ang = coords[:, :, None] * freqs[None, None, :]         # (B, 2, n_freqs)
    feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B,2,2*n_freqs)
    return feat.flatten(1)                                   # (B, 4*n_freqs)


class GroundingEncoder(nn.Module):
    """Fourier(x,y) -> grounding token of dim `embed_dim`, with a zero-init gate."""

    def __init__(self, embed_dim: int = 7680, n_freqs: int = 16, hidden: int = 1024):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 4 * n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )
        # GLIGEN-style gate: start at 0 so the base model is untouched at init,
        # opens as training makes the grounding token useful.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, target_xy: torch.Tensor) -> torch.Tensor:
        """target_xy: (B, 2) normalized [0,1]. Returns (B, 1, embed_dim) token."""
        w_dtype = self.mlp[0].weight.dtype                   # match the MLP's dtype
        f = fourier_embed(target_xy, self.n_freqs).to(w_dtype)
        tok = self.mlp(f)                                    # (B, embed_dim)
        return (torch.tanh(self.gate) * tok).unsqueeze(1)    # (B, 1, embed_dim)


def append_grounding(encoder_hidden_states: torch.Tensor, text_ids: torch.Tensor,
                     grounding_tok: torch.Tensor):
    """Append the grounding token to the text stream and give it a positional id.

    encoder_hidden_states: (B, T, D); text_ids: (B, T, 4) or (T,4). Grounding id is
    a fresh position (zeros) — RoPE on text ids in FLUX is typically all-zero for
    txt, so a zero id is consistent.
    """
    B = encoder_hidden_states.shape[0]
    ehs = torch.cat([encoder_hidden_states, grounding_tok.to(encoder_hidden_states.dtype)], dim=1)
    if text_ids.ndim == 2:
        text_ids = text_ids.unsqueeze(0).expand(B, -1, -1)
    gid = torch.zeros((B, 1, text_ids.shape[-1]), device=text_ids.device, dtype=text_ids.dtype)
    tids = torch.cat([text_ids, gid], dim=1)
    return ehs, tids
