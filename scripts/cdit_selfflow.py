"""CDiT + Self-Flow world model (frontier stack), from scratch, compact.

CDiT (Conditional Diffusion Transformer):
  - patchify the 16-ch latent (128x128) -> tokens
  - CONTEXT frame (previous latent) patchified too, prepended as conditioning tokens
    (cross-info via full self-attention over [context || noisy] tokens)
  - action + timestep -> adaLN-Zero modulation of every block (DiT-style)

Self-Flow (self-supervised flow matching, ICML'26 idea):
  - PER-TOKEN timesteps: each noisy token gets its own noise level (dual-timestep)
  - 25% token MASKING (heavier corruption on a subset) -> information asymmetry
  - self-distillation: a deep block's features (teacher, detached) supervise a
    shallow block's features (student) via an auxiliary reconstruction loss.
  No external models / no extra data.

This is a training-side + architecture experiment; EDM-style velocity target on
the latent, change-weighted like our DIAMOND runs for fair comparison.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LAT_CH = 16
LAT = 128


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dim = 256

    def forward(self, t):  # t: (...,) continuous -> (..., dim)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[..., None].float() * freqs
        emb = torch.cat([a.cos(), a.sin()], -1)
        return self.mlp(emb)


class DiTBlock(nn.Module):
    """Standard DiT block with adaLN-Zero conditioning."""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sa_sh, sa_sc, sa_g, ml_sh, ml_sc, ml_g = self.ada(c).chunk(6, -1)
        h = modulate(self.norm1(x), sa_sh, sa_sc)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + sa_g.unsqueeze(1) * a
        h = modulate(self.norm2(x), ml_sh, ml_sc)
        x = x + ml_g.unsqueeze(1) * self.mlp(h)
        return x


class CDiTSelfFlow(nn.Module):
    def __init__(self, dim=384, depth=8, heads=6, patch=2, n_actions=5,
                 teacher_layer=6, student_layer=2, mask_ratio=0.25):
        super().__init__()
        self.patch = patch
        self.gh = LAT // patch                     # tokens per side
        self.n_tok = self.gh * self.gh
        self.dim = dim
        self.teacher_layer, self.student_layer = teacher_layer, student_layer
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Conv2d(LAT_CH, dim, patch, patch)   # noisy latent -> tokens
        self.ctx_embed = nn.Conv2d(LAT_CH, dim, patch, patch)     # context latent -> tokens
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, dim) * 0.02)
        self.ctx_pos = nn.Parameter(torch.randn(1, self.n_tok, dim) * 0.02)
        self.t_embed = TimestepEmbed(dim)          # per-token timestep
        self.act_embed = nn.Embedding(n_actions, dim)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)

        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, patch * patch * LAT_CH)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        # self-distillation projector (student features -> teacher feature space)
        self.distill_proj = nn.Linear(dim, dim)

    def unpatch(self, tokens):
        b = tokens.size(0)
        x = tokens.view(b, self.gh, self.gh, self.patch, self.patch, LAT_CH)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, LAT_CH, LAT, LAT)
        return x

    def forward(self, noisy, ctx, act, sigma_tok, return_distill=False):
        """noisy,(ctx): (B,16,128,128). act: (B,). sigma_tok: (B, n_tok) per-token sigma."""
        b = noisy.size(0)
        x = self.patch_embed(noisy).flatten(2).transpose(1, 2) + self.pos      # (B,N,dim)
        c_tok = self.ctx_embed(ctx).flatten(2).transpose(1, 2) + self.ctx_pos

        # Self-Flow: mask a subset of noisy tokens (information asymmetry)
        if self.training and self.mask_ratio > 0:
            k = int(self.n_tok * self.mask_ratio)
            idx = torch.rand(b, self.n_tok, device=x.device).argsort(1)[:, :k]
            mask = torch.zeros(b, self.n_tok, device=x.device, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            x = torch.where(mask.unsqueeze(-1), self.mask_token, x)

        # per-token timestep embedding added to tokens; action+global-t -> adaLN cond
        c_noise = (sigma_tok.clamp(min=1e-3).log() / 4)
        x = x + self.t_embed(c_noise)                                          # (B,N,dim)
        cond = self.act_embed(act) + self.t_embed(c_noise.mean(1))             # (B,dim) global

        seq = torch.cat([c_tok, x], 1)                                         # context || noisy
        student_feat = teacher_feat = None
        for i, blk in enumerate(self.blocks):
            seq = blk(seq, cond)
            if i == self.student_layer:
                student_feat = seq[:, self.n_tok:].clone()
            if i == self.teacher_layer:
                teacher_feat = seq[:, self.n_tok:].detach()
        out = self.head(self.norm_out(seq[:, self.n_tok:]))                    # noisy-half only
        pred = self.unpatch(out)
        if return_distill and student_feat is not None and teacher_feat is not None:
            # NORMALIZE both feature sets before the MSE (cosine-style) so scale drift
            # of the moving teacher target can't blow the loss up (self-distillation
            # divergence fix; DINO/Self-Flow normalize teacher & student features).
            s = F.normalize(self.distill_proj(student_feat), dim=-1)
            teach = F.normalize(teacher_feat, dim=-1)
            distill = (1 - (s * teach).sum(-1)).mean()   # 1 - cosine sim, in [0,2]
            return pred, distill
        return pred, torch.zeros((), device=noisy.device)


HEATMAP_CH = 2


class RichCDiT(CDiTSelfFlow):
    """CDiT + Self-Flow + RICH conditioning:
      - heatmap (2ch: from/to cells) patchified and ADDED to the context tokens
        (tells the transformer WHERE the change is, at every token position)
      - target-object + HUD embeddings ADDED to the adaLN cond vector (the WHAT)
    Everything else (per-token timesteps, masking, self-distillation) inherited.
    Set net.rich = dict(heatmap=(B,2,H,W), target=(B,), hud=(B,), hud_delta=(B,)) per batch.
    """
    def __init__(self, *a, n_targets=8, n_hud=6, **k):
        super().__init__(*a, **k)
        d = self.dim
        self.hm_embed = nn.Conv2d(HEATMAP_CH, d, self.patch, self.patch)
        self.target_emb = nn.Embedding(n_targets, d)
        self.hud_emb = nn.Embedding(n_hud, d)
        self.hud_delta_proj = nn.Linear(1, d)
        self.rich = None

    def forward(self, noisy, ctx, act, sigma_tok, return_distill=False):
        b = noisy.size(0)
        x = self.patch_embed(noisy).flatten(2).transpose(1, 2) + self.pos
        c_tok = self.ctx_embed(ctx).flatten(2).transpose(1, 2) + self.ctx_pos
        # RICH-WHERE: add heatmap tokens to the context tokens
        if self.rich is not None and self.rich.get("heatmap") is not None:
            hm_tok = self.hm_embed(self.rich["heatmap"]).flatten(2).transpose(1, 2)
            c_tok = c_tok + hm_tok

        if self.training and self.mask_ratio > 0:
            kmask = int(self.n_tok * self.mask_ratio)
            idx = torch.rand(b, self.n_tok, device=x.device).argsort(1)[:, :kmask]
            mask = torch.zeros(b, self.n_tok, device=x.device, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            x = torch.where(mask.unsqueeze(-1), self.mask_token, x)

        c_noise = (sigma_tok.clamp(min=1e-3).log() / 4)
        x = x + self.t_embed(c_noise)
        cond = self.act_embed(act) + self.t_embed(c_noise.mean(1))
        # RICH-WHAT: add target + HUD embeddings to the cond
        if self.rich is not None:
            cond = cond + self.target_emb(self.rich["target"]) \
                        + self.hud_emb(self.rich["hud"]) \
                        + self.hud_delta_proj(self.rich["hud_delta"][:, None])

        seq = torch.cat([c_tok, x], 1)
        student_feat = teacher_feat = None
        for i, blk in enumerate(self.blocks):
            seq = blk(seq, cond)
            if i == self.student_layer:
                student_feat = seq[:, self.n_tok:].clone()
            if i == self.teacher_layer:
                teacher_feat = seq[:, self.n_tok:].detach()
        out = self.head(self.norm_out(seq[:, self.n_tok:]))
        pred = self.unpatch(out)
        if return_distill and student_feat is not None and teacher_feat is not None:
            # NORMALIZE both feature sets before the MSE (cosine-style) so scale drift
            # of the moving teacher target can't blow the loss up (self-distillation
            # divergence fix; DINO/Self-Flow normalize teacher & student features).
            s = F.normalize(self.distill_proj(student_feat), dim=-1)
            teach = F.normalize(teacher_feat, dim=-1)
            distill = (1 - (s * teach).sum(-1)).mean()   # 1 - cosine sim, in [0,2]
            return pred, distill
        return pred, torch.zeros((), device=noisy.device)
