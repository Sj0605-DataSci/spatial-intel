"""Axis 4: rich-conditioned controllable renderer, built ON DIAMOND.

Two conditioning streams, from the transition's effects (given as INPUT):
  1. HEATMAP (where): Gaussian blobs at moved_player.from & .to cells, rendered on
     the 128x128 latent grid via the calibrated per-game geometry, CONCATENATED to
     the latent channels. Tells the UNet exactly where change happens.
  2. STRUCTURED VECTOR (what): action, target object, and HUD int-changes, embedded
     and added to DIAMOND's FiLM cond vector (injected into every UNet block).

Implementation stays on DIAMOND's Denoiser/InnerModel:
  - heatmap channels are packed into `obs` (so InnerModelConfig.img_channels grows
    and DIAMOND's conv_in handles them with no code change);
  - the structured vector is added inside a thin InnerModel subclass that owns an
    extra projection and reads per-batch rich features stashed on the module.
"""
import json
import sys
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.inner_model import InnerModel, InnerModelConfig
from models.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from latent_wm import LATENT_CH, ChangeWeightedDenoiser

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
LAT = 128            # latent grid side (1024 / 8)
HEATMAP_CH = 2       # from-cell, to-cell


# ----------------------------------------------------------------------------
# 1. HEATMAP builder (WHERE) — uses the calibrated per-game geometry
# ----------------------------------------------------------------------------
def load_geom(game):
    return json.loads((BUNDLE / game / "geometry.json").read_text())


def cell_to_latent_xy(geom, col, row):
    """RDDL cell -> pixel center -> latent-grid coords (0..LAT-1). Frame is 512 wide
    in the bundle but we encode at 1024, so scale pixel coords by (1024/frame_w)/8."""
    ox, oy = geom["origin"]; cw, ch = geom["cell"]
    # geometry was fit in the native bundle frame (≈512 wide); latent is 128 over 1024.
    fx = fy = 1024.0 / 512.0            # bundle frame ~512 -> 1024 encode
    px = (ox + (col - 1) * cw) * fx
    py = (oy + (row - 1) * ch) * fy
    return px / 8.0, py / 8.0            # /8 VAE downscale -> latent grid


def gaussian_map(cx, cy, sigma=6.0):
    ys, xs = np.mgrid[0:LAT, 0:LAT]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return g.astype(np.float32)


def build_heatmap(geom, eff, device):
    """(2, LAT, LAT): channel0 = from-cell blob, channel1 = to-cell blob."""
    hm = np.zeros((HEATMAP_CH, LAT, LAT), np.float32)
    mp = eff.get("moved_player")
    if mp:
        fx, fy = cell_to_latent_xy(geom, *mp["from"])
        tx, ty = cell_to_latent_xy(geom, *mp["to"])
        hm[0] = gaussian_map(fx, fy)
        hm[1] = gaussian_map(tx, ty)
    else:
        # context action (Space): no move; mark player's current cell in both
        # (the model should keep the player put and change HUD/objects)
        pass
    return torch.from_numpy(hm).to(device)


# ----------------------------------------------------------------------------
# 2. STRUCTURED VECTOR (WHAT) — action + target object + HUD deltas
# ----------------------------------------------------------------------------
class RichFeatures:
    """Builds the per-game vocab and encodes a transition's structured features."""
    def __init__(self, game):
        trans = [json.loads(l) for l in open(BUNDLE / game / "transitions.jsonl")]
        self.actions = sorted({t["action_name"] for t in trans})
        self.targets = sorted({t["target"] for t in trans if t["target"]}) + ["<none>"]
        # HUD counters that ever change
        keys = set()
        for t in trans:
            keys |= set(t["effects"].get("int_changes", {}).keys())
        self.hud_keys = sorted(keys) + ["<none>"]
        self.a2i = {a: i for i, a in enumerate(self.actions)}
        self.t2i = {t: i for i, t in enumerate(self.targets)}
        self.h2i = {h: i for i, h in enumerate(self.hud_keys)}

    def encode(self, t):
        """-> dict of indices/values for one transition."""
        eff = t["effects"]
        ic = eff.get("int_changes", {})
        # take the first HUD change (usually score + one counter); encode its key & delta
        hud_key, hud_delta = "<none>", 0.0
        for k, (a, b) in ic.items():
            if k != "total_score":
                hud_key, hud_delta = k, float(b - a); break
        else:
            if "total_score" in ic:
                a, b = ic["total_score"]; hud_key, hud_delta = "total_score", float(b - a)
        return dict(
            action=self.a2i[t["action_name"]],
            target=self.t2i.get(t["target"] or "<none>", self.t2i["<none>"]),
            hud=self.h2i[hud_key],
            hud_delta=hud_delta,
        )


class RichInnerModel(InnerModel):
    """InnerModel + a projection for the structured vector, added to the cond.
    Rich features for the current batch are stashed on `.rich` before forward()."""
    def __init__(self, cfg, n_targets, n_hud):
        super().__init__(cfg)
        c = cfg.cond_channels
        self.target_emb = nn.Embedding(n_targets, c)
        self.hud_emb = nn.Embedding(n_hud, c)
        self.hud_delta_proj = nn.Linear(1, c)
        self.rich_proj = nn.Sequential(nn.SiLU(), nn.Linear(c, c))
        self.rich = None    # set per-batch: dict with target,hud,hud_delta tensors

    def forward(self, noisy_next_obs, c_noise, obs, act):
        cond = self.cond_proj(self.noise_emb(c_noise) + self.act_emb(act))
        if self.rich is not None:
            r = (self.target_emb(self.rich["target"])
                 + self.hud_emb(self.rich["hud"])
                 + self.hud_delta_proj(self.rich["hud_delta"][:, None]))
            cond = cond + self.rich_proj(r)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class RichDenoiser(Denoiser):
    """Change-weighted DIAMOND denoiser that ALSO takes a heatmap context (concat to
    the model input) and rich structured features (into the cond). The heatmap is
    NOT part of the predicted target; it is conditioning only.

    Per-batch, set:  net.heatmap  (B, HEATMAP_CH, H, W)  and  net.inner_model.rich
    before calling forward().
    """
    change_weight = 20.0
    heatmap = None
    # optional noise-gated LPIPS perceptual loss (set via set_perceptual)
    lpips_weight = 0.0
    sigma_gate = 1.0
    perc_max = 1
    gt_next_rgb = None

    def set_perceptual(self, vae, lpips_fn, decode_fn, weight=0.5, gate=1.0, perc_max=1):
        object.__setattr__(self, "_vae", vae)
        object.__setattr__(self, "_lpips", lpips_fn)
        object.__setattr__(self, "_decode_fn", decode_fn)
        self.lpips_weight = weight
        self.sigma_gate = gate
        self.perc_max = perc_max

    def compute_model_output(self, noisy_next_obs, obs, act, cs):
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        # concat heatmap channels to the CONTEXT obs (conditioning), not the noisy target
        if self.heatmap is not None:
            rescaled_obs = torch.cat([rescaled_obs, self.heatmap], dim=1)
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)

    def forward(self, batch):
        # single-step (num_steps_conditioning=1); obs=(B,1,C,H,W), next=(B,C,H,W)
        n = self.cfg.inner_model.num_steps_conditioning
        obs = batch.obs[:, :n].reshape(batch.obs.size(0), -1, *batch.obs.shape[-2:])
        next_obs = batch.obs[:, n]
        act = batch.act[:, :n]
        sigma = self.sample_sigma_training(obs.size(0), self.device)
        noisy = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy, obs, act, cs)
        target = (next_obs - cs.c_skip * noisy) / cs.c_out
        cur = obs[:, -next_obs.size(1):]
        w = 1.0 + self.change_weight * (next_obs - cur).abs()
        mse = ((model_output - target) ** 2 * w).mean()

        logs = {"loss_denoising": mse.detach()}
        perc = torch.zeros((), device=self.device)
        low = (sigma.flatten() < self.sigma_gate)
        if getattr(self, "_lpips", None) is not None and low.any():
            pred_clean = cs.c_skip * noisy + cs.c_out * model_output
            idx = low.nonzero(as_tuple=True)[0]
            if idx.numel() > self.perc_max:
                idx = idx[:self.perc_max]
            pred_rgb = self._decode_fn(pred_clean[idx])
            perc = self._lpips(pred_rgb, self.gt_next_rgb[idx]).mean()
            logs["loss_lpips"] = perc.detach()
        loss = mse + self.lpips_weight * perc
        return loss, logs


def make_rich_denoiser(device, n_actions, n_targets, n_hud,
                       cond_channels=512, sigma_data=0.612, change_weight=20.0,
                       channels=None):
    """DIAMOND-based denoiser with RichInnerModel; conv_in takes latent+heatmap chans."""
    inner = InnerModelConfig(
        img_channels=LATENT_CH,                  # target/context latent is 16ch...
        num_steps_conditioning=1,
        cond_channels=cond_channels,
        depths=[2, 2, 2, 2],
        channels=channels or [64, 128, 256, 512],
        attn_depths=[False, False, True, True],
        num_actions=n_actions,
    )
    cfg = DenoiserConfig(inner_model=inner, sigma_data=sigma_data, sigma_offset_noise=0.1)
    net = RichDenoiser(cfg)
    net.inner_model = RichInnerModel(inner, n_targets, n_hud)
    # conv_in must accept (context 16 + heatmap 2) + (noisy 16) = 34 channels
    from models.blocks import Conv3x3
    net.inner_model.conv_in = Conv3x3(LATENT_CH + HEATMAP_CH + LATENT_CH,
                                      inner.channels[0])
    net.change_weight = change_weight
    net.to(device)
    net.setup_training(SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20))
    return net
