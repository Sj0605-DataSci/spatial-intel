"""SDE flow-matching step with a closed-form per-step log-likelihood.

Background
----------
FLUX uses `FlowMatchEulerDiscreteScheduler`. Its deterministic (ODE) update at a
step with sigma -> sigma_next, given the model's velocity prediction `v` (the
transformer output), is:

    x_next = x + (sigma_next - sigma) * v                                   (ODE)

This is a deterministic map: P(x_next | x) is a Dirac delta, so log p is
undefined and there is no policy to do RL on.

Flow-GRPO (Liu et al., 2025, arXiv:2505.05470) converts this ODE to a
marginal-preserving reverse SDE. Discretised with Euler, each step becomes an
isotropic Gaussian:

    mean   = x + (sigma_next - sigma) * v  +  drift_correction(sigma) * dt
    std    = eta * sqrt(|dt|)                         (per-step noise scale)
    x_next ~ Normal(mean, std^2 * I)

so the per-step log-likelihood is the standard diagonal-Gaussian form. We use the
Flow-GRPO noise schedule sigma_t = a * sqrt(t / (1 - t)); here `t` is the
flow-matching time, which for this scheduler equals the running `sigma` (sigmas
run 1 -> 0 over the trajectory). `eta` (== `a`) is the single knob trading
exploration (large) vs. fidelity to the base ODE (eta -> 0 recovers the ODE).

We keep the SAME mean as the ODE plus the small Flow-GRPO drift term, and add
Gaussian noise of scale `std`. With eta small the sampler stays close to the base
model's trajectory (low distribution shift) while still being stochastic enough
to explore — which is what GRPO needs to see reward variance within a group.

Everything here operates on flattened latents; the caller reshapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

# Numerical floor so std is never exactly 0 (endpoints sigma in {0, 1}).
_STD_FLOOR = 1e-4


def sde_noise_std(sigma: torch.Tensor, dt: torch.Tensor, eta: float) -> torch.Tensor:
    """Flow-GRPO per-step noise std: eta * sqrt(t/(1-t)) * sqrt(|dt|).

    `sigma` is the flow-matching time t in (0, 1). At the endpoints the t/(1-t)
    term degenerates, so we clamp t into [eps, 1-eps] before the sqrt.
    """
    t = sigma.clamp(1e-4, 1.0 - 1e-4)
    scale = eta * torch.sqrt(t / (1.0 - t))
    std = scale * torch.sqrt(dt.abs())
    return std.clamp_min(_STD_FLOOR)


def sde_step_mean(x: torch.Tensor, v: torch.Tensor,
                  sigma: torch.Tensor, sigma_next: torch.Tensor,
                  eta: float) -> torch.Tensor:
    """Mean of the Gaussian transition for one SDE step.

    Base ODE mean `x + dt*v` plus the Flow-GRPO drift correction
    (sigma^2 / (2 t)) * (x + (1 - t) v) * dt, with t == sigma.
    """
    dt = sigma_next - sigma                       # negative (sigma decreases)
    t = sigma.clamp(1e-4, 1.0 - 1e-4)
    drift_corr = (sigma * sigma) / (2.0 * t) * (x + (1.0 - t) * v)
    return x + dt * v + drift_corr * dt


def gaussian_log_prob(x_next: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor) -> torch.Tensor:
    """Log-prob of an isotropic Gaussian, summed over all latent dims.

    Returns one scalar per batch element. `std` may be a scalar tensor broadcast
    over the latent dims (the noise is isotropic, so std is shared per element).
    """
    var = std * std
    # sum over every non-batch dim
    dims = tuple(range(1, x_next.ndim))
    se = (x_next - mean) ** 2
    log_unnorm = -(se / (2.0 * var)).sum(dim=dims)
    n = x_next[0].numel()
    log_norm = -0.5 * n * torch.log(2.0 * torch.pi * var.reshape(()))
    return log_unnorm + log_norm


@dataclass
class SDEStepResult:
    x_next: torch.Tensor       # sampled next latent
    log_prob: torch.Tensor     # log pi(x_next | x), one scalar per batch elem
    mean: torch.Tensor         # transition mean (for recomputing log-prob later)
    std: torch.Tensor          # transition std  (")


def sde_step(x: torch.Tensor, v: torch.Tensor,
             sigma: torch.Tensor, sigma_next: torch.Tensor,
             eta: float, *, generator: torch.Generator | None = None,
             noise: torch.Tensor | None = None) -> SDEStepResult:
    """One stochastic flow-matching step.

    Pass `noise` to reproduce a recorded sample (used when recomputing log-probs
    under the updated policy during the GRPO inner loop); otherwise we draw fresh
    noise with `generator`.
    """
    dt = sigma_next - sigma
    mean = sde_step_mean(x, v, sigma, sigma_next, eta)
    std = sde_noise_std(sigma, dt, eta)
    if noise is None:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    x_next = mean + std * noise
    log_prob = gaussian_log_prob(x_next, mean, std)
    return SDEStepResult(x_next=x_next, log_prob=log_prob, mean=mean, std=std)


def recompute_log_prob(x_next: torch.Tensor, x: torch.Tensor, v: torch.Tensor,
                       sigma: torch.Tensor, sigma_next: torch.Tensor,
                       eta: float) -> torch.Tensor:
    """Recompute log pi(x_next | x) under (possibly updated) velocity `v`.

    Used in the GRPO loss: the stored x, x_next are fixed; only `v` (hence the
    mean) changes as the LoRA updates, so the ratio pi_new/pi_old has a gradient.
    """
    mean = sde_step_mean(x, v, sigma, sigma_next, eta)
    std = sde_noise_std(sigma, sigma_next - sigma, eta)
    return gaussian_log_prob(x_next, mean, std)
