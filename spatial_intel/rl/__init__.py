"""Vanilla RLVR for FLUX.2 — REINFORCE on a verifiable +1/-1 reward.

Finetunes a LoRA on FLUX.2-klein-base-4B so it renders the spatial-move task
correctly, using ONLY the verifiable reward from `spatial_intel.reward` (+1 if
the red triangle lands on the commanded block, -1 otherwise). No target images,
no learned reward model — that is what makes it RLVR.

This is plain REINFORCE with a single running-mean baseline. NO grouping, critic,
importance ratio, or KL term — that machinery is GRPO/PPO and is deliberately not
here. The one variance reduction is the scalar baseline (subtract average reward).

The crux (see `sde.py`): FLUX is a flow-matching model whose default sampler is a
deterministic ODE, which has no per-step log-likelihood. We convert it to a
marginal-preserving SDE (Flow-GRPO, Liu et al. 2025, arXiv:2505.05470) so each
denoising step is an isotropic Gaussian with closed-form log-prob — the only
thing a policy gradient needs. (We borrow the SDE; not GRPO itself.)

Modules
-------
- sde.py        : SDE flow-matching step + per-step Gaussian log-prob (validated).
- policy.py     : FluxPolicy — base-4B + LoRA, records SDE rollouts, decodes images.
- reinforce.py  : verifiable reward, running baseline, REINFORCE loss, train_step.
"""
