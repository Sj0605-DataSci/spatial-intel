"""Vanilla RLVR = REINFORCE on a verifiable +1/-1 reward, updating a FLUX LoRA.

This is the simplest correct policy-gradient finetune. NO grouping, NO critic,
NO importance ratio, NO KL term (that machinery belongs to GRPO/PPO). Per command:

  1. Sample ONE image from the policy via the recorded SDE rollout (policy.py).
  2. Verifiable reward: +1 if the red triangle is on the commanded block, else -1
     (read straight off score_attempt(...).success — no learned reward model).
  3. REINFORCE loss with a single running-mean baseline b:
         loss = -(reward - b) * sum_t log pi(x_{t-1} | x_t)
     where the log-probs are recomputed WITH gradient under the current LoRA.
  4. Backprop into the LoRA adapters only; step the optimizer.

The baseline `b` is one scalar (the running average reward). It removes the
constant offset that otherwise makes every gradient point the same way when the
model mostly misses — the minimal variance fix, not a group of rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..grid import GridSpec
from ..reward import RewardConfig, score_attempt
from ..dataset import instruction_text
from .policy import FluxPolicy, Trajectory
from .sde import recompute_log_prob


@dataclass
class Command:
    start: tuple
    direction: str
    steps: int
    target: tuple

    @property
    def instruction(self) -> str:
        return instruction_text(self.direction, self.steps)


def build_prompt(cmd: Command) -> str:
    return (
        "A square grid with numbered columns and rows and a single solid red "
        "triangle occupying one square block. " + cmd.instruction +
        " Keep the grid, axes and labels identical; move ONLY the red triangle "
        "to its new grid block; output one solid red triangle snapped to the grid."
    )


def verifiable_reward(spec: GridSpec, image: np.ndarray, cmd: Command,
                      cfg: RewardConfig) -> tuple[float, bool]:
    """+1 on exact target block, -1 otherwise (strict verifiable spec)."""
    s = score_attempt(spec, image, start_topleft=cmd.start,
                      target_topleft=cmd.target, direction=cmd.direction, cfg=cfg)
    return (1.0 if s.success else -1.0), bool(s.success)


@dataclass
class RewardBreakdown:
    total: float            # combined scalar used by REINFORCE
    placement: float        # shaped distance+direction term [0,1]
    consistency: float      # image-match term (grid + shape IoU) [0,1]
    grid: float
    shape_match: float
    success: bool           # exact target block hit


def shaped_reward(spec: GridSpec, image: np.ndarray, cmd: Command, cfg: RewardConfig,
                  *, placement_w: float = 0.5, consistency_w: float = 0.5
                  ) -> RewardBreakdown:
    """Dense reward = placement (where) + image-consistency (exact shape/grid).

    Gives gradient even when nothing lands exactly (the 0% cold-start case): a
    near-miss with a clean grid still scores > a far-miss with a mangled grid, so
    the model gets a learning signal at every rollout.
    """
    from .consistency import consistency_reward
    from ..grid import from_array, to_array
    # the policy decodes at its work_size; score at env-native resolution
    if image.shape[:2] != (spec.img_h, spec.img_w):
        image = to_array(from_array(image).resize((spec.img_w, spec.img_h)))
    s = score_attempt(spec, image, start_topleft=cmd.start,
                      target_topleft=cmd.target, direction=cmd.direction, cfg=cfg)
    placement = float(s.reward)                     # already shaped, in [0,1]
    cons = consistency_reward(spec, image, cmd.target)
    total = placement_w * placement + consistency_w * cons.combined
    return RewardBreakdown(total=total, placement=placement, consistency=cons.combined,
                           grid=cons.grid_consistency, shape_match=cons.shape_match,
                           success=bool(s.success))


class RunningBaseline:
    """Single-scalar EMA of reward — the REINFORCE baseline."""
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.value = 0.0
        self._init = False

    def update(self, reward: float) -> float:
        if not self._init:
            self.value = reward; self._init = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * reward
        return self.value


def reinforce_loss(policy: FluxPolicy, traj: Trajectory,
                   advantage: float) -> torch.Tensor:
    """-(reward - baseline) * sum_t log pi, recomputed WITH gradient.

    Replays the recorded latents through the transformer so the LoRA gets a
    gradient; the sampled next-latents and noise are held fixed (they define
    which action we are scoring), only the policy mean moves with the weights.
    """
    policy.load()
    dev, dt = policy.device, policy.dtype
    prompt_embeds, text_ids = policy._cond_for(traj.prompt)
    img_ids = traj.img_ids.to(dev)
    total_logp = 0.0
    for i in range(len(traj.latents)):
        x = traj.latents[i].to(dev, dt)
        x_next = traj.next_latents[i].to(dev).float()
        sigma = torch.tensor(traj.sigmas[i], device=dev)
        sigma_next = torch.tensor(traj.sigmas_next[i], device=dev)
        t = torch.tensor(traj.timesteps_raw[i], device=dev).expand(x.shape[0]).to(dt)
        # recompute velocity WITH grad (LoRA active, no no_grad here)
        v = policy._predict_velocity(x, t, prompt_embeds, text_ids, img_ids).float()
        logp = recompute_log_prob(x_next, x.float(), v, sigma, sigma_next, policy.eta)
        total_logp = total_logp + logp.sum()
    return -advantage * total_logp


@dataclass
class StepStats:
    reward: float
    baseline: float
    advantage: float
    success: bool
    loss: float
    placement: float = 0.0
    consistency: float = 0.0
    grid: float = 0.0
    shape_match: float = 0.0
    image: object = None       # the rollout's decoded image (for per-iter saving)


def train_step(policy: FluxPolicy, spec: GridSpec, cmd: Command,
               baseline: RunningBaseline, optimizer, *,
               reward_cfg: RewardConfig | None = None,
               placement_w: float = 0.5, consistency_w: float = 0.5,
               seed: int = 0) -> StepStats:
    """One RLVR update with the dense shaped+consistency reward.

    sample -> score (placement + image-consistency) -> REINFORCE -> step.
    The image is returned so the caller can save a per-iteration trajectory.
    """
    reward_cfg = reward_cfg or RewardConfig()
    gen = torch.Generator(device=policy.device).manual_seed(seed)
    traj = policy.rollout(build_prompt(cmd), generator=gen)
    rb = shaped_reward(spec, traj.image, cmd, reward_cfg,
                       placement_w=placement_w, consistency_w=consistency_w)
    b = baseline.update(rb.total)
    advantage = rb.total - b

    def stats(loss):
        return StepStats(rb.total, b, advantage, rb.success, loss,
                         placement=rb.placement, consistency=rb.consistency,
                         grid=rb.grid, shape_match=rb.shape_match, image=traj.image)

    optimizer.zero_grad()
    if abs(advantage) < 1e-8 or not policy.trainable_parameters:
        return stats(0.0)
    loss = reinforce_loss(policy, traj, advantage)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.trainable_parameters, 1.0)
    optimizer.step()
    return stats(float(loss.detach()))
