"""Dry-run doubles: validate the RLVR loop wiring with no FLUX download.

DummyPolicy renders a real board image (so the actual reward/detector runs) but
chooses where to put the triangle stochastically: with probability `hit_rate` it
places it on the true target, otherwise on a random wrong block. This exercises
the full train_step path — verifiable reward, running baseline, advantage,
success logging — and lets us confirm the baseline tracks the reward and that
zero-advantage samples are skipped, without any GPU or model weights.
"""
from __future__ import annotations

import numpy as np

from ..grid import GridSpec, render_state, to_array, clamp_topleft
from .reinforce import build_prompt


class _DummyTraj:
    def __init__(self, image, prompt):
        self.image = image
        self.prompt = prompt
        # empty recorded tensors: the dummy never enters reinforce_loss because
        # we monkeypatch the loss out below.
        self.latents = []; self.next_latents = []; self.velocities = []
        self.log_probs = []; self.sigmas = []; self.sigmas_next = []


class DummyPolicy:
    device = "cpu"
    dtype = None

    def __init__(self, spec: GridSpec, hit_rate: float = 0.4, seed: int = 0):
        self.spec = spec
        self.hit_rate = hit_rate
        self._rng = np.random.default_rng(seed)

    def load(self):
        pass

    @property
    def trainable_parameters(self):
        return []

    def rollout(self, prompt, *, src_image=None, generator=None):
        # decode the target back out of the prompt isn't possible; instead the
        # train script passes the command, so we cheat via a module-level hook.
        cmd = _CURRENT_CMD["cmd"]
        if self._rng.random() < self.hit_rate:
            topleft = cmd.target
        else:
            m = self.spec.max_topleft
            topleft = clamp_topleft(self.spec, (int(self._rng.integers(0, m + 1)),
                                                int(self._rng.integers(0, m + 1))))
        img = to_array(render_state(self.spec, topleft))
        return _DummyTraj(img, prompt)


# train_step calls verifiable_reward(spec, traj.image, cmd) so we need the cmd
# available to the dummy rollout. The train script sets it each step.
_CURRENT_CMD = {"cmd": None}


def dummy_optimizer():
    class _Opt:
        def zero_grad(self): pass
        def step(self): pass
    return _Opt()
