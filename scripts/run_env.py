#!/usr/bin/env python
"""Run an agent in SpatialMoveEnv and report the headline metric:
how many feedback iterations it needs to land the triangle on target.

Examples
--------
    # validate env + reward (should solve in 1 step, 100% success):
    python scripts/run_env.py --agent oracle --episodes 50

    # exercise the feedback reloop with an imperfect mover:
    python scripts/run_env.py --agent noisy --episodes 50 --max-steps 8

    # floor baseline:
    python scripts/run_env.py --agent random --episodes 50

    # the real thing (downloads FLUX.2-klein-4B on first use):
    python scripts/run_env.py --agent flux --episodes 5 --action-mode image \
        --save-rollouts runs/flux_demo
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_intel.grid import GridSpec, from_array
from spatial_intel.env import SpatialMoveEnv
from spatial_intel.agents import OracleAgent, NoisyAgent, RandomAgent, make_flux


def build_agent(name, spec, action_mode):
    if name == "oracle":
        return OracleAgent(spec, action_mode)
    if name == "noisy":
        return NoisyAgent(spec, action_mode)
    if name == "random":
        return RandomAgent(spec, action_mode)
    if name == "flux":
        return make_flux(spec)  # image mode only
    raise SystemExit(f"unknown agent {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="oracle",
                    choices=["oracle", "noisy", "random", "flux"])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--action-mode", default="cell", choices=["cell", "image"])
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--center-start", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-rollouts", default=None,
                    help="dir to dump per-step PNGs of each episode")
    args = ap.parse_args()

    if args.agent == "flux":
        args.action_mode = "image"  # FLUX produces images

    spec = GridSpec(n=args.grid)
    env = SpatialMoveEnv(spec, action_mode=args.action_mode,
                         max_steps=args.max_steps, center_start=args.center_start,
                         seed=args.seed)
    agent = build_agent(args.agent, spec, args.action_mode)

    steps_to_success, successes, returns = [], 0, []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        agent.reset()
        ep_return, solved_at = 0.0, None
        for t in range(args.max_steps):
            action = agent.act(obs, info)
            obs, reward, term, trunc, info = env.step(action)
            ep_return += reward
            if args.save_rollouts:
                d = Path(args.save_rollouts) / f"ep{ep:03d}"
                d.mkdir(parents=True, exist_ok=True)
                from_array(obs["image"]).save(d / f"step{t:02d}_r{reward:.2f}.png")
            if term:
                solved_at = t + 1
                break
            if trunc:
                break
        if solved_at is not None:
            successes += 1
            steps_to_success.append(solved_at)
        returns.append(ep_return)

    n = args.episodes
    print(f"\nAgent={args.agent}  mode={args.action_mode}  episodes={n}  "
          f"max_steps={args.max_steps}")
    print(f"  success rate     : {successes}/{n} = {successes/n:.1%}")
    if steps_to_success:
        print(f"  iters to success : mean={statistics.mean(steps_to_success):.2f}  "
              f"median={statistics.median(steps_to_success)}  "
              f"min={min(steps_to_success)}  max={max(steps_to_success)}")
    print(f"  mean ep return   : {statistics.mean(returns):.3f}")
    if args.save_rollouts:
        print(f"  rollouts saved to {args.save_rollouts}/")


if __name__ == "__main__":
    main()
