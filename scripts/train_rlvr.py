"""Vanilla RLVR (REINFORCE) finetune of a FLUX.2 LoRA on the spatial-move task.

Per the agreed first-run scope: easy task (center-start, 1-step moves) so the
model lands on-target often enough to get a mix of +1/-1 rewards; strict +1/-1
verifiable reward; LoRA on FLUX.2-klein-base-4B; single running-mean baseline.

    python scripts/train_rlvr.py --steps 200 --out runs/rlvr1

Heavy: downloads base-4B (~13GB) and needs a GPU. Use --dry-run to validate the
loop wiring on a tiny synthetic policy with no model download.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_intel.grid import GridSpec, apply_move, clamp_topleft, DIRECTION_NAMES, max_steps_for
from spatial_intel.reward import RewardConfig
from spatial_intel.rl.reinforce import Command, RunningBaseline, train_step


def sample_command(spec: GridSpec, rng, *, center_start=True, max_step=1) -> Command:
    center = (spec.max_topleft // 2, spec.max_topleft // 2)
    for _ in range(200):
        start = center if center_start else (
            int(rng.integers(0, spec.max_topleft + 1)),
            int(rng.integers(0, spec.max_topleft + 1)))
        direction = DIRECTION_NAMES[int(rng.integers(len(DIRECTION_NAMES)))]
        hi = min(max_step, max_steps_for(spec, start, direction))
        if hi < 1:
            continue
        steps = int(rng.integers(1, hi + 1))
        target = clamp_topleft(spec, apply_move(spec, start, direction, steps))
        return Command(start, direction, steps, target)
    return Command(center, "right", 1, clamp_topleft(spec, apply_move(spec, center, "right", 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="runs/rlvr1")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B",
                    help="use FLUX.2-klein-4B for the cached distilled model")
    ap.add_argument("--num-denoise-steps", type=int, default=20,
                    help="20 keeps images coherent; 4-8 produces noise under the SDE")
    ap.add_argument("--eta", type=float, default=0.1,
                    help="SDE exploration noise; >0.2 degrades to static at low steps")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--max-step", type=int, default=1, help="max cells per move (easy=1)")
    ap.add_argument("--placement-w", type=float, default=0.5, help="weight of where-reward")
    ap.add_argument("--consistency-w", type=float, default=0.5, help="weight of image-match reward")
    ap.add_argument("--full-task", action="store_true", help="random starts, multi-step")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="wire-check, no FLUX download")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    spec = GridSpec()
    rng = np.random.default_rng(0)
    reward_cfg = RewardConfig()
    baseline = RunningBaseline(momentum=0.9)

    if args.dry_run:
        from spatial_intel.rl._dryrun import DummyPolicy, dummy_optimizer
        policy = DummyPolicy(spec)
        optimizer = dummy_optimizer()
    else:
        import torch
        from spatial_intel.rl.policy import FluxPolicy
        policy = FluxPolicy(model_id=args.model_id, num_steps=args.num_denoise_steps,
                            eta=args.eta, lora_rank=args.lora_rank)
        policy.load()
        optimizer = torch.optim.AdamW(policy.trainable_parameters, lr=args.lr)

    # per-iteration trajectory images so you can WATCH the model learn
    traj_dir = out / "trajectory"; traj_dir.mkdir(exist_ok=True)
    from PIL import Image

    log = []
    succ_window = []
    for step in range(args.steps):
        cmd = sample_command(spec, rng, center_start=not args.full_task,
                             max_step=args.max_step)
        if args.dry_run:
            from spatial_intel.rl import _dryrun
            _dryrun._CURRENT_CMD["cmd"] = cmd
        stats = train_step(policy, spec, cmd, baseline, optimizer,
                           reward_cfg=reward_cfg, placement_w=args.placement_w,
                           consistency_w=args.consistency_w, seed=step)
        succ_window.append(stats.success)
        if len(succ_window) > 50:
            succ_window.pop(0)
        rec = dict(step=step, reward=round(stats.reward, 4), baseline=round(stats.baseline, 3),
                   advantage=round(stats.advantage, 3), success=stats.success,
                   placement=round(stats.placement, 3), consistency=round(stats.consistency, 3),
                   grid=round(stats.grid, 3), shape_match=round(stats.shape_match, 3),
                   loss=round(stats.loss, 4), success_rate_50=round(float(np.mean(succ_window)), 3),
                   cmd=f"{cmd.direction}x{cmd.steps} {cmd.start}->{cmd.target}")
        log.append(rec)
        # save EVERY iteration's image (the learning trajectory)
        if stats.image is not None and not args.dry_run:
            Image.fromarray(stats.image).save(traj_dir / f"step{step:04d}.png")
        print(f"[{step:4d}] R={stats.reward:.3f} place={stats.placement:.2f} "
              f"shape={stats.shape_match:.2f} grid={stats.grid:.2f} "
              f"adv={stats.advantage:+.3f} loss={stats.loss:.3f} "
              f"hit={int(stats.success)} succ@50={rec['success_rate_50']:.2f}", flush=True)
        # write log incrementally so you can tail it live
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        if not args.dry_run and args.save_every and step and step % args.save_every == 0:
            policy._pipe.transformer.save_lora_adapter(str(out / f"lora_step{step}"))

    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    if not args.dry_run:
        policy._pipe.transformer.save_lora_adapter(str(out / "lora_final"))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
