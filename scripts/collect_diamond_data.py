"""Collect (frame, action, next_frame) rollout data for the toy DIAMOND
prototype on Santa's Gift Journey, rendered via `SantaEnv`'s deterministic
cutpaste-style compositor (`spatial_intel/game/santa_env.py`).

Gemini image-editing (gemini-3.1-flash-image) was tried first and rejected:
even on plain valid single-cell moves it misplaced the S badge by several
cells (see plan doc, "Tested and rejected: Gemini as frame renderer" section)
— unusable as a ground-truth label source. The deterministic renderer is
exact and ~4 orders of magnitude faster (~1ms vs ~14s), so it's used instead.

Dataset size is DERIVED from the game's state space (see
`/home/sanyam-0605/.claude/plans/sprightly-soaring-hippo.md` "Dataset scope"
section), not picked arbitrarily:
  - 256 "movement" transitions: every one of the 64 board cells x 4 actions,
    starting fresh each time (ground-truth reset to that cell) — exhaustively
    covers pure movement dynamics, including off-board/blocked moves.
  - ~210 transitions from 6 full near-optimal episodes (~35 steps each,
    nearest-neighbor present/house/stable order) — covers collect/deliver/
    rest/game-over events that pure single-step resets never trigger.
  - total ~466 transitions.

Every stage (env reset/step, render, disk write) is timed with `time.time()`,
same discipline as `scripts/world_model_bench.py`. Writes:
  data/diamond_toy/frames/{idx:05d}_before.png
  data/diamond_toy/frames/{idx:05d}_after.png
  data/diamond_toy/manifest.json   (per-transition metadata)
  data/diamond_toy/timings.json    (latency benchmark: per-stage stats)

Usage:
  python scripts/collect_diamond_data.py --out data/diamond_toy
  python scripts/collect_diamond_data.py --out data/diamond_toy --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from spatial_intel.game.santa_env import SantaEnv, ACTIONS, PRESENTS, HOUSES, STABLE

ORDER = ["present1", "house_a", "present2", "house_b",
         "present3", "house_c", "present4", "house_d", "reinder_stable"]
ALL_POINTS = {**PRESENTS, **HOUSES, **STABLE}


def manhattan_path(cur, target):
    """Greedy 4-connected path from cur to target, returns list of move names."""
    moves = []
    cx, cy = cur
    tx, ty = target
    while cx != tx:
        moves.append("right" if tx > cx else "left")
        cx += 1 if tx > cx else -1
    while cy != ty:
        moves.append("down" if ty > cy else "up")
        cy += 1 if ty > cy else -1
    return moves


def episode_moves():
    """Full near-optimal episode: visit ORDER via greedy manhattan paths."""
    moves = []
    cur = (1, 1)
    for name in ORDER:
        tgt = ALL_POINTS[name]
        moves.extend(manhattan_path(cur, tgt))
        cur = tgt
    return moves


def movement_sweep_transitions():
    """Every (cell, action) pair on the 8x8 board — 256 transitions."""
    transitions = []
    for cx in range(1, 9):
        for cy in range(1, 9):
            for move in ACTIONS:
                transitions.append(dict(kind="movement", santa_cell=(cx, cy), move=move))
    return transitions


def episode_transitions(n_episodes: int):
    """n_episodes full near-optimal playthroughs -> transitions in sequence."""
    transitions = []
    for ep in range(n_episodes):
        transitions.append(dict(kind="episode_start", episode=ep))
        moves = episode_moves()
        for move in moves:
            transitions.append(dict(kind="episode", episode=ep, move=move))
    return transitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/diamond_toy")
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap total transitions (smoke-test runs)")
    args = ap.parse_args()

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = SantaEnv()

    plan = movement_sweep_transitions() + episode_transitions(args.n_episodes)
    if args.limit:
        plan = plan[: args.limit]
    print(f"planned transitions: {len(plan)}")

    manifest = []
    timings = {"env_reset_step_s": [], "render_s": [], "disk_write_s": []}

    idx = 0
    cur_info = None
    for step in plan:
        if step["kind"] == "episode_start":
            t0 = time.time()
            before_obs, cur_info = env.reset()
            timings["env_reset_step_s"].append(time.time() - t0)
            continue

        move = step["move"]
        if step["kind"] == "movement":
            t0 = time.time()
            _, cur_info = env.reset(options={"santa_cell": step["santa_cell"]})
            timings["env_reset_step_s"].append(time.time() - t0)

        t0 = time.time()
        before_obs = env.render()
        timings["render_s"].append(time.time() - t0)

        t0 = time.time()
        after_obs, reward, terminated, truncated, next_info = env.step(ACTIONS.index(move))
        timings["env_reset_step_s"].append(time.time() - t0)

        t0 = time.time()
        after_render = env.render()
        timings["render_s"].append(time.time() - t0)

        blocked = tuple(cur_info["santa_cell"]) == tuple(next_info["santa_cell"])

        rec = dict(
            idx=idx, kind=step["kind"], move=move, blocked=blocked,
            santa_cell_before=list(cur_info["santa_cell"]),
            santa_cell_after=list(next_info["santa_cell"]),
            reward=reward, terminated=bool(terminated), truncated=bool(truncated),
        )

        t0 = time.time()
        Image.fromarray(before_obs).save(frames_dir / f"{idx:05d}_before.png")
        Image.fromarray(after_render).save(frames_dir / f"{idx:05d}_after.png")
        timings["disk_write_s"].append(time.time() - t0)

        manifest.append(rec)
        cur_info = next_info
        idx += 1

        print(f"[{idx}/{len(plan)}] {step['kind']:9} move={move:5} "
              f"{str(step.get('santa_cell', '')):>8} -> {next_info['santa_cell']} "
              f"blocked={blocked}", flush=True)

        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def stage_stats(vals):
        if not vals:
            return {}
        arr = np.array(vals)
        return dict(n=len(arr), median_s=float(np.median(arr)),
                    p95_s=float(np.percentile(arr, 95)),
                    total_s=float(arr.sum()))

    timing_summary = {k: stage_stats(v) for k, v in timings.items()}
    (out / "timings.json").write_text(json.dumps(timing_summary, indent=2))

    print(f"\ndone: {len(manifest)} transitions")
    print(f"-> {out}/manifest.json, {out}/timings.json")


if __name__ == "__main__":
    main()
