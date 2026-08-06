"""Dataset loader + train/val split for the game_datasets_bundle.

Unifies 10 games under ONE action space keyed by the control key
(W/A/S/D/Space -> ids 0..4) — every game's dozens of 'base actions' are just
Space (context action on the cell), so 5 ids cover all of them.

Builds an index of (obs_t, action_id, obs_next, game) rows from every
transitions.jsonl, then a deterministic train/val split.

    python scripts/bundle_dataset.py --build --val-frac 0.1
"""
import argparse
import json
import random
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
CONTROL_TO_ID = {"W": 0, "S": 1, "A": 2, "D": 3, "Space": 4}
N_ACTIONS = 5
INDEX = BUNDLE / "wm_index.json"


def build_index():
    games = sorted(p.name for p in BUNDLE.iterdir()
                   if p.is_dir() and (p / "transitions.jsonl").exists())
    rows = []
    per_game = {}
    for g in games:
        n = 0
        with open(BUNDLE / g / "transitions.jsonl") as f:
            for line in f:
                t = json.loads(line)
                ctrl = t.get("control")
                if ctrl not in CONTROL_TO_ID:
                    continue
                rows.append({
                    "game": g,
                    "action": CONTROL_TO_ID[ctrl],
                    "obs_t": f"{g}/{t['obs_t']}",
                    "obs_next": f"{g}/{t['obs_next']}",
                    "moved": bool(t.get("effects", {}).get("moved_player")),
                })
                n += 1
        per_game[g] = n
    return rows, per_game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, per_game = build_index()
    print(f"total unified transitions: {len(rows)}")
    for g, n in per_game.items():
        print(f"  {g:22} {n}")

    # STRATIFIED split: split EACH game separately so every game gets exactly
    # val_frac in val (no game drifts off-ratio), then combine + shuffle.
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r)

    train, val = [], []
    print(f"\nstratified split (val_frac={args.val_frac:.0%} per game):")
    print(f"  {'game':22} {'train':>6} {'val':>5}  {'val %':>6}")
    for g in sorted(by_game):
        grows = by_game[g]
        random.Random(hash((args.seed, g)) & 0xffffffff).shuffle(grows)
        k = round(len(grows) * args.val_frac)
        gva, gtr = grows[:k], grows[k:]
        val += gva; train += gtr
        print(f"  {g:22} {len(gtr):>6} {len(gva):>5}  {100*len(gva)/len(grows):>5.1f}%")

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)

    def mv(rs):
        m = sum(r["moved"] for r in rs)
        return f"{m} moved / {len(rs)-m} act-or-blocked"
    print(f"\ntotals: train={len(train)} ({mv(train)}) | val={len(val)} ({mv(val)})")

    if args.build:
        INDEX.write_text(json.dumps(
            {"n_actions": N_ACTIONS, "control_to_id": CONTROL_TO_ID,
             "train": train, "val": val}, indent=0))
        print(f"-> wrote {INDEX} ({INDEX.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
