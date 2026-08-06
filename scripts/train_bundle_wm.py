"""Train the latent world model (Qwen VAE + DIAMOND denoiser) on the game bundle.

Reads wm_index.json (27k train / 3k val transitions across 10 games, unified to
5 actions W/S/A/D/Space). Encodes frames with the Qwen VAE on the fly (cached per
epoch is too big for 30k frames), trains the latent DIAMOND denoiser to predict
obs_next latent from obs_t latent + action, and reports held-out VAL loss.

    python scripts/train_bundle_wm.py --res 512 --steps 20000 --out runs/bundle_wm
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diamond" / "src"))

import numpy as np
import torch
from PIL import Image

from latent_wm import load_vae, encode, make_latent_denoiser, LATENT_CH
from train_diamond_santa import Batch

BUNDLE = Path(__file__).resolve().parents[1] / "game_datasets_bundle"
N_ACTIONS = 5


@torch.no_grad()
def encode_batch(vae, rows, key, res, device):
    """Encode a list of transition rows' `key` frame -> stacked latents (B,C,h,w)."""
    lats = []
    for r in rows:
        img = Image.open(BUNDLE / r[key])
        lats.append(encode(vae, img, res, device)[0])
    return torch.stack(lats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/bundle_wm")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--change-weighted", action="store_true",
                    help="upweight loss in the region that changes (fixes copy/blur)")
    ap.add_argument("--change-weight", type=float, default=20.0)
    ap.add_argument("--cond-channels", type=int, default=256,
                    help="action-conditioning strength (bigger = harder to ignore action)")
    ap.add_argument("--balanced", action="store_true",
                    help="game-balanced sampling (each game equally likely; fixes tiny-game 0%)")
    ap.add_argument("--game", default=None,
                    help="Axis 1: train a SPECIALIST on one game only")
    ap.add_argument("--sigma-data", type=float, default=0.55,
                    help="per-game data-derived latent std (Axis 1)")
    ap.add_argument("--wide", action="store_true",
                    help="Axis 3: wider UNet [96,192,384,768]")
    ap.add_argument("--resume", default=None,
                    help="warm-start from a checkpoint (fine-tune for sharpness)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    idx = json.loads((BUNDLE / "wm_index.json").read_text())
    train, val = idx["train"], idx["val"]
    if args.game:                                  # Axis 1: single-game specialist
        train = [r for r in train if r["game"] == args.game]
        val = [r for r in val if r["game"] == args.game]
        print(f"SPECIALIST on {args.game}: sigma_data={args.sigma_data}")
    print(f"train {len(train)} | val {len(val)} | {N_ACTIONS} actions | res {args.res}")

    # game-balanced sampling: group train rows by game so we can sample game-first
    # (uniform over games, then a transition within) — otherwise hospital (8520)
    # is picked ~40x more than reduce_reuse_recycle (204) and tiny games never learn.
    from collections import defaultdict
    train_by_game = defaultdict(list)
    for r in train:
        train_by_game[r["game"]].append(r)
    games = sorted(train_by_game)
    if args.balanced:
        print(f"BALANCED sampling across {len(games)} games (each game equally likely)")

    print("loading Qwen VAE (frozen)...")
    vae = load_vae(device)
    # probe latent size
    h0 = encode(vae, Image.open(BUNDLE / train[0]["obs_t"]), args.res, device)
    print(f"latent shape per frame: {tuple(h0.shape)}  (C={LATENT_CH})")

    channels = [96, 192, 384, 768] if args.wide else None
    net = make_latent_denoiser(device, num_actions=N_ACTIONS,
                               change_weighted=args.change_weighted,
                               change_weight=args.change_weight,
                               cond_channels=args.cond_channels,
                               sigma_data=args.sigma_data,
                               channels=channels)
    print(f"change_weighted={args.change_weighted} (K={args.change_weight}) | "
          f"cond_channels={args.cond_channels} | sigma_data={args.sigma_data} | "
          f"channels={channels or '[64,128,256,512]'}")
    print(f"denoiser: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"RESUMED from {args.resume} (warm-start: keeps learned dynamics, "
              f"fine-tune for sharpness)")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    mask2 = torch.ones(args.batch, 2, dtype=torch.bool, device=device)

    def make_batch(rows):
        lt = encode_batch(vae, rows, "obs_t", args.res, device)      # (B,C,h,w)
        ln = encode_batch(vae, rows, "obs_next", args.res, device)
        obs = torch.stack([lt, ln], 1)                               # (B,2,C,h,w)
        act = torch.tensor([[r["action"], 0] for r in rows], device=device)
        m = torch.ones(len(rows), 2, dtype=torch.bool, device=device)
        return Batch(obs=obs, act=act, mask_padding=m)

    @torch.no_grad()
    def validate():
        net.eval(); losses = []
        for _ in range(args.val_batches):
            rows = [val[i] for i in rng.integers(len(val), size=args.batch)]
            losses.append(float(net(make_batch(rows))[0]))
        net.train()
        return float(np.mean(losses))

    log, t0 = [], time.time()
    def sample_rows():
        if args.balanced:
            # pick a game uniformly, then a random transition from it, per batch slot
            gs = rng.choice(len(games), size=args.batch)
            rows = []
            for gi in gs:
                gl = train_by_game[games[gi]]
                rows.append(gl[rng.integers(len(gl))])
            return rows
        return [train[i] for i in rng.integers(len(train), size=args.batch)]

    for step in range(args.steps):
        rows = sample_rows()
        loss, _ = net(make_batch(rows))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        lv = float(loss.detach())
        rec = dict(step=step, loss=lv, t=round(time.time() - t0, 1))
        if step % args.val_every == 0 or step == args.steps - 1:
            rec["val"] = validate()
            print(f"[{step:5d}/{args.steps}] train={lv:.4f} val={rec['val']:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
        elif step % 50 == 0:
            print(f"[{step:5d}/{args.steps}] train={lv:.4f} {time.time()-t0:.0f}s", flush=True)
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log))
        if args.save_every and step and step % args.save_every == 0:
            torch.save(net.state_dict(), out / f"denoiser_{step}.pt")

    torch.save(net.state_dict(), out / "denoiser_final.pt")
    print(f"done -> {out}/denoiser_final.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
