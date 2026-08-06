"""Memory-guarded probe for latent-denoiser configs.

A watchdog thread polls process RSS + GPU allocation; if EITHER crosses --limit-gb
it prints the offending config and hard-kills this process (os._exit) BEFORE the
OS OOM-killer can thrash the whole machine. Each config runs in sequence; a
config that survives 3 train steps prints its params / ms / peak GB.

    python scripts/probe_latent_mem.py --limit-gb 90
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

DIAMOND = Path(__file__).resolve().parents[2] / "diamond" / "src"
sys.path.insert(0, str(DIAMOND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from models.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from train_diamond_santa import Batch

_current = {"name": "startup"}


def _rss_gb():
    # resident set size from /proc (Linux), GB
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return 0.0


def watchdog(limit_gb, stop):
    while not stop.is_set():
        gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        rss = _rss_gb()
        if gpu > limit_gb or rss > limit_gb:
            print(f"\n!!! KILL: config '{_current['name']}' exceeded {limit_gb}GB "
                  f"(gpu={gpu:.1f} rss={rss:.1f}) — hard-killing before OOM.\n", flush=True)
            os._exit(137)
        time.sleep(0.05)


def probe(name, channels, depths, attn, batch, dev):
    _current["name"] = name
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    inner = InnerModelConfig(img_channels=32, num_steps_conditioning=1, cond_channels=256,
                             depths=depths, channels=channels, attn_depths=attn, num_actions=4)
    net = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3)).to(dev)
    net.setup_training(SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20))
    p = sum(x.numel() for x in net.parameters()) / 1e6
    obs = torch.randn(batch, 2, 32, 128, 128, device=dev)
    act = torch.zeros(batch, 2, dtype=torch.long, device=dev)
    mask = torch.ones(batch, 2, dtype=torch.bool, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        loss, _ = net(Batch(obs=obs, act=act, mask_padding=mask))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  OK  {name}: {p:.1f}M params | {(time.time()-t0)/3*1000:.0f}ms/step | "
          f"{peak:.1f}GB peak | attn={attn}", flush=True)
    del net, opt, obs; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-gb", type=float, default=90.0)
    ap.add_argument("--batch", type=int, default=9)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    stop = threading.Event()
    threading.Thread(target=watchdog, args=(args.limit_gb, stop), daemon=True).start()
    print(f"watchdog armed at {args.limit_gb}GB | latent grid 128x128 (32ch) | batch {args.batch}\n")

    # 4 U-Net levels: 128->64->32->16. Attention only at deep (small) levels is cheap.
    configs = [
        ("conv-only",          [128, 256, 384, 512], [2, 2, 2, 2], [False, False, False, False]),
        ("attn-deepest-16",    [128, 256, 384, 512], [2, 2, 2, 2], [False, False, False, True]),
        ("attn-deep2-32-16",   [128, 256, 384, 512], [2, 2, 2, 2], [False, False, True, True]),
        ("attn-top-128 (BAD)", [128, 256, 384, 512], [2, 2, 2, 2], [True, False, False, False]),
    ]
    for name, ch, dp, at in configs:
        try:
            probe(name, ch, dp, at, args.batch, dev)
        except RuntimeError as e:
            print(f"  OOM {name}: {str(e)[:70]}", flush=True)
            torch.cuda.empty_cache()
    stop.set()
    print("\ndone.")


if __name__ == "__main__":
    main()
