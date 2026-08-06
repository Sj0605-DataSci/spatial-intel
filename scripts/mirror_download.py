"""Direct mirror downloader — bypasses `hf download`'s Xet/metadata handshake that
breaks on hf-mirror.com. Fetches each repo file over plain HTTP from the mirror
(with resume) and places it in the HF cache snapshot layout so from_pretrained
finds it normally.

    python scripts/mirror_download.py Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import model_info

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def dl(url, dest: Path, expected: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == expected and expected > 0:
        print(f"  ✓ done {dest.name}"); return
    mode = "ab" if have else "wb"
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    t0 = time.time(); last = have
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); have += len(chunk)
            if time.time() - t0 > 3:
                sp = (have - last) / (time.time() - t0) / 1e6
                pct = 100 * have / expected if expected else 0
                print(f"    {dest.name}: {have/1e9:.2f}/{expected/1e9:.2f}GB "
                      f"({pct:.0f}%) {sp:.1f}MB/s", flush=True)
                t0 = time.time(); last = have
    print(f"  ✓ {dest.name} ({have/1e9:.2f}GB)")


def main():
    repo = sys.argv[1]
    info = model_info(repo, files_metadata=True)
    rev = info.sha
    snap = CACHE / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    files = sorted(info.siblings, key=lambda f: (f.size or 0))  # small first
    total = sum((f.size or 0) for f in files) / 1e9
    print(f"{repo} @ {rev[:8]} — {len(files)} files, {total:.1f}GB -> {snap}")
    for i, f in enumerate(files):
        url = f"{MIRROR}/{repo}/resolve/{rev}/{f.rfilename}"
        dest = snap / f.rfilename
        print(f"[{i+1}/{len(files)}] {f.rfilename} ({(f.size or 0)/1e9:.2f}GB)")
        for attempt in range(4):
            try:
                dl(url, dest, f.size or 0); break
            except Exception as e:
                print(f"    retry {attempt+1}: {type(e).__name__} {str(e)[:60]}")
                time.sleep(3)
    # write refs/main so from_pretrained resolves the snapshot
    refs = CACHE / f"models--{repo.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(rev)
    print(f"\nDONE. Load with from_pretrained('{repo}') (offline ok).")


if __name__ == "__main__":
    main()
