#!/usr/bin/env python3
"""Upload ablation PNGs to catbox.moe and save URL mapping."""
import os, json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

RUNS = Path(__file__).resolve().parent.parent / "runs"
ARMS = ["abl_arm1", "abl_arm2", "abl_arm3", "abl_arm4_qwen", "abl_arm4_vigorl", "abl_arm5", "abl_arm6"]
OUT = RUNS / "_image_urls.json"


def upload(path: Path) -> tuple[str, str]:
    """Upload a single file to catbox.moe, return (relative_key, url)."""
    key = f"{path.parent.name}/{path.name}"
    try:
        r = subprocess.run(
            ["curl", "-sf", "-F", "reqtype=fileupload",
             "-F", f"fileToUpload=@{path}",
             "https://catbox.moe/user/api.php"],
            capture_output=True, text=True, timeout=60,
        )
        url = r.stdout.strip()
        if url.startswith("https://"):
            print(f"  OK  {key} -> {url}")
            return key, url
        else:
            print(f"  FAIL {key}: {r.stdout[:120]}", file=sys.stderr)
            return key, ""
    except Exception as e:
        print(f"  ERR  {key}: {e}", file=sys.stderr)
        return key, ""


def main():
    files = []
    for arm in ARMS:
        arm_dir = RUNS / arm
        for f in sorted(arm_dir.glob("*.png")):
            files.append(f)

    print(f"Uploading {len(files)} images to catbox.moe ...")
    mapping = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(upload, f): f for f in files}
        for fut in as_completed(futs):
            key, url = fut.result()
            mapping[key] = url

    ok = sum(1 for v in mapping.values() if v)
    print(f"\nDone: {ok}/{len(files)} uploaded successfully.")
    OUT.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"Mapping saved to {OUT}")


if __name__ == "__main__":
    main()
