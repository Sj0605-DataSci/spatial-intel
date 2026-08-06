"""Convert our data/lora/ (manifest + inp/tgt PNGs) into the imagefolder +
metadata.jsonl format the official HF Kontext trainer expects.

Kontext dataset row = { file_name: <target img>, cond_image: <input img>, caption: <prompt> }
  --image_column      image        (target — the file_name column that imagefolder loads)
  --cond_image_column cond_image
  --caption_column    caption

Output: data/kontext/  with all images flat + metadata.jsonl
    python scripts/convert_to_kontext_dataset.py --prompt-key prompt_coords
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/lora")
    ap.add_argument("--out", default="data/kontext")
    ap.add_argument("--prompt-key", choices=["prompt_nl", "prompt_coords"], default="prompt_coords")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    rows = []
    for ex in manifest[args.split]:
        # copy target + input into the flat imagefolder with unique names
        tgt_name = f"tgt_{ex['id']:04d}.png"
        cond_name = f"cond_{ex['id']:04d}.png"
        shutil.copy(src / ex["target_img"], out / tgt_name)
        shutil.copy(src / ex["input"], out / cond_name)
        rows.append({
            "file_name": tgt_name,          # imagefolder loads this as the `image` column
            "cond_image": cond_name,        # --cond_image_column
            "caption": ex[args.prompt_key], # --caption_column
        })
    with open(out / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows + {len(rows)*2} images -> {out}")
    print(f"metadata.jsonl columns: file_name (target), cond_image (input), caption")
    print(f"sample row: {json.dumps(rows[0])}")


if __name__ == "__main__":
    main()
