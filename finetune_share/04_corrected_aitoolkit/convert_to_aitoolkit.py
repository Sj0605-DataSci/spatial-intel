"""Convert our data/lora/ into ai-toolkit's EDIT-LoRA format (per the official BFL
FLUX.2-klein LoRA blog):

    out/
      reference/   input boards (triangle at START):  0000.png
      target/      output boards (triangle at TARGET): 0000.png + 0000.txt (caption)

ai-toolkit config points folder_path -> target/, control_path -> reference/.
Captions describe the TRANSFORMATION (the move instruction).

    python scripts/convert_to_aitoolkit.py --prompt-key prompt_coords --out data/aitoolkit
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
    ap.add_argument("--out", default="data/aitoolkit")
    ap.add_argument("--prompt-key", choices=["prompt_nl", "prompt_coords"], default="prompt_coords")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    ref = out / "reference"; tgt = out / "target"
    ref.mkdir(parents=True, exist_ok=True); tgt.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())

    n = 0
    for ex in manifest[args.split]:
        stem = f"{ex['id']:04d}"
        shutil.copy(src / ex["input"], ref / f"{stem}.png")        # reference = input board
        shutil.copy(src / ex["target_img"], tgt / f"{stem}.png")    # target = output board
        (tgt / f"{stem}.txt").write_text(ex[args.prompt_key])       # caption in target folder
        n += 1
    print(f"wrote {n} pairs -> {out}")
    print(f"  reference/ (inputs): {n} png")
    print(f"  target/    (outputs): {n} png + {n} txt captions")
    print(f"\nai-toolkit dataset config:")
    print(f"""  datasets:
    - folder_path:  "{tgt.resolve()}"
      control_path: "{ref.resolve()}"
      caption_ext: "txt"
      resolution: [512]""")
    print(f"sample caption: {(tgt / '0000.txt').read_text()}")


if __name__ == "__main__":
    main()
