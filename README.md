# spatial-intel

A synthetic benchmark + Gymnasium environment to probe **spatial intelligence in
image models**: can a model take a grid image with a solid red triangle in one
block and *render the triangle in a new block when told a direction* — i.e. does
it understand "move to the top-right by 2"?

The agent under test is **FLUX.2-klein-4B** (Black Forest Labs), used as an
image-to-image editor. The env scores each attempt and feeds a correction back,
so we can measure **how many feedback iterations the model needs to get it
right** vs. scripted baselines.

## Layout

```
spatial_intel/
  grid.py        # board geometry, directions, triangle rendering, object detection
  reward.py      # score an attempt vs the commanded target (success + shaped reward)
  dataset.py     # synthetic (input, instruction, output) sample generation
  env.py         # SpatialMoveEnv (Gymnasium): image/cell action, feedback reloop
  agents/
    base.py      # Agent interface (act(obs, info) -> action)
    scripted.py  # OracleAgent / NoisyAgent / RandomAgent (baselines, no GPU)
    flux.py      # FluxAgent — FLUX.2-klein-4B img2img (lazy-loaded)
scripts/
  generate_dataset.py
  run_env.py
data/            # generated datasets
```

## Setup (uses the shared `3dgit/.venv`)

`torch (cu130)`, `numpy`, `pillow` are already in that venv. Add the rest with **uv**:

```bash
cd ../3dgit
uv pip install --python .venv/bin/python gymnasium          # core (data + env)

# FLUX agent only (heavy; ~13GB VRAM, open weights / Apache-2.0):
uv pip install --python .venv/bin/python \
    "git+https://github.com/huggingface/diffusers.git" \
    transformers accelerate safetensors sentencepiece
```

Run everything with `../3dgit/.venv/bin/python`.

## Use

**1. Generate the dataset** — input (start) + output (target) PNG pairs + manifest:

```bash
python scripts/generate_dataset.py --out data/v1 --n 128            # random starts
python scripts/generate_dataset.py --out data/center --n 64 --center-start
```

`--center-start` always starts at the centre block (3,3) (the brief's example),
so you isolate *direction* generalisation from *position*.

**2. Validate the env / set baselines** (no GPU):

```bash
python scripts/run_env.py --agent oracle --episodes 50 --action-mode cell   # ~100% in 1 step
python scripts/run_env.py --agent noisy  --episodes 50 --max-steps 8        # converges via feedback
python scripts/run_env.py --agent random --episodes 50                       # floor
```

Reference numbers (8×8 board): oracle 100% @ 1 iter, noisy 100% @ ~2.1 iters,
random ~22% success.

**3. Run FLUX.2-klein-4B** (downloads weights on first call):

```bash
python scripts/run_env.py --agent flux --episodes 5 --max-steps 6 \
    --save-rollouts runs/flux_demo
```

Each episode: FLUX edits the board → env reads back where the triangle landed →
reward + a textual correction → FLUX re-edits with that feedback, up to
`--max-steps`. `--save-rollouts` dumps per-step PNGs so you can watch it
(hopefully) converge.

## How it works

- **Board** (`GridSpec`): N×N cells (default 8), object = `object_size²` block
  (default 2×2). Coords are `(col, row)`, row increases downward; "up"/"top" =
  row decreases. 8 directions incl. diagonals, with natural-language aliases
  ("top right" → `up_right`).
- **Reward** (`score_attempt`): detects the reddest blob, maps its centroid back
  to a block, compares to the commanded target. `success` = exact block match;
  otherwise a shaped reward in [0,1] from cell-distance + a "moved the right
  direction" cosine term. Tolerant enough for messy model output.
- **Env** (`SpatialMoveEnv`): two action modes — `image` (FLUX's HxWx3 output)
  and `cell` (`[col,row]`, for fast model-free experiments). The episode is a
  feedback reloop: on a miss it keeps the command, draws a hint arrow on the
  obs, and puts a textual critique in `info["feedback"]`.

## Notes

- The FLUX pipeline class is resolved defensively (`Flux2KleinPipeline` →
  `Flux2Pipeline` → auto img2img) so a diffusers naming drift won't break it.
  If `from_pretrained` fails, you likely need the **git** build of diffusers.
- vLLM / SGLang are LLM serving engines and do **not** apply to FLUX (a diffusion
  model) — the relevant GB10 optimisation here is the bf16 + `cu130` torch
  already in the venv, plus optional `enable_model_cpu_offload()` / `torch.compile`.
