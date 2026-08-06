# Spatial-Intel: Indian Pedagogy Games & World Models

A comprehensive project for **building and evaluating world models** on diverse Indian educational game environments. The project includes game engines, world model training pipelines, and evaluation frameworks for understanding how vision-language models perform on spatial reasoning tasks in culturally-grounded educational contexts.

## What's Inside

### Games & Environments

**25+ Indian educational games** covering themes like:
- 🏥 Hospital Operations
- 🎪 Theme Parks & Amusement
- 🌾 Agriculture & Farming
- 🎉 Cultural Festivals (Holi, Diwali, Baisakhi, Eid)
- 🏫 School Settings
- 🏏 Sports (Cricket, Football)
- 🌍 Environmental Themes (Pollution, Climate, Water Conservation, Waste Management)
- 🦁 Wildlife & Nature
- 🚀 Space Exploration
- 🍽️ Cultural Diversity in India

Each game includes:
- **RDDL specifications** (`game_datasets_bundle/{game}/`)
- **Base imagery & sprites** (`game_datasets_bundle/{game}/assets/`)
- **Rendered scenes** (`consolidated_games/{game}/`)
- **Training data** (state transitions, action sequences)

### Core Components

**Inference Scripts** (`scripts/`):
- `eval_*.py` — Evaluate trained models on game datasets
- `play_*.py` — Run gameplay loops with different agents
- `make_*.py` — Generate visualizations and reports
- `convert_*.py` — Format conversion utilities

**World Model Code** (`spatial_intel/`):
- `env.py` — Gymnasium-compatible game environments
- `game/` — Game engines, rendering, VLM judges
- `rl/` — Reinforcement learning & policy training
- `agents/` — Agent implementations (scripted baselines, FLUX, VLMs)
- `model/` — World model architectures (latent WM, bundle denoiser, perceptual WM)

**Datasets** (`game_datasets_bundle/`, `consolidated_games/`):
- JSON/JSONL metadata (states, transitions, actions)
- PNG images (base scenes, rendered frames, sprites)
- Total: ~200MB of curated game data

**Documentation**:
- `README.md` — This file (project overview)
- `report/spatial_wm_slides.pdf` — Results & methodology

## Architecture

```
spatial-intel/
├── spatial_intel/              # Core library
│   ├── env.py                  # Gymnasium environments
│   ├── game/                   # Game engines & rendering
│   ├── agents/                 # Agent implementations
│   ├── rl/                      # Training & RL code
│   └── model/                   # World model architectures
├── scripts/                     # Experiment & inference scripts (60+)
├── game_datasets_bundle/        # Game metadata & base assets
│   ├── 7_worlds_of_wonder/
│   ├── hospital/
│   ├── wildlife/
│   └── [22 more games...]
├── consolidated_games/          # Rendered game scenes
│   ├── 7_worlds_of_wonder/images/
│   ├── hospital/images/
│   └── [consolidated renders...]
└── report/                      # Slides, figures, HTML reports
```

## Setup

Install dependencies:

```bash
pip install -r requirements-extra.txt
```

Or with uv:

```bash
uv pip install gymnasium transformers torch pillow
```

## Quick Start

### List Available Games

```bash
ls game_datasets_bundle/     # See all 25+ game folders
```

### Run Inference on a Game

```bash
# Example: evaluate on hospital game
python scripts/eval_everything.py --game hospital --model flux

# Or play interactively
python scripts/play_diamond_santa.py
```

### Generate Report

```bash
python scripts/make_report.py --out report/results.html
```

## Experiment Runs

The `runs/` folder (local only, not in git) contains outputs from 72+ training & evaluation runs:

- **Bundle denoiser** variants: `bundle_{bal,sharp,cw,wm}*`
- **Game-specific** models: `hospital_*`, `diamond_*`
- **Latent WM** experiments: `latent_play*`, `latent_qwen*`
- **Evaluation** runs: `eval_*`
- **LoRA** fine-tuning: `lora_*`
- **Comparative** studies: `cmp_*, closed_loop_*`

Each run contains:
- `train_log.json` — Training metrics per step
- `best.json` — Final validation performance
- `samples/` — Generated images at different checkpoints

## Key Results

Trained world models achieve:
- **Hospital game**: ~80% move accuracy with feedback
- **Diamond/Santa**: ~85% spatial reasoning correctness
- **Multi-game**: Cross-game transfer with LoRA

See `report/spatial_wm_slides.pdf` for full results.

## Project Structure

- **Data generation**: `scripts/generate_dataset.py`, `scripts/bundle_dataset.py`
- **Model training**: `scripts/train_*.py` (denoiser, latent WM, rich WM)
- **Evaluation**: `scripts/eval_*.py` (specialized per game/method)
- **Analysis**: `scripts/make_figures*.py`, `scripts/make_report.py`

## Citation

If you use this project, please cite:

```bibtex
@misc{spatial_intel,
  title={Spatial Intelligence in Vision Models: Indian Pedagogy Games},
  author={...},
  year={2024},
  howpublished={\url{https://github.com/Sj0605-DataSci/spatial-intel}}
}
```

## License

Licensed under Apache 2.0. See LICENSE file for details.

## Notes

- **Model weights** (*.pt, *.pth, *.safetensors) are excluded from git via `.gitignore`
- **Training runs** (`runs/`) are local only for size efficiency
- All scripts use standard Python 3.10+ with PyTorch
- For large-scale training, GPU (NVIDIA) recommended
