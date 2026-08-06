# Spatial Intelligence - Runs Documentation

This document describes all experiments in the `runs/` folder. Each run contains:
- **Training logs**: `train_log.json` (training metrics per step)
- **Generated images**: `samples/` folder (inference outputs at different training steps)
- **Model weights**: `*.pt`, `*.pth`, `*.safetensors` files (excluded from git, see `.gitignore`)
- **Evaluation results**: `best.json`, `summary.json`

## Folder Structure

```
runs/
├── abl_arm*          # Ablation studies (architecture variants)
├── bundle_*          # Bundle denoiser experiments
├── eval_*            # Evaluation runs on different datasets
├── latent_*          # Latent space experiments
├── hospital_*        # Hospital scene experiments
├── diamond_*         # Diamond shape experiments
├── lora_*            # LoRA fine-tuning experiments
├── *_smoke           # Smoke tests (small scale)
├── *_eval            # Evaluation on test sets
├── aitoolkit_out/    # Diffusion outputs with checkpoints
└── [many more...]
```

## Experiment Categories

### 1. **Ablation Studies** (`abl_arm*`)
Architecture ablation experiments testing different model configurations.

| Run | Purpose | Output |
|-----|---------|--------|
| `abl_arm1-6` | Baseline architecture variants | `samples/`, `train_log.json` |
| `abl_arm4_qwen` | Qwen-based variant | Images + training metrics |
| `abl_arm4_vigorl` | VigoRL-based variant | Images + training metrics |

### 2. **Bundle Denoiser Experiments** (`bundle_*`)
Denoiser training with different configurations (bal=balanced, cw=class-weighted, sharp=sharp attention, wm=weighted method).

| Run | Purpose | Scale | Output |
|-----|---------|-------|--------|
| `bundle_bal*` | Balanced class weighting | Full (eval) + Smoke | Train logs + `best.json` |
| `bundle_cw*` | Class-weighted denoiser | Full (eval) + Smoke | Train logs + samples |
| `bundle_sharp*` | Sharp attention mechanism | Full (eval) + Smoke | Train logs + samples |
| `bundle_wm*` | Weighted method approach | Full (eval) + Smoke | Train logs + samples |
| `bundle_wm_1k*` | WM method with 1k steps | Full (eval) + Smoke | Train logs + samples |

**Key Outputs**: Each has `*_smoke` (quick test) and `*_eval` (full evaluation) variants.

### 3. **Scene-Specific Experiments**

#### Hospital Scene (`hospital_*`)
Experiments focused on hospital environment simulation and noise robustness.

| Run | Purpose | Output |
|-----|---------|--------|
| `hospital_base` | Baseline | Training metrics |
| `hospital_drift_free` | Without drift | Training metrics |
| `hospital_full_game` | Full game loop | Training metrics |
| `hospital_noiseaug` | Noise augmentation | `train_log.json` + samples |
| `hospital_xpred_noiseaug` | Cross-prediction + noise aug | `train_log.json` + `best.json` |
| `everything_hospital` | Full kitchen sink | `train_log.json` + samples |
| `perc_hospital` | Perceptual loss | Training metrics + samples |
| `rich_hospital_800` | High-res (800px) | Training metrics |
| `spec_hospital` | Specialized setup | Training metrics |
| `spec_hospital_800` | Spec at 800px | Training logs + samples |

#### Diamond Scene (`diamond_*`)
Experiments with diamond/triangle shape objects.

| Run | Purpose | Output |
|-----|---------|--------|
| `diamond_dev` | Development | Training metrics |
| `diamond_play` | Gameplay loop | Training logs + samples |
| `diamond_play_smoke` | Quick test | Training logs |
| `diamond_play_video` | With video output | Video + logs |
| `diamond_santa` | Santa variant | Training metrics |
| `diamond_santa_smoke` | Quick smoke test | Training logs |

### 4. **Latent Space Experiments** (`latent_*`)
Experiments exploring learned latent representations.

| Run | Purpose | Output |
|-----|---------|--------|
| `latent_play` | Latent space gameplay | Train logs + `best.json` |
| `latent_play_smoke` | Quick latent test | Samples + logs |
| `latent_play_video` | With video outputs | Video + logs |
| `latent_qwen` | Qwen-based latent | Train logs + samples |
| `latent_qwen_smoke` | Quick Qwen test | Samples |
| `latent_qwen_video` | Qwen with video | Video outputs |

### 5. **Fine-tuning & Adaptation** (`lora_*`)
LoRA (Low-Rank Adaptation) fine-tuning experiments for efficient model adaptation.

| Run | Purpose | Output |
|-----|---------|--------|
| `lora_coords` | Coordinate-focused LoRA | Training logs |
| `lora_coords_align` | Aligned coordinates | Training metrics |
| `lora_gligen` | GLIGEN-based LoRA | Training logs + `optimizer.pt` |
| `lora_nl` | Natural language LoRA | Training metrics |
| `lora_santa_weighted` | Santa with LoRA | Training logs |
| `lora_weighted*` | Weighted variants | Training logs |

### 6. **Evaluation Runs** (`eval_*`)
Evaluation on various datasets and metrics.

| Run | Purpose | Key Output |
|-----|---------|-----------|
| `eval_base` | Baseline evaluation | `best.json` |
| `eval_coords` | Coordinate accuracy | Metrics |
| `eval_diamond_santa` | Diamond/Santa scenes | Metrics + samples |
| `eval_everything_hospital` | Full hospital eval | `best.json` + metrics |
| `eval_perc_hospital` | Perceptual loss eval | Metrics |
| `eval_santa` | Santa-specific eval | Metrics |
| `eval_spec_hospital*` | Specialized hospital eval | `best.json` + detailed samples |
| `eval_weighted` | Weighted method eval | Metrics |

### 7. **Comparative Studies** (`cmp_*`, `closed_loop_*`)
Comparisons between different approaches and closed-loop feedback experiments.

| Run | Purpose | Output |
|-----|---------|--------|
| `cmp_all` | Compare all methods | Comparison metrics |
| `cmp_hospital` | Hospital comparisons | Metrics |
| `closed_loop_3way` | 3-way feedback loop | Convergence analysis |
| `closed_loop_compare` | Loop method comparison | Metrics |
| `closed_loop_cvc` | CVC variant | Training logs |

### 8. **Bias & Robustness Studies** (`bias_*`)
Testing robustness to different bias levels.

| Run | Purpose | Output |
|-----|---------|--------|
| `bias_s15` | Bias strength 0.15 | Training logs |
| `bias_s30` | Bias strength 0.30 | Training logs |
| `bias_s60` | Bias strength 0.60 | Training logs |

### 9. **Special/Emerging Approaches**

| Run | Purpose | Output |
|-----|---------|--------|
| `aitoolkit_out/` | AI Toolkit diffusion outputs | `.safetensors` models + samples |
| `rlvr_*` | RL-based viewer/reasoner | Training logs |
| `neural_engine_hospital` | Neural engine for hospital | Training metrics |
| `vqa_check` | VQA validation | Evaluation results |
| `sprite_santa` | Sprite-based Santa | Rendering outputs |
| `plate_santa` | Plate/tray based variant | Training outputs |
| `edit_bench*` | Editing benchmarks | `samples/` with edits |

### 10. **Probe & Benchmark Runs** (`probe*`, `*_bench`)

| Run | Purpose | Output |
|-----|---------|--------|
| `probe1` | Single probe experiment | `summary.json` |
| `scene_bench` | Scene benchmark | Metrics |
| `wm_bench` | Working memory benchmark | Metrics |
| `edit_bench` | Editing benchmark | `samples/` |

### 11. **Testing & Development** (Prefixed with `_` or `*_test`)

| Run | Purpose |
|-----|---------|
| `_variant_test` | Architecture variant testing |
| `base_infer` | Base inference testing |
| `composite_test` | Composite scene testing |
| `engine_test` | Engine component testing |
| `fixed_input_test` | Fixed input validation |
| `one_pred_test` | Single prediction test |
| `upsample_test` | Upsampling testing |

## Key Files in Each Run

- **`train_log.json`**: Step-by-step training metrics (loss, learning rate, time)
- **`best.json`**: Best model statistics and validation performance
- **`samples/`**: Generated images showing model outputs at different steps
  - Format: `{timestamp}__{step}_{variation}.jpg`
  - Multiple variations per step show different samples/attempts
- **`ablation.json`**: Ablation-specific metrics (if applicable)
- **`*.pt` / `*.pth`**: PyTorch model checkpoints (**excluded from git**)
- **`*.safetensors`**: HuggingFace safetensors model format (mostly in `aitoolkit_out/`)

## Understanding Run Names

**Naming Convention**: `{method}_{scene}_{variant}[_{scale}]`

- **method**: `bundle`, `latent`, `lora`, `hospital`, `diamond`, `rlvr`, etc.
- **scene**: `hospital`, `santa`, `diamond`, `play`, `smoke`, etc.
- **variant**: architecture or param name (`qwen`, `vigorl`, `sharp`, `cw`, etc.)
- **scale**: `smoke` (quick test), `eval` (evaluation on test set), `800` (resolution)

### Examples
- `bundle_sharp_smoke` → Bundle denoiser with sharp attention, quick smoke test
- `hospital_xpred_noiseaug` → Hospital scene with cross-prediction + noise augmentation
- `latent_qwen_video` → Latent space approach using Qwen, with video outputs
- `eval_spec_hospital_det` → Specialized hospital evaluation with detailed outputs

## Datasets

### Input Data (`data/` & `game_datasets_bundle/`)
- **v1**: Random starting positions (128 samples)
- **center**: Center-start position for position-generalization (64 samples)
- **game_datasets_bundle/** (6.3GB): Full training dataset

### Generated Assets (`report/`)
- **`spatial_wm_slides.pdf`**: Presentation slides on spatial working memory results

## Training Outputs

Most runs include:
1. **Convergence plots**: Via `train_log.json` (plot with `jq . runs/*/train_log.json | python -c "import json,sys; ..."`
2. **Sample images**: `runs/{run}/samples/` → visual progression from step 0 → final
3. **Metrics**: `best.json` contains final validation performance (accuracy, loss, etc.)

## Reproduction

To reproduce a run:

```bash
# Example: rebuild bundle_sharp_smoke
python scripts/run_env.py \
    --config runs/bundle_sharp_smoke/config.json \
    --save-rollouts runs/bundle_sharp_smoke_v2 \
    --max-steps 6
```

(Note: Exact commands depend on the codebase's CLI structure — check `scripts/run_env.py --help`)

## Notes

- **Model weights are NOT in git** — see `.gitignore`. Download from the original training if needed.
- **Smoke tests** are quick validation runs (fewer steps, smaller dataset).
- **Eval runs** use full test sets and produce detailed metrics.
- **Images in `samples/`** are sufficient for visual inspection and comparison.
