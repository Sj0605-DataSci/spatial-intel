# Hospital Neural Game Engine

A playable "Hospital's Journey" game where the **board frames are rendered by a
neural world model** (rich-conditioned latent diffusion), while the HUD, effects,
and game logic run deterministically. Fully self-contained — no internet needed.

## How it works
- The **MDP** (`HospitalMDP` in `engine.py`, rules from the RDDL) tracks the player
  cell + HUD counters (patients treated, score).
- Each move: the engine **composites a clean input frame** (actor at the current
  cell + hospital objects + the matching HUD strip), then the **world model renders
  the next board frame** from (frame, action, effects).
- Because the input is always a *clean composite* (never the model's own previous
  output), the image **never drifts/degrades** across a full playthrough.
- pygame draws the frame + status; the HUD comes from real strips keyed to counters.

## Requirements
- **A CUDA GPU** (the model is ~2.4 s/frame at 30 denoise steps, ~1.5 s at 20).
  CPU is impractical (>100 s/frame).
- Python 3.10+, and: `torch`, `diffusers`, `pygame`, `pillow`, `numpy`.
  (Same torch/diffusers versions as the training env; CUDA build.)

## Run
```bash
python engine.py
```
Controls: **arrow keys** = move, **space** = treat patient (when on a patient cell),
**R** = reset, **Esc** = quit. Treat the 2 patients to reach SCORE 8/8 and win.

## What's bundled (self-contained)
```
engine.py                     # the game + pygame loop + MDP
model/
  denoiser_final.pt           # trained world model (rich+LPIPS, 2000 steps, SSIM 0.84)
  qwen_vae/                   # Qwen-Image VAE (encode/decode), bundled (no download)
  latent_wm.py, rich_wm.py    # model + conditioning code
  composite_frame.py          # frame compositor
  diamond_src/                # DIAMOND diffusion code (denoiser/sampler), bundled
assets/
  themed_ground.png           # clean board background
  sprites/                    # actor + object sprites (player_admin, doctor, nurse, patients, supply_room)
  hud_strips/                 # 165 real HUD strips keyed by counter values
  geometry.json               # calibrated cell->pixel mapping
  meta.json, states.jsonl, transitions.jsonl, source/  # MDP + RDDL
  state_ref.png               # reference frame (size/HUD fallback)
```

## Notes
- Turn-based, not real-time (each rendered frame takes ~1.5-2.4 s on GPU).
- To speed up at slight quality cost, lower `DENOISE` in `engine.py` (e.g. 20 -> 10).
- The Qwen VAE loads from `model/qwen_vae`; override with `QWEN_VAE_PATH` env if needed.
- This is a *renderer*: the MDP is the source of truth for state; the model draws it.
