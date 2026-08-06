#!/bin/bash
# Train the WINNING config (rich + LPIPS conv-UNet) per game, all 10, sequentially.
# Step budget scales with data size (tiny games converge fast; big games get more).
# Per-game sigma_data from game_sigmas.json. Clock-capped run.
#
#   ./scripts/train_all_games.sh          # all 10
#   ./scripts/train_all_games.sh hospital # one game
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# steps per game: capped at 2000 (well past useful range for most games).
# tiny games get fewer (converge/overfit early on ~200 transitions).
declare -A STEPS=(
  [hospital]=2000 [shopping_mall]=2000 [wildlife]=2000 [crops_and_farming]=2000
  [7_worlds_of_wonder]=2000 [space_exploration]=2000 [eid]=1800 [cricket]=1800
  [diwali]=1500 [reduce_reuse_recycle]=1200
)

GAMES=("$@")
if [ ${#GAMES[@]} -eq 0 ]; then
  GAMES=(hospital shopping_mall wildlife crops_and_farming 7_worlds_of_wonder \
         space_exploration eid cricket diwali reduce_reuse_recycle)
fi

SIGMAS=game_datasets_bundle/game_sigmas.json
for g in "${GAMES[@]}"; do
  steps=${STEPS[$g]:-3000}
  sig=$($PY -c "import json;print(json.load(open('$SIGMAS'))['$g'])")
  echo ""
  echo "======================================================================"
  echo " TRAINING $g  |  steps=$steps  sigma_data=$sig  |  $(date +%H:%M:%S)"
  echo "======================================================================"
  # NOTE: no `ulimit -v` here. On GB10's unified memory, CUDA reserves a huge
  # virtual address range; a 100GB VA cap makes model.to(cuda) fail with
  # "CUDA driver error: out of memory". Thermal protection = the CLOCK CAP
  # (nvidia-smi --lock-gpu-clocks), NOT ulimit.
  $PY scripts/train_rich_wm.py --game "$g" --steps "$steps" --batch 3 --lr 1e-4 \
      --sigma-data "$sig" --cond-channels 512 --lpips-weight 0.5 \
      --val-every 250 --save-every 1000 --out "runs/final/$g"
  echo "done: $g -> runs/final/$g/denoiser_final.pt"
done
echo ""
echo "ALL GAMES DONE. Checkpoints in runs/final/<game>/denoiser_final.pt"
