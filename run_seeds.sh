#!/usr/bin/env bash
# Run the full IVHM pipeline once per seed, then aggregate mean +/- std across runs.
# Usage:  bash run_seeds.sh            (seeds 1-5)
#         SEEDS="1 2 3" bash run_seeds.sh
set -euo pipefail
SEEDS="${SEEDS:-1 2 3 4 5}"
mkdir -p runs
for s in $SEEDS; do
  echo "=== seed $s ==="
  IVHM_SEED=$s IVHM_OUT_DIR="runs/seed_$s" python ivhm_pipeline.py > "runs/seed_$s.log" 2>&1 \
    || { echo "seed $s failed, see runs/seed_$s.log"; exit 1; }
  tail -n 1 "runs/seed_$s.log"
done
python aggregate_seeds.py runs
python make_submission.py runs
