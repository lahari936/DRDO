# Results

All runs were executed on the college GPU server (`bvrit-csm-gpu`) with the code in this repository, 5 seeds each.

| Folder | What it is | Code version |
|---|---|---|
| **`final_5seeds/`** | **Final results.** `Results_Summary.md` (mean ± std, worst seed, pass/fail against 85 %), `seed_summary_heldout.csv`, and each seed's `all_results.csv` | commit `4a071a9` |
| `round3_5seeds/` | Round 3, kept for traceability | commit `17b16f0` |
| `round2_5seeds/` | Round 2, kept for traceability | commit `98f7c19` |

The full submission package (models, figures, per-seed reports, code) is attached to the GitHub release `v1.0` as `IVHM_submission.zip`.

## Final headline (held-out data, mean of 5 seeds)

31 of 33 headline rows reach 85 % on the mean: REAL 6/7, SIMULATED-BENCHMARK 1/2 (the 1D-CNN passes; the tree baseline does not), SYNTHETIC 24/24. Counting only the deployed C-MAPSS model (the 1D-CNN), that is 31 of 32. (An earlier version of this file said "33 of 35"; that was a counting error.)

## Stress tests (`experiments/`, 5 seeds each)

| Test | Result | Meets 85 %? |
|---|---|---|
| Real UAV flights, ALFA (45 flights, leave-flights-out): fault detection | 0.835 ± 0.007 balanced accuracy, 33–34 of 35 failure flights detected, median delay 1.8 s | no (close) |
| ALFA 5-class diagnosis | 0.737 ± 0.018 (engine 0.985, aileron 0.520) | no |
| Synthetic diagnosis, new vehicles with 2× manufacturing tolerance | 0.914 | yes |
| Synthetic diagnosis, 2× sensor noise | 0.737 (RUL state accuracy holds at 0.856) | no |
| Vehicle-level status, 500 virtual vehicles | 0.905 ± 0.017 | yes |
| Inference, one CPU core | 0.5–32 ms per diagnosis model, 4–15 ms per RUL model | n/a |

Scripts: `experiments/alfa_real_uav.py`, `experiments/robustness_and_deployment.py`; outputs in `experiments/alfa_results/` and `experiments/robustness/`.

| Task | Round 2 | Round 3 | Final |
|---|---|---|---|
| Li-ion battery RUL (REAL) | 0.782 | 0.873 | **0.873 ± 0.005** |
| Electronics RUL | 0.817 | 0.892 | **0.892 ± 0.026** |
| Fuel RUL | 0.843 | 0.901 | **0.901 ± 0.024** |
| Hydraulic critical-fault detection (REAL) | 0.868 | 0.883 | **0.915 ± 0.079** |
| C-MAPSS RUL, 1D-CNN | 0.860 | 0.860 | **0.860 ± 0.014** |
| Hydraulic accumulator (REAL) | 0.776 | 0.747 | **0.766 ± 0.026**, below target |

The accumulator explanation and the C-MAPSS model-choice note are in `docs/IVHM_Technical_Report.md`, section 8.
