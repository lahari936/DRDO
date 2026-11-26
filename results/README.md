# Results

All runs were executed on the college GPU server (`bvrit-csm-gpu`) with the code in this repository, 5 seeds each.

| Folder | What it is | Code version |
|---|---|---|
| **`final_5seeds/`** | **Final results.** `Results_Summary.md` (mean ± std, worst seed, pass/fail against 85 %), `seed_summary_heldout.csv`, and each seed's `all_results.csv` | commit `4a071a9` |
| `round3_5seeds/` | Round 3, kept for traceability | commit `17b16f0` |
| `round2_5seeds/` | Round 2, kept for traceability | commit `98f7c19` |

The full submission package (models, figures, per-seed reports, code) is attached to the GitHub release `v1.0` as `IVHM_submission.zip`.

## Final headline (held-out data, mean of 5 seeds)

33 of 35 headline rows reach 85 % on the mean: REAL 6/7, SIMULATED-BENCHMARK 1/2 (the 1D-CNN passes; the tree baseline does not), SYNTHETIC 24/24.

| Task | Round 2 | Round 3 | Final |
|---|---|---|---|
| Li-ion battery RUL (REAL) | 0.782 | 0.873 | **0.873 ± 0.005** |
| Electronics RUL | 0.817 | 0.892 | **0.892 ± 0.026** |
| Fuel RUL | 0.843 | 0.901 | **0.901 ± 0.024** |
| Hydraulic critical-fault detection (REAL) | 0.868 | 0.883 | **0.915 ± 0.079** |
| C-MAPSS RUL, 1D-CNN | 0.860 | 0.860 | **0.860 ± 0.014** |
| Hydraulic accumulator (REAL) | 0.776 | 0.747 | **0.766 ± 0.026**, below target |

The accumulator explanation and the C-MAPSS model-choice note are in `docs/IVHM_Technical_Report.md`, section 8.
