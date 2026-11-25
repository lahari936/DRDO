# Saved results (interim)

These are the outputs of runs that have finished so far, all executed on the college GPU server (`bvrit-csm-gpu`) with the code in this repository.

| Folder | What it is | Code version |
|---|---|---|
| `round2_5seeds/` | Complete round-2 evaluation: `seed_summary.csv` (mean / std / min / max over seeds 1–5) and each seed's `all_results.csv` | commit `98f7c19` |
| `round3_seed1/` | Round-3 seed 1 only: `all_results.csv`, auto-generated test and analysis report, figures | commit `17b16f0` |

Round 3 seeds 2–5 were still running when these files were saved. The final package (`Results_Summary.md` over 5 seeds) is produced by `bash run_seeds.sh` and `make_submission.py`, and it replaces this interim folder.

## Round 3, seed 1 vs round 2, seed 1 (held-out data, the four tasks that were below 85 %)

| Task | Data | Round 2 | Round 3 |
|---|---|---|---|
| Li-ion battery RUL, state accuracy | REAL | 0.782 | 0.864 |
| Electronics RUL, state accuracy | SYNTHETIC | 0.851 | 0.908 |
| Fuel RUL, state accuracy | SYNTHETIC | 0.889 | 0.916 |
| Hydraulic accumulator diagnosis, balanced accuracy | REAL | 0.840 | 0.783 |

These are single-seed figures. The 5-seed round-2 means for the same tasks were 0.782, 0.817, 0.843 and 0.776 (see `round2_5seeds/seed_summary.csv`). For the accumulator, training cross-validation was about 0.82 in both rounds, and its test score swings between 0.72 and 0.84 across seeds, so the seed-1 drop is within that range. Final conclusions should wait for the 5-seed summary.
