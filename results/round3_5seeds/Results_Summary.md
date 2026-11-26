# IVHM for UAVs: Final Results Summary
Generated 2026-11-25 10:42 UTC from 5 independent runs (seed_1, seed_2, seed_3, seed_4, seed_5). Each run regenerates the synthetic fleet and re-draws every train/test split with its own seed. Splits are by unit, model choice uses training data only, and the test split is scored once.

**Figure of merit (problem statement):** accuracy of the model > 85 %. Detection and diagnosis use balanced accuracy. RUL uses maintenance-state accuracy (critical / warning / healthy band). Battery state of health uses 1 − MAPE of capacity. Definitions are in the Technical Report, section 6.

## 1. Headline results (held-out data, mean over seeds)
| subsystem | data_type | task | mean ± std | worst seed | meets 85 % (mean) | meets 85 % (every seed) | model(s) selected |
|---|---|---|---|---|---|---|---|
| hydraulic | REAL | detection | 0.883 ± 0.095 | 0.729 | YES | no | binary:HistGB, binary:LogReg |
| hydraulic:accumulator | REAL | diagnosis | 0.747 ± 0.030 | 0.711 | no | no | HistGB, LogReg |
| hydraulic:cooler | REAL | diagnosis | 0.994 ± 0.008 | 0.983 | YES | YES | HistGB, LogReg, RandomForest |
| hydraulic:pump | REAL | diagnosis | 0.945 ± 0.052 | 0.862 | YES | YES | HistGB, LogReg |
| hydraulic:valve | REAL | diagnosis | 0.998 ± 0.002 | 0.996 | YES | YES | RandomForest |
| li_ion_battery | REAL | rul | 0.873 ± 0.005 | 0.864 | YES | YES | nested-LOBO |
| li_ion_battery | REAL | soh | 0.962 ± 0.000 | 0.962 | YES | YES | nested-LOBO |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | 0.836 ± 0.011 | 0.820 | no | no | HistGB, RandomForest |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul (1D-CNN) | 0.860 ± 0.014 | 0.850 | YES | YES | CNN1D |
| actuators | SYNTHETIC | detection | 0.962 ± 0.012 | 0.943 | YES | YES | binary:HistGB, diagnosis:HistGB |
| actuators | SYNTHETIC | diagnosis | 0.948 ± 0.016 | 0.921 | YES | YES | HistGB |
| actuators | SYNTHETIC | rul | 0.907 ± 0.013 | 0.896 | YES | YES | HistGB, RandomForest |
| airframe | SYNTHETIC | detection | 0.961 ± 0.006 | 0.952 | YES | YES | diagnosis:LogReg |
| airframe | SYNTHETIC | diagnosis | 0.955 ± 0.007 | 0.945 | YES | YES | LogReg |
| airframe | SYNTHETIC | rul | 0.865 ± 0.030 | 0.817 | YES | no | HistGB, RandomForest |
| battery_pack | SYNTHETIC | detection | 0.993 ± 0.003 | 0.988 | YES | YES | diagnosis:LogReg |
| battery_pack | SYNTHETIC | diagnosis | 0.988 ± 0.006 | 0.979 | YES | YES | LogReg |
| battery_pack | SYNTHETIC | rul | 0.873 ± 0.039 | 0.812 | YES | no | HistGB, RandomForest |
| ecs | SYNTHETIC | detection | 0.948 ± 0.011 | 0.939 | YES | YES | diagnosis:LogReg |
| ecs | SYNTHETIC | diagnosis | 0.922 ± 0.015 | 0.911 | YES | YES | LogReg |
| ecs | SYNTHETIC | rul | 0.900 ± 0.011 | 0.882 | YES | YES | HistGB, RandomForest |
| electrical | SYNTHETIC | detection | 0.981 ± 0.004 | 0.976 | YES | YES | diagnosis:LogReg |
| electrical | SYNTHETIC | diagnosis | 0.967 ± 0.006 | 0.957 | YES | YES | LogReg |
| electrical | SYNTHETIC | rul | 0.874 ± 0.021 | 0.854 | YES | YES | HistGB, RandomForest |
| electronics | SYNTHETIC | detection | 0.989 ± 0.009 | 0.977 | YES | YES | binary:RandomForest, diagnosis:HistGB, diagnosis:LogReg |
| electronics | SYNTHETIC | diagnosis | 0.990 ± 0.008 | 0.980 | YES | YES | HistGB, LogReg |
| electronics | SYNTHETIC | rul | 0.892 ± 0.026 | 0.860 | YES | YES | HistGB, RandomForest |
| fuel | SYNTHETIC | detection | 0.958 ± 0.016 | 0.940 | YES | YES | diagnosis:HistGB, diagnosis:LogReg |
| fuel | SYNTHETIC | diagnosis | 0.960 ± 0.007 | 0.949 | YES | YES | HistGB, LogReg |
| fuel | SYNTHETIC | rul | 0.901 ± 0.024 | 0.858 | YES | YES | HistGB, RandomForest |
| landing_gear | SYNTHETIC | detection | 0.939 ± 0.011 | 0.931 | YES | YES | diagnosis:LogReg |
| landing_gear | SYNTHETIC | diagnosis | 0.933 ± 0.014 | 0.920 | YES | YES | LogReg |
| landing_gear | SYNTHETIC | rul | 0.890 ± 0.019 | 0.864 | YES | YES | HistGB, RandomForest |

Tasks meeting the target on the mean, by data type: {'REAL': '6/7', 'SIMULATED-BENCHMARK': '1/2', 'SYNTHETIC': '24/24'}.

**Below target on the mean:** hydraulic:accumulator / diagnosis = 0.747; propulsion_cmapss / rul = 0.836.

**Meets target on the mean but not in every seed:** hydraulic / detection (worst 0.729); airframe / rul (worst 0.817); battery_pack / rul (worst 0.812).

## 2. Error and secondary metrics (mean ± std over seeds)
RUL errors are in flights (synthetic) or cycles (C-MAPSS, battery). Battery capacity errors are in Ah.

| subsystem | data_type | task | split | metric | mean ± std |
|---|---|---|---|---|---|
| hydraulic | REAL | detection | test | roc_auc | 0.980 ± 0.010 |
| hydraulic | REAL | detection_unsupervised | test | roc_auc | 0.881 ± 0.036 |
| hydraulic:accumulator | REAL | diagnosis | test | macro_f1 | 0.659 ± 0.078 |
| hydraulic:cooler | REAL | diagnosis | test | macro_f1 | 0.993 ± 0.010 |
| hydraulic:pump | REAL | diagnosis | test | macro_f1 | 0.932 ± 0.070 |
| hydraulic:valve | REAL | diagnosis | test | macro_f1 | 0.998 ± 0.002 |
| li_ion_battery | REAL | rul | lobo | mae | 7.663 ± 0.100 |
| li_ion_battery | REAL | rul | lobo | rmse | 9.437 ± 0.103 |
| li_ion_battery | REAL | rul | lobo | tolerance_accuracy | 0.703 ± 0.017 |
| li_ion_battery | REAL | soh | lobo | rmse_ah | 0.069 ± 0.000 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | test_last_cycle | mae | 13.420 ± 0.163 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | test_last_cycle | nasa_score | 762.701 ± 34.808 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | test_last_cycle | rmse | 18.408 ± 0.301 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | test_last_cycle | tolerance_accuracy | 0.652 ± 0.019 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul (1D-CNN) | test_last_cycle | mae | 11.281 ± 0.610 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul (1D-CNN) | test_last_cycle | nasa_score | 427.151 ± 83.975 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul (1D-CNN) | test_last_cycle | rmse | 15.012 ± 0.704 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul (1D-CNN) | test_last_cycle | tolerance_accuracy | 0.710 ± 0.047 |
| actuators | SYNTHETIC | detection | test | roc_auc | 0.985 ± 0.012 |
| actuators | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.753 ± 0.020 |
| actuators | SYNTHETIC | diagnosis | test | macro_f1 | 0.954 ± 0.018 |
| actuators | SYNTHETIC | rul | test | mae | 7.728 ± 1.211 |
| actuators | SYNTHETIC | rul | test | rmse | 10.477 ± 1.224 |
| actuators | SYNTHETIC | rul | test | tolerance_accuracy | 0.854 ± 0.036 |
| airframe | SYNTHETIC | detection | test | roc_auc | 0.986 ± 0.003 |
| airframe | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.772 ± 0.021 |
| airframe | SYNTHETIC | diagnosis | test | macro_f1 | 0.960 ± 0.006 |
| airframe | SYNTHETIC | rul | test | mae | 10.324 ± 1.980 |
| airframe | SYNTHETIC | rul | test | rmse | 14.006 ± 2.426 |
| airframe | SYNTHETIC | rul | test | tolerance_accuracy | 0.736 ± 0.084 |
| battery_pack | SYNTHETIC | detection | test | roc_auc | 1.000 ± 0.000 |
| battery_pack | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.758 ± 0.034 |
| battery_pack | SYNTHETIC | diagnosis | test | macro_f1 | 0.991 ± 0.004 |
| battery_pack | SYNTHETIC | rul | test | mae | 9.260 ± 2.355 |
| battery_pack | SYNTHETIC | rul | test | rmse | 11.954 ± 2.698 |
| battery_pack | SYNTHETIC | rul | test | tolerance_accuracy | 0.783 ± 0.109 |
| ecs | SYNTHETIC | detection | test | roc_auc | 0.958 ± 0.013 |
| ecs | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.725 ± 0.018 |
| ecs | SYNTHETIC | diagnosis | test | macro_f1 | 0.935 ± 0.015 |
| ecs | SYNTHETIC | rul | test | mae | 7.081 ± 0.371 |
| ecs | SYNTHETIC | rul | test | rmse | 9.709 ± 0.717 |
| ecs | SYNTHETIC | rul | test | tolerance_accuracy | 0.872 ± 0.031 |
| electrical | SYNTHETIC | detection | test | roc_auc | 0.992 ± 0.003 |
| electrical | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.840 ± 0.025 |
| electrical | SYNTHETIC | diagnosis | test | macro_f1 | 0.976 ± 0.004 |
| electrical | SYNTHETIC | rul | test | mae | 9.980 ± 2.509 |
| electrical | SYNTHETIC | rul | test | rmse | 13.184 ± 2.786 |
| electrical | SYNTHETIC | rul | test | tolerance_accuracy | 0.753 ± 0.076 |
| electronics | SYNTHETIC | detection | test | roc_auc | 0.999 ± 0.000 |
| electronics | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.888 ± 0.008 |
| electronics | SYNTHETIC | diagnosis | test | macro_f1 | 0.990 ± 0.011 |
| electronics | SYNTHETIC | rul | test | mae | 8.197 ± 1.955 |
| electronics | SYNTHETIC | rul | test | rmse | 10.925 ± 2.007 |
| electronics | SYNTHETIC | rul | test | tolerance_accuracy | 0.837 ± 0.080 |
| fuel | SYNTHETIC | detection | test | roc_auc | 0.990 ± 0.005 |
| fuel | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.791 ± 0.018 |
| fuel | SYNTHETIC | diagnosis | test | macro_f1 | 0.962 ± 0.010 |
| fuel | SYNTHETIC | rul | test | mae | 7.970 ± 2.206 |
| fuel | SYNTHETIC | rul | test | rmse | 10.533 ± 2.564 |
| fuel | SYNTHETIC | rul | test | tolerance_accuracy | 0.843 ± 0.097 |
| landing_gear | SYNTHETIC | detection | test | roc_auc | 0.967 ± 0.008 |
| landing_gear | SYNTHETIC | detection_unsupervised | test | roc_auc | 0.793 ± 0.039 |
| landing_gear | SYNTHETIC | diagnosis | test | macro_f1 | 0.938 ± 0.016 |
| landing_gear | SYNTHETIC | rul | test | mae | 9.749 ± 0.796 |
| landing_gear | SYNTHETIC | rul | test | rmse | 12.964 ± 0.933 |
| landing_gear | SYNTHETIC | rul | test | tolerance_accuracy | 0.776 ± 0.035 |

## 3. How to read these numbers
* **REAL** rows (UCI hydraulic rig, NASA Li-ion cells) are the evidence that the methods work on measured sensor data.
* **SIMULATED-BENCHMARK** (NASA C-MAPSS) is a public, peer-reviewed turbofan simulation and is comparable with the published literature.
* **SYNTHETIC** rows come from this project's physics-informed simulators. They show that the algorithms recover the modelled fault physics. They are **not** evidence of field performance and must be re-validated on DRDO/ADE rig or flight data.
* The "every seed" column is the conservative reading. Quote that column, or the mean ± std, rather than a single best run.