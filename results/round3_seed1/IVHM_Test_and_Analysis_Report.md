# IVHM for UAVs - Test and Analysis Report
Generated 2026-11-25 09:08 UTC | Colab: False | quick mode: False | sklearn 1.9.1 | runtime 12.6 min

## 1. Data provenance
| dataset | type | available | source |
|---|---|---|---|
| cmapss | SIMULATED-BENCHMARK | True | NASA PCoE C-MAPSS turbofan degradation simulation, FD001 (Saxena et al., PHM 2008) |
| battery | REAL | True | NASA PCoE Li-ion 18650 aging tests B0005/6/7/18 (Saha & Goebel, 2007) |
| hydraulic | REAL | True | UCI #447, ZeMA hydraulic test rig, 2205 cycles (Helwig et al., 2015) |

Synthetic subsystems (8): airframe, ecs, landing_gear, fuel, battery_pack, electronics, actuators, electrical. Fleet size per subsystem: 48 units x 50 labelled windows, 40 run-to-failure units.
Datasets that could not be downloaded (their tasks were NOT run): none.

## 2. Headline held-out results vs the >85 % figure of merit
| subsystem | data_type | task | model | metric | value | meets_85pct |
|---|---|---|---|---|---|---|
| hydraulic | REAL | detection_selected | binary:LogReg | balanced_accuracy | 0.909 | YES |
| hydraulic:accumulator | REAL | diagnosis | LogReg | balanced_accuracy | 0.783 | no |
| hydraulic:cooler | REAL | diagnosis | RandomForest | balanced_accuracy | 1.000 | YES |
| hydraulic:pump | REAL | diagnosis | HistGB | balanced_accuracy | 1.000 | YES |
| hydraulic:valve | REAL | diagnosis | RandomForest | balanced_accuracy | 1.000 | YES |
| li_ion_battery | REAL | rul | nested-LOBO | state_accuracy | 0.864 | YES |
| li_ion_battery | REAL | soh | nested-LOBO | soh_accuracy | 0.962 | YES |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | state_accuracy | 0.830 | no |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | state_accuracy | 0.870 | YES |
| actuators | SYNTHETIC | detection_selected | diagnosis:HistGB | balanced_accuracy | 0.975 | YES |
| actuators | SYNTHETIC | diagnosis | HistGB | balanced_accuracy | 0.960 | YES |
| actuators | SYNTHETIC | rul | RandomForest | state_accuracy | 0.896 | YES |
| airframe | SYNTHETIC | detection_selected | diagnosis:LogReg | balanced_accuracy | 0.963 | YES |
| airframe | SYNTHETIC | diagnosis | LogReg | balanced_accuracy | 0.963 | YES |
| airframe | SYNTHETIC | rul | HistGB | state_accuracy | 0.867 | YES |
| battery_pack | SYNTHETIC | detection_selected | diagnosis:LogReg | balanced_accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | diagnosis | LogReg | balanced_accuracy | 0.995 | YES |
| battery_pack | SYNTHETIC | rul | RandomForest | state_accuracy | 0.906 | YES |
| ecs | SYNTHETIC | detection_selected | diagnosis:LogReg | balanced_accuracy | 0.945 | YES |
| ecs | SYNTHETIC | diagnosis | LogReg | balanced_accuracy | 0.920 | YES |
| ecs | SYNTHETIC | rul | HistGB | state_accuracy | 0.900 | YES |
| electrical | SYNTHETIC | detection_selected | diagnosis:LogReg | balanced_accuracy | 0.985 | YES |
| electrical | SYNTHETIC | diagnosis | LogReg | balanced_accuracy | 0.974 | YES |
| electrical | SYNTHETIC | rul | RandomForest | state_accuracy | 0.862 | YES |
| electronics | SYNTHETIC | detection_selected | binary:RandomForest | balanced_accuracy | 0.986 | YES |
| electronics | SYNTHETIC | diagnosis | HistGB | balanced_accuracy | 0.995 | YES |
| electronics | SYNTHETIC | rul | RandomForest | state_accuracy | 0.908 | YES |
| fuel | SYNTHETIC | detection_selected | diagnosis:LogReg | balanced_accuracy | 0.945 | YES |
| fuel | SYNTHETIC | diagnosis | LogReg | balanced_accuracy | 0.957 | YES |
| fuel | SYNTHETIC | rul | RandomForest | state_accuracy | 0.916 | YES |
| landing_gear | SYNTHETIC | detection_selected | diagnosis:LogReg | balanced_accuracy | 0.935 | YES |
| landing_gear | SYNTHETIC | diagnosis | LogReg | balanced_accuracy | 0.927 | YES |
| landing_gear | SYNTHETIC | rul | RandomForest | state_accuracy | 0.893 | YES |

Tasks meeting the target, by data type: {'REAL': '6/7', 'SIMULATED-BENCHMARK': '1/2', 'SYNTHETIC': '24/24'}.

## 3. All held-out accuracy-type metrics
| subsystem | data_type | task | model | split | metric | value | meets_85pct |
|---|---|---|---|---|---|---|---|
| hydraulic | REAL | detection | LogReg | test | accuracy | 0.926 | YES |
| hydraulic | REAL | detection | LogReg | test | balanced_accuracy | 0.909 | YES |
| hydraulic | REAL | detection_selected | binary:LogReg | test | accuracy | 0.926 | YES |
| hydraulic | REAL | detection_selected | binary:LogReg | test | balanced_accuracy | 0.909 | YES |
| hydraulic | REAL | detection_unsupervised | IsolationForest | test | accuracy | 0.881 | YES |
| hydraulic | REAL | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.736 | no |
| hydraulic:accumulator | REAL | diagnosis | LogReg | test | accuracy | 0.694 | no |
| hydraulic:accumulator | REAL | diagnosis | LogReg | test | balanced_accuracy | 0.783 | no |
| hydraulic:cooler | REAL | diagnosis | RandomForest | test | accuracy | 1.000 | YES |
| hydraulic:cooler | REAL | diagnosis | RandomForest | test | balanced_accuracy | 1.000 | YES |
| hydraulic:pump | REAL | diagnosis | HistGB | test | accuracy | 1.000 | YES |
| hydraulic:pump | REAL | diagnosis | HistGB | test | balanced_accuracy | 1.000 | YES |
| hydraulic:valve | REAL | diagnosis | RandomForest | test | accuracy | 1.000 | YES |
| hydraulic:valve | REAL | diagnosis | RandomForest | test | balanced_accuracy | 1.000 | YES |
| li_ion_battery | REAL | rul | nested-LOBO | lobo | state_accuracy | 0.864 | YES |
| li_ion_battery | REAL | rul | nested-LOBO | lobo | tolerance_accuracy | 0.692 | no |
| li_ion_battery | REAL | soh | nested-LOBO | lobo | soh_accuracy | 0.962 | YES |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_last_cycle | state_accuracy | 0.830 | no |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_all_cycles | state_accuracy | 0.914 | YES |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_last_cycle | state_accuracy | 0.870 | YES |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_all_cycles | state_accuracy | 0.931 | YES |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_last_cycle | tolerance_accuracy | 0.680 | no |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_all_cycles | tolerance_accuracy | 0.748 | no |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_last_cycle | tolerance_accuracy | 0.730 | no |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_all_cycles | tolerance_accuracy | 0.613 | no |
| actuators | SYNTHETIC | detection | HistGB | test | accuracy | 0.980 | YES |
| actuators | SYNTHETIC | detection | HistGB | test | balanced_accuracy | 0.981 | YES |
| actuators | SYNTHETIC | detection_from_diagnosis | HistGB | test | accuracy | 0.973 | YES |
| actuators | SYNTHETIC | detection_from_diagnosis | HistGB | test | balanced_accuracy | 0.975 | YES |
| actuators | SYNTHETIC | detection_selected | diagnosis:HistGB | test | accuracy | 0.973 | YES |
| actuators | SYNTHETIC | detection_selected | diagnosis:HistGB | test | balanced_accuracy | 0.975 | YES |
| actuators | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.490 | no |
| actuators | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.556 | no |
| actuators | SYNTHETIC | diagnosis | HistGB | test | accuracy | 0.968 | YES |
| actuators | SYNTHETIC | diagnosis | HistGB | test | balanced_accuracy | 0.960 | YES |
| actuators | SYNTHETIC | rul | RandomForest | test | state_accuracy | 0.896 | YES |
| actuators | SYNTHETIC | rul | RandomForest | test | tolerance_accuracy | 0.839 | no |
| airframe | SYNTHETIC | detection | LogReg | test | accuracy | 0.943 | YES |
| airframe | SYNTHETIC | detection | LogReg | test | balanced_accuracy | 0.943 | YES |
| airframe | SYNTHETIC | detection_from_diagnosis | LogReg | test | accuracy | 0.963 | YES |
| airframe | SYNTHETIC | detection_from_diagnosis | LogReg | test | balanced_accuracy | 0.963 | YES |
| airframe | SYNTHETIC | detection_selected | diagnosis:LogReg | test | accuracy | 0.963 | YES |
| airframe | SYNTHETIC | detection_selected | diagnosis:LogReg | test | balanced_accuracy | 0.963 | YES |
| airframe | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.637 | no |
| airframe | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.682 | no |
| airframe | SYNTHETIC | diagnosis | LogReg | test | accuracy | 0.963 | YES |
| airframe | SYNTHETIC | diagnosis | LogReg | test | balanced_accuracy | 0.963 | YES |
| airframe | SYNTHETIC | rul | HistGB | test | state_accuracy | 0.867 | YES |
| airframe | SYNTHETIC | rul | HistGB | test | tolerance_accuracy | 0.742 | no |
| battery_pack | SYNTHETIC | detection | LogReg | test | accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | detection | LogReg | test | balanced_accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | detection_from_diagnosis | LogReg | test | accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | detection_from_diagnosis | LogReg | test | balanced_accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | detection_selected | diagnosis:LogReg | test | accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | detection_selected | diagnosis:LogReg | test | balanced_accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.567 | no |
| battery_pack | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.616 | no |
| battery_pack | SYNTHETIC | diagnosis | LogReg | test | accuracy | 0.997 | YES |
| battery_pack | SYNTHETIC | diagnosis | LogReg | test | balanced_accuracy | 0.995 | YES |
| battery_pack | SYNTHETIC | rul | RandomForest | test | state_accuracy | 0.906 | YES |
| battery_pack | SYNTHETIC | rul | RandomForest | test | tolerance_accuracy | 0.889 | YES |
| ecs | SYNTHETIC | detection | LogReg | test | accuracy | 0.902 | YES |
| ecs | SYNTHETIC | detection | LogReg | test | balanced_accuracy | 0.895 | YES |
| ecs | SYNTHETIC | detection_from_diagnosis | LogReg | test | accuracy | 0.943 | YES |
| ecs | SYNTHETIC | detection_from_diagnosis | LogReg | test | balanced_accuracy | 0.945 | YES |
| ecs | SYNTHETIC | detection_selected | diagnosis:LogReg | test | accuracy | 0.943 | YES |
| ecs | SYNTHETIC | detection_selected | diagnosis:LogReg | test | balanced_accuracy | 0.945 | YES |
| ecs | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.495 | no |
| ecs | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.559 | no |
| ecs | SYNTHETIC | diagnosis | LogReg | test | accuracy | 0.937 | YES |
| ecs | SYNTHETIC | diagnosis | LogReg | test | balanced_accuracy | 0.920 | YES |
| ecs | SYNTHETIC | rul | HistGB | test | state_accuracy | 0.900 | YES |
| ecs | SYNTHETIC | rul | HistGB | test | tolerance_accuracy | 0.850 | no |
| electrical | SYNTHETIC | detection | LogReg | test | accuracy | 0.980 | YES |
| electrical | SYNTHETIC | detection | LogReg | test | balanced_accuracy | 0.984 | YES |
| electrical | SYNTHETIC | detection_from_diagnosis | LogReg | test | accuracy | 0.982 | YES |
| electrical | SYNTHETIC | detection_from_diagnosis | LogReg | test | balanced_accuracy | 0.985 | YES |
| electrical | SYNTHETIC | detection_selected | diagnosis:LogReg | test | accuracy | 0.982 | YES |
| electrical | SYNTHETIC | detection_selected | diagnosis:LogReg | test | balanced_accuracy | 0.985 | YES |
| electrical | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.783 | no |
| electrical | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.816 | no |
| electrical | SYNTHETIC | diagnosis | LogReg | test | accuracy | 0.980 | YES |
| electrical | SYNTHETIC | diagnosis | LogReg | test | balanced_accuracy | 0.974 | YES |
| electrical | SYNTHETIC | rul | RandomForest | test | state_accuracy | 0.862 | YES |
| electrical | SYNTHETIC | rul | RandomForest | test | tolerance_accuracy | 0.702 | no |
| electronics | SYNTHETIC | detection | RandomForest | test | accuracy | 0.987 | YES |
| electronics | SYNTHETIC | detection | RandomForest | test | balanced_accuracy | 0.986 | YES |
| electronics | SYNTHETIC | detection_from_diagnosis | HistGB | test | accuracy | 0.997 | YES |
| electronics | SYNTHETIC | detection_from_diagnosis | HistGB | test | balanced_accuracy | 0.997 | YES |
| electronics | SYNTHETIC | detection_selected | binary:RandomForest | test | accuracy | 0.987 | YES |
| electronics | SYNTHETIC | detection_selected | binary:RandomForest | test | balanced_accuracy | 0.986 | YES |
| electronics | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.638 | no |
| electronics | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.693 | no |
| electronics | SYNTHETIC | diagnosis | HistGB | test | accuracy | 0.997 | YES |
| electronics | SYNTHETIC | diagnosis | HistGB | test | balanced_accuracy | 0.995 | YES |
| electronics | SYNTHETIC | rul | RandomForest | test | state_accuracy | 0.908 | YES |
| electronics | SYNTHETIC | rul | RandomForest | test | tolerance_accuracy | 0.875 | YES |
| fuel | SYNTHETIC | detection | HistGB | test | accuracy | 0.940 | YES |
| fuel | SYNTHETIC | detection | HistGB | test | balanced_accuracy | 0.930 | YES |
| fuel | SYNTHETIC | detection_from_diagnosis | LogReg | test | accuracy | 0.952 | YES |
| fuel | SYNTHETIC | detection_from_diagnosis | LogReg | test | balanced_accuracy | 0.945 | YES |
| fuel | SYNTHETIC | detection_selected | diagnosis:LogReg | test | accuracy | 0.952 | YES |
| fuel | SYNTHETIC | detection_selected | diagnosis:LogReg | test | balanced_accuracy | 0.945 | YES |
| fuel | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.578 | no |
| fuel | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.647 | no |
| fuel | SYNTHETIC | diagnosis | LogReg | test | accuracy | 0.947 | YES |
| fuel | SYNTHETIC | diagnosis | LogReg | test | balanced_accuracy | 0.957 | YES |
| fuel | SYNTHETIC | rul | RandomForest | test | state_accuracy | 0.916 | YES |
| fuel | SYNTHETIC | rul | RandomForest | test | tolerance_accuracy | 0.919 | YES |
| landing_gear | SYNTHETIC | detection | LogReg | test | accuracy | 0.890 | YES |
| landing_gear | SYNTHETIC | detection | LogReg | test | balanced_accuracy | 0.884 | YES |
| landing_gear | SYNTHETIC | detection_from_diagnosis | LogReg | test | accuracy | 0.935 | YES |
| landing_gear | SYNTHETIC | detection_from_diagnosis | LogReg | test | balanced_accuracy | 0.935 | YES |
| landing_gear | SYNTHETIC | detection_selected | diagnosis:LogReg | test | accuracy | 0.935 | YES |
| landing_gear | SYNTHETIC | detection_selected | diagnosis:LogReg | test | balanced_accuracy | 0.935 | YES |
| landing_gear | SYNTHETIC | detection_unsupervised | IsolationForest | test | accuracy | 0.552 | no |
| landing_gear | SYNTHETIC | detection_unsupervised | IsolationForest | test | balanced_accuracy | 0.604 | no |
| landing_gear | SYNTHETIC | diagnosis | LogReg | test | accuracy | 0.928 | YES |
| landing_gear | SYNTHETIC | diagnosis | LogReg | test | balanced_accuracy | 0.927 | YES |
| landing_gear | SYNTHETIC | rul | RandomForest | test | state_accuracy | 0.893 | YES |
| landing_gear | SYNTHETIC | rul | RandomForest | test | tolerance_accuracy | 0.809 | no |

## 4. Error metrics (RUL in flights/cycles, battery capacity in Ah)
| subsystem | data_type | task | model | split | metric | value |
|---|---|---|---|---|---|---|
| hydraulic | REAL | detection | LogReg | test | roc_auc | 0.972 |
| hydraulic | REAL | detection_unsupervised | IsolationForest | test | roc_auc | 0.901 |
| hydraulic:accumulator | REAL | diagnosis | LogReg | test | macro_f1 | 0.638 |
| hydraulic:cooler | REAL | diagnosis | RandomForest | test | macro_f1 | 1.000 |
| hydraulic:pump | REAL | diagnosis | HistGB | test | macro_f1 | 1.000 |
| hydraulic:valve | REAL | diagnosis | RandomForest | test | macro_f1 | 1.000 |
| li_ion_battery | REAL | rul | nested-LOBO | lobo | mae | 7.676 |
| li_ion_battery | REAL | rul | nested-LOBO | lobo | r2 | 0.917 |
| li_ion_battery | REAL | rul | nested-LOBO | lobo | rmse | 9.467 |
| li_ion_battery | REAL | soh | nested-LOBO | lobo | r2 | 0.880 |
| li_ion_battery | REAL | soh | nested-LOBO | lobo | rmse_ah | 0.069 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_last_cycle | mae | 13.422 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_all_cycles | mae | 10.430 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_last_cycle | mae | 10.666 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_all_cycles | mae | 13.285 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_last_cycle | nasa_score | 794.705 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_last_cycle | nasa_score | 315.502 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_last_cycle | r2 | 0.785 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_all_cycles | r2 | 0.637 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_last_cycle | r2 | 0.876 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_all_cycles | r2 | 0.624 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_last_cycle | rmse | 18.573 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | HistGB | test_all_cycles | rmse | 16.620 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_last_cycle | rmse | 14.096 |
| propulsion_cmapss | SIMULATED-BENCHMARK | rul | CNN1D | test_all_cycles | rmse | 16.922 |
| actuators | SYNTHETIC | detection | HistGB | test | roc_auc | 0.993 |
| actuators | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.748 |
| actuators | SYNTHETIC | diagnosis | HistGB | test | macro_f1 | 0.967 |
| actuators | SYNTHETIC | rul | RandomForest | test | mae | 8.683 |
| actuators | SYNTHETIC | rul | RandomForest | test | r2 | 0.881 |
| actuators | SYNTHETIC | rul | RandomForest | test | rmse | 11.467 |
| airframe | SYNTHETIC | detection | LogReg | test | roc_auc | 0.988 |
| airframe | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.778 |
| airframe | SYNTHETIC | diagnosis | LogReg | test | macro_f1 | 0.965 |
| airframe | SYNTHETIC | rul | HistGB | test | mae | 10.158 |
| airframe | SYNTHETIC | rul | HistGB | test | r2 | 0.829 |
| airframe | SYNTHETIC | rul | HistGB | test | rmse | 13.870 |
| battery_pack | SYNTHETIC | detection | LogReg | test | roc_auc | 1.000 |
| battery_pack | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.775 |
| battery_pack | SYNTHETIC | diagnosis | LogReg | test | macro_f1 | 0.997 |
| battery_pack | SYNTHETIC | rul | RandomForest | test | mae | 6.839 |
| battery_pack | SYNTHETIC | rul | RandomForest | test | r2 | 0.926 |
| battery_pack | SYNTHETIC | rul | RandomForest | test | rmse | 9.115 |
| ecs | SYNTHETIC | detection | LogReg | test | roc_auc | 0.966 |
| ecs | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.712 |
| ecs | SYNTHETIC | diagnosis | LogReg | test | macro_f1 | 0.932 |
| ecs | SYNTHETIC | rul | HistGB | test | mae | 7.524 |
| ecs | SYNTHETIC | rul | HistGB | test | r2 | 0.904 |
| ecs | SYNTHETIC | rul | HistGB | test | rmse | 10.294 |
| electrical | SYNTHETIC | detection | LogReg | test | roc_auc | 0.995 |
| electrical | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.860 |
| electrical | SYNTHETIC | diagnosis | LogReg | test | macro_f1 | 0.981 |
| electrical | SYNTHETIC | rul | RandomForest | test | mae | 11.897 |
| electrical | SYNTHETIC | rul | RandomForest | test | r2 | 0.795 |
| electrical | SYNTHETIC | rul | RandomForest | test | rmse | 15.161 |
| electronics | SYNTHETIC | detection | RandomForest | test | roc_auc | 0.999 |
| electronics | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.883 |
| electronics | SYNTHETIC | diagnosis | HistGB | test | macro_f1 | 0.996 |
| electronics | SYNTHETIC | rul | RandomForest | test | mae | 6.922 |
| electronics | SYNTHETIC | rul | RandomForest | test | r2 | 0.926 |
| electronics | SYNTHETIC | rul | RandomForest | test | rmse | 9.151 |
| fuel | SYNTHETIC | detection | HistGB | test | roc_auc | 0.987 |
| fuel | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.785 |
| fuel | SYNTHETIC | diagnosis | LogReg | test | macro_f1 | 0.953 |
| fuel | SYNTHETIC | rul | RandomForest | test | mae | 6.235 |
| fuel | SYNTHETIC | rul | RandomForest | test | r2 | 0.933 |
| fuel | SYNTHETIC | rul | RandomForest | test | rmse | 8.668 |
| landing_gear | SYNTHETIC | detection | LogReg | test | roc_auc | 0.964 |
| landing_gear | SYNTHETIC | detection_unsupervised | IsolationForest | test | roc_auc | 0.760 |
| landing_gear | SYNTHETIC | diagnosis | LogReg | test | macro_f1 | 0.933 |
| landing_gear | SYNTHETIC | rul | RandomForest | test | mae | 9.205 |
| landing_gear | SYNTHETIC | rul | RandomForest | test | r2 | 0.869 |
| landing_gear | SYNTHETIC | rul | RandomForest | test | rmse | 12.182 |

## 5. Analysis
* Splits are by unit (vehicle, engine, battery, block of rig cycles); model choice used GroupKFold CV on training units only.
* Most frequent diagnosis confusion per subsystem (held-out):
  * airframe: normal -> delamination (9 windows)
  * ecs: temp_sensor_bias -> normal (18 windows)
  * landing_gear: oil_leak -> normal (13 windows)
  * fuel: normal -> pump_wear (20 windows)
  * battery_pack: cell_imbalance -> normal (2 windows)
  * electronics: solder_fatigue -> normal (2 windows)
  * actuators: backlash -> normal (9 windows)
  * electrical: insulation_leakage -> normal (7 windows)
  * hydraulic:cooler: no errors on test split
  * hydraulic:valve: no errors on test split
  * hydraulic:pump: no errors on test split
  * hydraulic:accumulator: accumulator=130 -> accumulator=115 (105 windows)
* Top features driving each diagnosis model (permutation importance):
  * airframe: accel_wing_tip__diff_std, accel_fuselage__std, accel_fuselage__rms, accel_wing_tip__mean
  * ecs: bay_temp__min, bay_temp__mean, bay_temp__rms, bay_temp__max
  * landing_gear: decel__kurt, strut_pressure__slope, strut_pressure__skew, strut_stroke_mm__skew
  * fuel: phys__massbalance_mismatch_lph, tank_level__p2p, tank_level__kurt, tank_level__std
  * battery_pack: cell_v_min__max, cell_v_min__mean, cell_v_min__rms, cell_v_min__min
  * electronics: phys__esr_est, phys__rds_on_25c, phys__thermal_index, supply_voltage__min
  * actuators: phys__coef_viscous, phys__track_err_mean, phys__reversal_err, phys__coef_inertia
  * electrical: main_current__mean, main_current__rms, main_current__max, main_current__min
  * hydraulic:cooler: PS1__mean, PS1__std, PS1__min, PS1__max
  * hydraulic:valve: PS1__mean, PS1__std, PS1__min, PS1__max
  * hydraulic:pump: SE__seg3_mean, FS1__prof25, SE__seg4_mean, SE__seg5_mean
  * hydraulic:accumulator: FS2__spec_centroid, PS1__prof00, PS3__seg0_std, FS1__prof05
* Tasks below 85 %: hydraulic:accumulator/diagnosis = 0.783; propulsion_cmapss/rul = 0.830

## 6. Limitations and risks (apply to every number above)
* SYNTHETIC scores measure recovery of faults from this project's simulators. The simulators encode assumed physics, so high scores can be optimistic: real faults may have signatures the models never saw. They must not be cited as field performance.
* Real-data coverage is limited to the hydraulic rig (industrial, not a UAV) and Li-ion cells tested in a lab. C-MAPSS is a turbofan simulation; small-UAV engines differ.
* Battery RUL rests on very few cells (leave-one-battery-out over the cells that reach 1.4 Ah); treat it as a feasibility result.
* Hydraulic rig cycles are correlated in time; block-grouped splits reduce, but do not remove, optimism.
* RUL accuracy figures depend on the stated state thresholds and tolerances; RMSE/MAE are reported alongside.
* The C-MAPSS 1D-CNN uses one validation split for early stopping; its score varies more between seeds than the tree models (use run_seeds.sh).
* Single-run scores near 85 % are not decisive; compare mean and spread over seeds (aggregate_seeds.py).
* Next step: validate on DRDO/ADE flight logs or rig data, calibrate simulator constants against it, and re-run this notebook unchanged with the real tables.