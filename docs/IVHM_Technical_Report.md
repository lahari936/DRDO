# Algorithms for Integrated Vehicle Health Management (IVHM) of UAVs: Technical Report

**Problem:** DRDO DIA-CoE research problem, ADE. Research vertical: Aerospace Systems Design and Characterization (ASDC). Thrust area: Integrated Vehicle Health Management.
**Repository:** https://github.com/lahari936/DRDO
**Final numbers:** `Results_Summary.md`, generated from the multi-seed run by `make_submission.py`. This report describes the method. It deliberately contains no result numbers except the development history in section 8, so it can never disagree with the generated summary.

---

## 1. Problem and objectives

IVHM monitors, diagnoses and prognosticates the health of a UAV, and estimates the Remaining Useful Life (RUL) of its parts. The problem statement asks for algorithm development, data-set generation, and model training and testing for diagnostic and prognostic analysis of nine UAV subsystems. It sets a figure of merit of **model accuracy > 85 %**.

| Requirement in problem statement | Where it is met |
|---|---|
| Algorithm development, diagnostic | Fault detection (section 5.2) and fault diagnosis/isolation (section 5.3) for every subsystem |
| Algorithm development, prognostic | RUL estimation (section 5.4) for every subsystem that has run-to-failure data |
| Data-set generation | Physics-informed simulators for the seven subsystems without public data (section 3.2) |
| Model training and testing | Leakage-controlled training and held-out testing, repeated over five seeds (section 6) |
| Subsystems: airframe, ECS, hydraulic, landing gear, fuel, Li-ion battery, electronics, actuators, electrical | All nine are covered (table in section 3) |
| Deliverable: trained and tested models and source code | `models/`, `IVHM_UAV_Colab.ipynb`, `ivhm_pipeline.py` |
| Deliverable: test report and analysis report | `Results_Summary.md`, `results/*_Test_and_Analysis_Report.md`, this document |

## 2. System architecture

```
 subsystem sensors ──► cleaning ──► windowed features ──┬──► DETECTION  : healthy vs faulty (P(fault))
 (per flight / cycle)  (dropouts,   (statistical,        ├──► DIAGNOSIS  : which fault mode / component state
                        constants)   spectral, physics)  │
                     causal prognostic features ─────────┴──► PROGNOSIS  : RUL -> maintenance state
                                                                   │
                         IVHMSystem.assess() ◄─────────────────────┘
                         per-subsystem table + vehicle status NORMAL / CAUTION / CRITICAL
```

* Each subsystem has its own detector, diagnoser and RUL model, registered in an `IVHMSystem` object.
* `assess()` takes the latest feature row of each subsystem and returns: P(fault), the diagnosed condition with its confidence, RUL, and a status.
* The vehicle status is the worst subsystem status. CRITICAL means RUL is at or below the critical band. CAUTION means a fault was detected or a non-nominal condition was diagnosed.
* Everything is scikit-learn plus one optional PyTorch model. It runs on a free Google Colab CPU, and a GPU is used only by the 1D-CNN.

## 3. Data

| Subsystem | Data | Type | Detection | Diagnosis | RUL |
|---|---|---|---|---|---|
| Hydraulic system | UCI #447, ZeMA hydraulic test rig: 2205 cycles, 17 sensors, 4 component conditions | **REAL** | critical fault | cooler, valve, pump, accumulator | no run-to-failure data |
| Li-ion battery | NASA PCoE 18650 cells B0005/6/7/18 | **REAL** | (synthetic pack, below) | (synthetic pack, below) | yes, plus state of health |
| Propulsion (additional) | NASA C-MAPSS FD001 | SIMULATED-BENCHMARK | via RUL band | — | yes |
| Airframe | strain and modal-vibration simulator | SYNTHETIC | yes | crack, delamination, loose fastener | crack growth |
| Environment control system | avionics-bay cooling loop | SYNTHETIC | yes | HX fouling, fan wear, coolant leak, sensor bias | HX fouling |
| Landing gear | oleo strut, tyre, brake (landing events) | SYNTHETIC | yes | N2 leak, oil leak, tyre under-inflation, brake wear | brake wear |
| Fuel system | pump, filter, tank | SYNTHETIC | yes | filter clog, pump wear, leak, flow-meter drift | filter clogging |
| Li-ion battery pack | 6S pack model | SYNTHETIC | yes | imbalance, high resistance, internal short, thermal-sensor fault | resistance growth |
| Electronic systems | power switch, DC-link, solder | SYNTHETIC | yes | MOSFET aging, capacitor ESR, solder fatigue | MOSFET aging |
| Actuators | electro-mechanical actuator (EMA) | SYNTHETIC | yes | friction, backlash, winding degradation, position-sensor bias | friction wear |
| Electrical systems | 28 V distribution | SYNTHETIC | yes | connector resistance, intermittent open, insulation leakage, regulator drift | connector degradation |

### 3.1 Real and benchmark data (direct downloads, no login)
* UCI hydraulic: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems (CC BY 4.0). Helwig et al., *Condition monitoring of a complex hydraulic system using multivariate statistics*, I2MTC 2015.
* NASA battery: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip. Saha & Goebel, NASA Ames PCoE, 2007.
* NASA C-MAPSS: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip. Saxena et al., *Damage propagation modeling for aircraft engine run-to-failure simulation*, PHM 2008.
* Repository index: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

### 3.2 Synthetic data generation
No public, labelled UAV fault data exists for seven of the nine subsystems, so the notebook generates it with lumped-parameter, physics-informed simulators. Section 2.2 of the notebook lists every equation and constant. The realism measures are:

* Each simulated vehicle ("unit") has its own manufacturing tolerance (±4 %) on key parameters. Every split is made by unit.
* Each window has a random mission load profile and a random ambient temperature (−10 to 40 °C).
* Every channel carries Gaussian sensor noise, and 0.3 % of samples are random telemetry dropouts.
* Fault severity is continuous from 0.2 to 1.0, so mild faults overlap with normal variation.
* Run-to-failure fleets follow damage `d = (k/L)^p`, where life `L` is 80–200 flights and the exponent `p` is 1.5–3, both random for each unit. Failure is defined as d = 1.

**Limit:** the simulators encode assumed physics. Scores on them show that the algorithms recover that physics, not that they work on a real UAV (section 9).

## 4. Pre-processing and features

| Step | Detail |
|---|---|
| Cleaning | Synthetic dropouts are replaced by the window's per-sensor median. Battery records shorter than 1000 s or with non-physical capacity are dropped. C-MAPSS constant sensors are dropped using the training set only. |
| Window features (all) | For each sensor: mean, std, min, max, RMS, skew, kurtosis, slope, peak-to-peak, diff-std, dominant frequency, spectral centroid |
| Hydraulic | The 60 s cycle is cut into 6 segments (mean and std of each). Pressure, flow and motor-power sensors also get a 30-bin cycle profile. Every rig cycle runs the same load programme, so the bins are comparable across cycles, and accumulator precharge mainly changes the pressure transients. |
| Actuators | Per window, a least-squares fit of `current = a·acc + b·vel + c·pos + d·sign(vel) + e` estimates inertia, viscous friction, hinge load and Coulomb friction. Servo tracking error and error at command reversals (backlash) are added. |
| Fuel | Flow-corrected filter resistance dP/Q², pump head per amp, and the mismatch between tank-level drop and metered flow (leak or flow-meter drift) |
| Electronics | DC-link ESR estimate (ripple ÷ current), case-to-board temperature rise per I² (on-resistance × thermal resistance), and on-resistance corrected to 25 °C (+0.5 %/K) |
| Prognostic features (synthetic RUL) | All computed within each unit from past and present flights only: 5- and 15-flight rolling means, drift from the unit's own baseline (its first 10 flights), and the change of the 5-flight mean over the last 10 flights (degradation rate). Unit age is included. |
| Battery | Health indicators from the first 1000 s of each discharge only: voltage at 100–1000 s, voltage slope, temperature rise, current, ambient. For RUL, the capacity history measured up to and including the current discharge is added: latest, 5-cycle mean, 15-cycle slope, running maximum, margin to end of life (EOL), and trend-extrapolated cycles to EOL. |
| C-MAPSS | Informative sensors, plus 10-cycle causal rolling mean and std, plus cycle count. For the 1D-CNN, the last 30 cycles of normalised sensors. |

## 5. Algorithms

### 5.1 Model selection protocol (all tasks)
For each task the candidates compete in 5-fold **GroupKFold** cross-validation on the training units only. Groups are units, engines, cells, or blocks of 50 rig cycles. The winner is refit on all training units and scored once on the held-out units. Scalers live inside sklearn pipelines.

### 5.2 Fault detection
* Candidates: logistic regression, random forest and histogram gradient boosting, all binary.
* Detection derived from the diagnoser, as P(fault) = 1 − P(normal), competes with the binary detector in training cross-validation. The winner is the vehicle-level detector.
* Baseline: an Isolation Forest trained on healthy windows only, representing the case where no fault examples exist.

### 5.3 Fault diagnosis
* Multiclass classification over normal plus each fault mode, with the same three candidates.
* Hydraulic: one condition classifier per component, following the UCI labels.

### 5.4 Prognostics (RUL)
* Candidates: Ridge, random forest and histogram gradient boosting. C-MAPSS also tries an MLP.
* C-MAPSS also gets a **1D-CNN** over 30-cycle windows: 3 convolution layers and 2 dense layers in PyTorch. Early stopping uses 20 % of the training engines, and the official test set is scored once.
* Targets are capped piecewise-linearly: 100 flights for synthetic subsystems, 125 cycles for C-MAPSS.
* Battery: nested leave-one-battery-out. The outer loop holds out one cell; the inner loop selects the model on the remaining cells. EOL is 1.4 Ah, a 30 % fade.

## 6. Evaluation protocol and metric definitions

| Quantity | Definition |
|---|---|
| Detection / diagnosis accuracy | **Balanced accuracy** (mean recall over classes) on held-out units. Plain accuracy, macro-F1 and ROC-AUC are also reported. |
| RUL accuracy | **Maintenance-state accuracy**: the share of predictions whose band (critical / warning / healthy) matches the band of the true RUL. Bands: synthetic ≤20 / ≤50 flights; C-MAPSS ≤30 / ≤75 cycles; battery ≤15 / ≤40 cycles. RMSE, MAE, R², tolerance accuracy (\|error\| ≤ 15, or ≤ 10 for battery) and the C-MAPSS NASA score are reported alongside. |
| Battery state of health accuracy | 1 − MAPE of estimated capacity, leave-one-battery-out |
| Robustness | Five independent seeds. Each seed regenerates the synthetic data and re-draws every split. The summary reports the mean ± std, the worst seed, and a pass/fail against 85 % on both the mean and the worst seed. |

**Leakage controls:**
* Splits are by unit, and an assertion checks that train and test units never overlap.
* Model selection never sees the test split, and feature changes were judged on cross-validation and multi-seed means, not on the test split of a single run.
* Every rolling or trend feature is causal. A unit test confirms that changing a future value leaves earlier features unchanged.
* Battery features come from the first 1000 s of a discharge. For RUL, capacity history uses only completed discharges.

## 7. Integration demonstration
Section 9 of the notebook assembles a **virtual vehicle** from held-out samples of every dataset. It runs `IVHMSystem.assess()` and prints each prediction next to the ground truth. This demonstrates the integration layer; it is not a real UAV.

## 8. Development history (for traceability)

| Round | Change | Effect observed (held-out data) |
|---|---|---|
| 1 (single run) | Baseline pipeline | 25/32 headline tasks ≥ 85 %. Below target: accumulator 0.639, battery RUL 0.782, C-MAPSS 0.830, actuator diagnosis 0.780, actuator RUL 0.833, electronics RUL 0.844, fuel detection 0.807 |
| 2 (5 seeds) | Detection from diagnoser (chosen by training CV), hydraulic segment features, actuator system-identification features, C-MAPSS 1D-CNN | Fixed: fuel detection 0.943 ± 0.025, actuator diagnosis 0.948 ± 0.016, actuator RUL 0.865 ± 0.031, C-MAPSS 1D-CNN 0.860 ± 0.014. Improved but still below: accumulator 0.776 ± 0.047. Unchanged: battery RUL 0.782 ± 0.005. Still below: electronics RUL 0.817 ± 0.044, fuel RUL 0.843 ± 0.040 |
| 3 (final) | Hydraulic cycle profiles; battery capacity-history and trend features; causal prognostic features (baseline drift, degradation rate) for all synthetic RUL; physics indicators for fuel and electronics; extended-battery experiment removed (only 3 qualifying cells) | See `Results_Summary.md` |

## 9. Limitations and technical risks

1. **Synthetic is not field data.** Seven subsystems are validated only against this project's simulators. Real faults can show signatures the simulators do not model, such as sensor-specific noise, coupled faults, or operating regimes never simulated. The scores are an upper bound for the method, not a field-performance claim.
2. **The real data is not from UAVs.** The hydraulic data comes from an industrial rig and the battery data from lab-cycled 18650 cells. C-MAPSS simulates a large turbofan.
3. **Battery RUL rests on 3 cells.** It is statistically weak and should be treated as a feasibility result.
4. **The hydraulic accumulator is hard to separate.** Its 100 and 90 bar levels give very similar signals, and rig cycles are correlated in time; block-grouped splits reduce but do not remove that optimism.
5. **RUL "accuracy" depends on the chosen bands.** RMSE and MAE are always reported next to it.
6. **The deep model is variable.** The 1D-CNN varies more between seeds than the tree models.
7. **External links may change.** The URL list in the notebook is the single place to update them.

## 10. Recommended next steps
1. Validate on DRDO/ADE rig or flight logs (PX4/ArduPilot ULog or a flight-test DAQ). Calibrate the simulator constants against them, then re-run the pipeline unchanged with those tables.
2. Use real UAV actuator-failure data (CMU AirLab ALFA dataset) to test the actuator models.
3. Collect more battery cells cycled under a UAV mission profile, enough to train RUL on more than 3 cells.
4. Extract flight-control features on board (onboard computer) and run the `IVHMSystem` model in shadow mode during flights.

## 11. Reproducibility

| Item | Value |
|---|---|
| Environment | Google Colab Free (CPU) or any Python ≥ 3.10 machine. RAM < 3 GB (estimated). PyTorch is optional. |
| One run | `python ivhm_pipeline.py`, or Run all in Colab |
| Final package | `bash run_seeds.sh` runs 5 seeds, then `aggregate_seeds.py`, then `make_submission.py`, which writes `submission/` and `IVHM_submission.zip` |
| Seeds | `IVHM_SEED` environment variable (default 42; the submission uses 1–5) |
