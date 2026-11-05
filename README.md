# Algorithms for Integrated Vehicle Health Management (IVHM) of UAVs

Fault detection, fault diagnosis and Remaining Useful Life (RUL) prediction for the nine UAV subsystems named in the DRDO DIA-CoE problem statement (ADE, Aerospace Systems Design and Characterization vertical).

Everything runs in **one Google Colab notebook on the free CPU runtime**. It uses no GPU, no paid API and no paid cloud service.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lahari936/DRDO/blob/main/IVHM_UAV_Colab.ipynb)

## Repository contents

| Path | What it is |
|---|---|
| `IVHM_UAV_Colab.ipynb` | The notebook. It has 13 sections, from environment setup through to the final report. |
| `ivhm_pipeline.py` | The same pipeline as one plain Python script, for servers and GPU machines without Jupyter |
| `requirements.txt` | Versions for running outside Colab (Colab already has all of them) |

## How to run (Google Colab, recommended)

1. Click **Open in Colab** above, or go to https://colab.research.google.com, choose **File > Open notebook > GitHub**, paste `lahari936/DRDO` and pick `IVHM_UAV_Colab.ipynb`.
2. Keep the default CPU runtime. You don't need a GPU (**Runtime > Change runtime type > CPU**).
3. Click **Runtime > Run all**.
4. Wait about 20 to 35 minutes. The notebook downloads about 300 MB of public data, generates the synthetic fleet, and trains and evaluates every model.
5. When it finishes, the last cell downloads `ivhm_outputs.zip`, which contains the models, figures, `all_results.csv` and `IVHM_Test_and_Analysis_Report.md`. If your browser blocks the download, open the Colab **Files** panel and download `/content/ivhm_outputs.zip` yourself.

Smoke test (about 5 minutes): insert a cell at the top with
```python
import os; os.environ['IVHM_QUICK'] = '1'
```
then use Run all. This mode shrinks only the synthetic fleet. Its scores are noisier and you shouldn't report them.

If a download fails (for example the NASA S3 bucket is temporarily unreachable), the notebook prints that dataset as unavailable, skips its tasks and marks them **NOT RUN** in the report. It never fills in numbers. Running it again later picks up any files already cached.

### GPU runtime
A Colab GPU runtime (**Runtime > Change runtime type > T4 GPU**) works with no changes. The models are scikit-learn, though, so they run on the CPU cores either way, and a GPU gives no speed-up. Choosing a GPU runtime only uses up your Colab GPU quota.

## How to run on a workstation, server or GPU machine

Requirements: Python 3.10 or newer, 4 GB of free RAM (8 GB recommended), about 1.5 GB of disk, and internet access for the first run.

```bash
git clone https://github.com/lahari936/DRDO.git
```
```bash
cd DRDO
```
```bash
python -m venv .venv && . .venv/bin/activate
```
```bash
pip install -r requirements.txt
```

Option A, plain script (no Jupyter needed; figures are saved to files instead of shown):
```bash
python ivhm_pipeline.py 2>&1 | tee ivhm_run.log
```

Option B, execute the notebook so the outputs are saved inside it:
```bash
pip install jupyter && jupyter nbconvert --to notebook --execute IVHM_UAV_Colab.ipynb --output IVHM_executed.ipynb --ExecutePreprocessor.timeout=3600
```

Outputs go to `./ivhm_outputs/` (models, figures, `all_results.csv`, `IVHM_Test_and_Analysis_Report.md`) and the downloaded data to `./ivhm_data/`. You can override these with the environment variables `IVHM_OUT_DIR` and `IVHM_DATA_DIR`. Set `IVHM_QUICK=1` for the short smoke test.

The first setup cell caps OpenMP at 4 threads (`OMP_NUM_THREADS`). On many-core CPUs, gradient boosting without that cap was about 40 times slower in testing.

## Accounts, access and hardware

| Item | Needed? |
|---|---|
| Google account (for Colab) | Yes. It is the only account required. |
| Kaggle / NASA / UCI accounts | No. All links are direct downloads with no login. |
| Google Cloud project or billing | No. Colab Free is enough. You can optionally save outputs to Google Drive. |
| GPU / TPU | No |
| RAM | Estimated to stay under about 3 GB. Colab Free gives about 12 GB. |

## Methodology

### Architecture

```
 sensors (per subsystem) --> cleaning --> windowed features --+--> fault DETECTION  (healthy / faulty; supervised + Isolation Forest baseline)
                                                              +--> fault DIAGNOSIS  (which fault mode / component condition)
                     causal rolling features + unit age ------+--> PROGNOSIS        (RUL regression -> maintenance state)
                                                                          |
                                              IVHMSystem.assess()  <------+  vehicle roll-up: NORMAL / CAUTION / CRITICAL
```

### Data: real vs synthetic

| Subsystem (problem statement) | Data | Type | Download |
|---|---|---|---|
| Hydraulic system | UCI #447 "Condition monitoring of hydraulic systems" (ZeMA rig, 2205 cycles, 17 sensors, 4 component conditions) | **REAL** | https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems |
| Li-ion battery (SOH + RUL) | NASA PCoE battery aging, cells B0005/6/7/18 | **REAL** | https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip |
| Propulsion (extra RUL benchmark) | NASA C-MAPSS FD001 | **SIMULATED-BENCHMARK** | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip |
| Airframe | strain and modal vibration simulator: crack, delamination, loose fastener | SYNTHETIC | generated in the notebook |
| Environment control system | avionics-bay cooling loop: HX fouling, fan degradation, coolant leak, sensor bias | SYNTHETIC | generated |
| Landing gear | oleo strut, tyre, brake: N2 leak, oil leak, tyre under-inflation, brake wear | SYNTHETIC | generated |
| Fuel system | pump, filter, tank: clogging, pump wear, leak, flow-sensor drift | SYNTHETIC | generated |
| Li-ion battery pack faults | 6S pack: imbalance, high resistance, internal short, thermal-sensor fault | SYNTHETIC | generated |
| Electronic systems | power switch, DC-link, solder: MOSFET aging, capacitor ESR rise, solder fatigue | SYNTHETIC | generated |
| Actuators | electro-mechanical actuator: friction, backlash, winding degradation, position-sensor bias | SYNTHETIC | generated |
| Electrical systems | 28 V distribution: connector resistance, intermittent open, insulation leakage, regulator drift | SYNTHETIC | generated |

NASA PCoE repository index: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ (mirror: https://data.phmsociety.org/nasa/).

No public, labelled UAV fault dataset exists for seven of the nine subsystems, so the notebook generates them with **physics-informed simulators**. These are lumped-parameter models with unit-to-unit tolerance, random mission profiles, sensor noise, telemetry dropouts, continuous fault severity (0.2 to 1.0) and power-law run-to-failure degradation. Section 2.2 of the notebook lists every assumption.

### Algorithms
* **Features:** for each sensor and window, 12 statistical and spectral features (mean, std, min, max, RMS, skew, kurtosis, slope, peak-to-peak, diff-std, dominant frequency, spectral centroid). Battery features come from only the first 1000 s of each discharge. RUL tables add causal rolling means and unit age.
* **Detection and diagnosis:** logistic regression, random forest and histogram gradient boosting compete. The winner is picked by GroupKFold cross-validation on training units only. An Isolation Forest trained on healthy data only serves as the unsupervised baseline.
* **RUL:** Ridge, random forest and histogram gradient boosting, plus an MLP for C-MAPSS. The target uses a piecewise-linear cap. Batteries use nested leave-one-battery-out evaluation.
* **Integration:** `IVHMSystem` registers every subsystem model and returns a per-subsystem table and a vehicle status.

### Leakage controls
* Data is split **by unit** (simulated vehicle, engine, battery, or block of 50 rig cycles). An assertion checks that no group appears in both train and test.
* Model selection runs only on training units, and the test split is used once.
* Scalers live inside pipelines. Constant-sensor removal is decided on the training data only.
* Rolling features are causal (past only) and computed within each unit.

### Figures of merit
The problem statement's FoM is "accuracy of the model > 85 %". The notebook reports:
* detection and diagnosis: accuracy, **balanced accuracy**, macro-F1, ROC-AUC;
* RUL: RMSE, MAE, R², plus two accuracy-type numbers it defines explicitly. **State accuracy** is the share of predictions in the correct critical/warning/healthy band. **Tolerance accuracy** is the share with |error| ≤ 15 cycles, or 10 for batteries;
* battery SOH: capacity accuracy = 1 − MAPE.

Every number in the report is computed during the run and tagged REAL / SIMULATED-BENCHMARK / SYNTHETIC. The report lists which tasks pass or miss 85 % and doesn't hide the misses.

## Results

This repository publishes **no accuracy numbers**. Every figure comes from your own run and is written to `ivhm_outputs/IVHM_Test_and_Analysis_Report.md`, with a YES/no column against the 85 % target. Please share that report, not numbers copied from elsewhere.

## Limitations and technical risks

* **Synthetic scores are not field performance.** They measure how well the algorithms recover faults from the simulator's own physics. Real faults can show signatures the simulator does not model. Calibrating and validating against DRDO/ADE rig or flight data is the required next step.
* The real data is not from UAVs. The hydraulic set comes from an industrial test rig and the battery set from lab-cycled 18650 cells. C-MAPSS simulates a large turbofan.
* Battery RUL is based on only the three cells that reach the 1.4 Ah end-of-life threshold (B0007 never reaches it), so treat it as a feasibility result.
* Hydraulic rig cycles are correlated in time. Block-grouped splits reduce that optimism but don't remove it.
* No raw-signal deep learning (CNN/LSTM) is used, which keeps the notebook CPU-friendly. Adding it is an upgrade path if GPU time becomes available.
* External links can change. If one does, update the `url` field in `REAL_DATASETS` (Section 2).
