"""Robustness, deployment-cost and vehicle-level evaluation for the synthetic subsystems.

Answers three reviewer questions the main pipeline does not:
  1. Distribution shift: models are trained exactly as in ivhm_pipeline.py (nominal simulator settings), then tested
     on NEW vehicles generated with harsher settings they never saw: 2x sensor noise, 2x manufacturing tolerance,
     and both combined with 1 % extra telemetry dropouts. A large drop would mean the models memorised the simulator.
  2. Deployment cost: single-sample inference latency and serialised size of every detection/diagnosis/RUL model.
  3. Vehicle-level accuracy: 500 virtual vehicles, each assembled from one fresh fault window and one fresh
     run-to-failure flight per subsystem; the IVHM status rule (NORMAL / CAUTION / CRITICAL) is compared with the truth.

How: the script executes the pipeline's own cells up to and including synthetic RUL training (same code, same seed),
then evaluates. Usage:  IVHM_SEED=1 python experiments/robustness_and_deployment.py   (writes experiments/robustness/)
"""
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED_ENV = int(os.environ.get('IVHM_SEED', '1'))
os.environ['IVHM_SEED'] = str(SEED_ENV)
os.environ.setdefault('IVHM_OUT_DIR', f'runs_robustness/seed_{SEED_ENV}')
OUT = Path(__file__).with_name('robustness'); OUT.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

src = Path(__file__).resolve().parents[1].joinpath('ivhm_pipeline.py').read_text(encoding='utf-8')
cells = re.split(r'^# %% \[\d+\]\n', src, flags=re.M)
stop = next(i for i, c in enumerate(cells) if 'RUL_CAP_SYN, RUL_TOL_SYN, CRIT_SYN, WARN_SYN' in c)
g = {'__name__': '__main__'}
t0 = time.time()
for c in cells[:stop + 1]:
    exec(compile(c, 'ivhm_pipeline', 'exec'), g)
print(f'pipeline cells executed in {time.time() - t0:.0f} s', flush=True)

SYN, meas_orig = g['SYN_SPECS'], g['meas']
rng_master = np.random.default_rng(SEED_ENV + 1000)          # fresh vehicles, never used in training or testing

def set_noise(k):
    g['meas'] = (lambda x, sigma, rng: x + k * sigma * rng.standard_normal(x.shape)) if k != 1 else meas_orig

def features(name, X):
    sp = SYN[name]
    X, _ = g['clean_windows'](X)
    F = g['tensor_features'](X, sp['sensors'], sp['fs'])
    if name in g['PHYSICS_FEATURES']:
        F = pd.concat([F, pd.DataFrame(g['PHYSICS_FEATURES'][name](X, sp['fs']))], axis=1)
    return F

def fault_set(name, n_units, wpu, rng, tol, extra_drop):
    F_ = len(SYN[name]['faults'])
    unit = np.repeat(np.arange(n_units), wpu); n = len(unit)
    uf = rng.normal(1.0, tol, (n_units, 4))[unit]
    label = np.where(rng.random(n) < 0.4, 0, rng.integers(1, F_ + 1, n))
    sev = np.zeros((n, F_)); idx = np.flatnonzero(label > 0)
    sev[idx, label[idx] - 1] = rng.uniform(0.2, 1.0, len(idx))
    X = g['simulate'](name, sev, uf, rng)
    X[rng.random(X.shape) < extra_drop] = np.nan
    return features(name, X), label

def rtf_set(name, n_units, rng, tol, extra_drop):
    sp = SYN[name]; pidx = sp['faults'].index(sp['progressive'])
    U, K, R, Dm = [], [], [], []
    for u in range(n_units):
        life = int(rng.integers(80, 200)); p = rng.uniform(1.5, 3.0); k = np.arange(1, life + 1)
        d = np.clip((k / life) ** p * (1 + 0.05 * rng.standard_normal(life)), 0, 1)
        U.append(np.full(life, u)); K.append(k); R.append(life - k); Dm.append(d)
    unit, cyc, rul, d = map(np.concatenate, (U, K, R, Dm))
    sev = np.zeros((len(unit), len(sp['faults']))); sev[:, pidx] = d
    uf = rng.normal(1.0, tol, (n_units, 4))[unit]
    X = g['simulate'](name, sev, uf, rng)
    X[rng.random(X.shape) < extra_drop] = np.nan
    base = features(name, X)
    tab = pd.concat([pd.DataFrame({'unit': unit, 'age': cyc, 'rul': rul}), base], axis=1)
    tab = pd.concat([tab, g['prognostic_features'](tab, 'unit', list(base.columns))], axis=1)
    return tab

def detector(name):
    return g['NotNormalDetector'](g['DIAG_MODELS'][name]['model']) if g.get('DET_SOURCE', {}).get(name) == 'diagnosis' \
        else g['DET_MODELS'][name]['model']

# NotNormalDetector / DET_SOURCE are defined in Section 7.1, which runs before synthetic RUL; fall back if absent
if 'NotNormalDetector' not in g:
    raise SystemExit('pipeline layout changed: Section 7.1 not executed')

SCENARIOS = {'nominal (fresh vehicles)': (1.0, 0.04, 0.0), 'sensor noise x2': (2.0, 0.04, 0.0),
             'tolerance x2 (+/-8 %)': (1.0, 0.08, 0.0), 'noise x2 + tolerance x2 + 1 % dropouts': (2.0, 0.08, 0.01)}
rows, vehicle_pool = [], {}
crit, warn, cap = g['CRIT_SYN'], g['WARN_SYN'], g['RUL_CAP_SYN']
from sklearn.metrics import balanced_accuracy_score, accuracy_score
for scen, (noise, tol, drop) in SCENARIOS.items():
    set_noise(noise)
    for name in SYN:
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        F, lab = fault_set(name, 24, 50, rng, tol, drop)
        F = F[g['FAULT_TASKS'][name]['X'].columns]
        diag = g['DIAG_MODELS'][name]['model']
        yd = diag.predict(F)
        pf = detector(name).predict_proba(F)[:, 1]
        tab = rtf_set(name, 20, rng, tol, drop)
        rm = g['RUL_MODELS'][name]
        yr = np.minimum(tab.rul.to_numpy(float), cap)
        pr = np.clip(rm['model'].predict(tab[rm['feats']]), 0, cap)
        rows.append({'seed': SEED_ENV, 'scenario': scen, 'subsystem': name,
                     'detection_balanced_accuracy': balanced_accuracy_score((lab > 0).astype(int), (pf >= 0.5).astype(int)),
                     'diagnosis_balanced_accuracy': balanced_accuracy_score(lab, yd),
                     'rul_state_accuracy': accuracy_score(g['health_state'](yr, crit, warn), g['health_state'](pr, crit, warn)),
                     'rul_rmse': float(np.sqrt(np.mean((pr - yr) ** 2)))})
        if scen.startswith('nominal'):
            vehicle_pool[name] = (lab, yd, pf, yr, pr)
        print(f'{scen:40s} {name:13s} diag={rows[-1]["diagnosis_balanced_accuracy"]:.3f} '
              f'rul_state={rows[-1]["rul_state_accuracy"]:.3f}', flush=True)
set_noise(1.0)
pd.DataFrame(rows).to_csv(OUT / f'shift_seed{SEED_ENV}.csv', index=False)

# ---- deployment cost: single-sample latency and serialised size
dep = []
def bench(name, task, model, X1):
    for _ in range(5):
        model.predict(X1)
    ts = []
    for _ in range(100):
        a = time.perf_counter(); model.predict(X1); ts.append(time.perf_counter() - a)
    dep.append({'seed': SEED_ENV, 'subsystem': name, 'task': task, 'model': type(model).__name__,
                'latency_ms_median': 1000 * float(np.median(ts)), 'size_kb': len(pickle.dumps(model)) / 1024})
for name in SYN:
    X1 = g['FAULT_TASKS'][name]['X'].iloc[[0]]
    bench(name, 'diagnosis', g['DIAG_MODELS'][name]['model'], X1)
    rm = g['RUL_MODELS'][name]
    bench(name, 'rul', rm['model'], g['RTF_TABLES'][name][rm['feats']].iloc[[0]])
for name, d in g['DIAG_MODELS'].items():
    if name.startswith('hydraulic'):
        bench(name, 'diagnosis', d['model'], g['DIAG_TASKS'][name]['X'].iloc[[0]])
pd.DataFrame(dep).to_csv(OUT / f'deployment_seed{SEED_ENV}.csv', index=False)

# ---- vehicle-level status accuracy over 500 virtual vehicles (nominal fresh data)
order = {'NORMAL': 0, 'CAUTION': 1, 'CRITICAL': 2}
def status(fault, rul):
    return 'CRITICAL' if rul <= crit else ('CAUTION' if fault else 'NORMAL')
vr = np.random.default_rng(SEED_ENV + 2000)
truth_v, pred_v, sub_ok = [], [], []
for _ in range(500):
    ts_, ps_ = [], []
    for name, (lab, yd, pf, yr, pr) in vehicle_pool.items():
        i, j = vr.integers(len(lab)), vr.integers(len(yr))
        t_s = status(lab[i] > 0, yr[j]); p_s = status(pf[i] >= 0.5 or yd[i] != 0, pr[j])
        ts_.append(t_s); ps_.append(p_s); sub_ok.append(t_s == p_s)
    truth_v.append(max(ts_, key=order.get)); pred_v.append(max(ps_, key=order.get))
veh = {'seed': SEED_ENV, 'vehicle_status_accuracy': float(np.mean(np.array(truth_v) == np.array(pred_v))),
       'subsystem_status_accuracy': float(np.mean(sub_ok)),
       'true_status_mix': pd.Series(truth_v).value_counts().to_dict()}
pd.DataFrame([veh]).to_csv(OUT / f'vehicle_seed{SEED_ENV}.csv', index=False)
print('vehicle-level:', veh, flush=True)
print(f'done in {time.time() - t0:.0f} s')
