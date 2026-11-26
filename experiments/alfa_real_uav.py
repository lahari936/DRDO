"""Real-UAV validation on the CMU AirLab ALFA dataset (Keipour, Mousaei & Scherer, IJRR 2021).

ALFA: 47 autonomous flights of a Carbon-Z T-28 fixed-wing UAV with in-flight engine, aileron, rudder and elevator
failures and ground-truth failure onset times (CC BY 4.0, https://doi.org/10.1184/R1/12707963).
This adds REAL UAV evidence for the actuator (control-surface) and propulsion subsystems, which the main pipeline can
only test on simulated data.

Protocol (no information from test flights is used for any choice):
  * 5 Hz common time grid; 2 s windows with 1 s hop; features = mean, std, min, max, change per signal.
  * Window label = failure type if the window centre is after the failure onset, otherwise 'normal'.
  * Outer split: leave-flights-out StratifiedGroupKFold (5 folds, grouped by flight), repeated for seeds 1-5.
  * Model (LogReg / RandomForest / HistGB) chosen per outer fold by 3-fold GroupKFold CV on the training flights only.
  * Metrics on pooled out-of-fold predictions: diagnosis balanced accuracy (5 classes), detection balanced accuracy
    (fault vs normal), false-alarm rate on normal windows, and per-flight detection delay
    (first time after onset with two consecutive fault windows).

Usage: python experiments/alfa_real_uav.py ivhm_data/alfa/processed
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'ivhm_data/alfa/processed')
OUT = Path(__file__).with_name('alfa_results')
OUT.mkdir(exist_ok=True)
FS, WIN, HOP = 5.0, 10, 5                      # 5 Hz grid, 2 s window, 1 s hop
CLASSES = ['normal', 'engine', 'aileron', 'rudder', 'elevator']

def flight_type(name):
    n = name.lower()
    if 'no_ground_truth' in n or ('rudder' in n and 'aileron' in n):
        return None                             # unlabeled or mixed-surface flight: excluded
    for c in ('engine', 'elevator', 'rudder', 'aileron'):
        if c in n:
            return c
    return 'normal' if 'no_failure' in n else None

def read(seq_dir, topic):
    f = glob.glob(str(seq_dir / f'*-{topic}.csv'))
    if not f:
        return None
    d = pd.read_csv(f[0])
    d['t'] = d['%time'].astype('int64') / 1e9
    return d

def load_flight(seq_dir):
    topics = {
        'mavros-nav_info-roll': {'roll_cmd': 'field.commanded', 'roll_meas': 'field.measured'},
        'mavros-nav_info-pitch': {'pitch_cmd': 'field.commanded', 'pitch_meas': 'field.measured'},
        'mavros-nav_info-yaw': {'yaw_cmd': 'field.commanded', 'yaw_meas': 'field.measured'},
        'mavros-vfr_hud': {'airspeed': 'field.airspeed', 'groundspeed': 'field.groundspeed', 'throttle': 'field.throttle',
                           'altitude': 'field.altitude', 'climb': 'field.climb'},
        'mavros-imu-data': {'gyro_x': 'field.angular_velocity.x', 'gyro_y': 'field.angular_velocity.y',
                            'gyro_z': 'field.angular_velocity.z', 'acc_x': 'field.linear_acceleration.x',
                            'acc_y': 'field.linear_acceleration.y', 'acc_z': 'field.linear_acceleration.z'},
        'mavros-rc-out': {'rc_aileron': 'field.channels0', 'rc_elevator': 'field.channels1',
                          'rc_throttle': 'field.channels2', 'rc_rudder': 'field.channels3'},
    }
    frames = {k: read(seq_dir, k) for k in topics}
    if any(v is None or len(v) < 5 for v in frames.values()):
        return None
    t0 = max(v.t.min() for v in frames.values()); t1 = min(v.t.max() for v in frames.values())
    grid = np.arange(t0, t1, 1 / FS)
    sig = {}
    for topic, cols in topics.items():
        d = frames[topic].sort_values('t')
        for name, col in cols.items():
            sig[name] = np.interp(grid, d.t.to_numpy(), pd.to_numeric(d[col], errors='coerce').ffill().bfill().to_numpy())
    s = pd.DataFrame(sig)
    s['roll_err'] = s.roll_cmd - s.roll_meas
    s['pitch_err'] = s.pitch_cmd - s.pitch_meas
    s['yaw_err'] = (s.yaw_cmd - s.yaw_meas + 180) % 360 - 180
    s = s.drop(columns=['yaw_cmd', 'yaw_meas'])
    onset = None
    for f in glob.glob(str(seq_dir / '*-failure_status-*.csv')):
        d = pd.read_csv(f)
        on = d.loc[pd.to_numeric(d['field.data'], errors='coerce') != 0, '%time']
        if len(on):
            o = on.astype('int64').min() / 1e9
            onset = o if onset is None else min(onset, o)
    return grid - t0, s, (None if onset is None else onset - t0)

def windows(t, s, onset, ftype, flight):
    rows, labels, times = [], [], []
    X = s.to_numpy(float)
    for st in range(0, len(t) - WIN + 1, HOP):
        w = X[st:st + WIN]
        centre = t[st + WIN // 2]
        f = np.concatenate([w.mean(0), w.std(0), w.min(0), w.max(0), w[-1] - w[0]])
        rows.append(f); times.append(centre)
        labels.append(ftype if (onset is not None and centre >= onset) else 'normal')
    cols = [f'{c}__{k}' for k in ('mean', 'std', 'min', 'max', 'delta') for c in s.columns]
    df = pd.DataFrame(rows, columns=cols)
    df['label'] = labels; df['flight'] = flight; df['t'] = times; df['onset'] = onset if onset is not None else np.nan
    df['flight_type'] = ftype
    return df

data = []
for seq in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    ftype = flight_type(seq.name)
    if ftype is None:
        continue
    loaded = load_flight(seq)
    if loaded is None:
        print('skipped (missing topics):', seq.name); continue
    t, s, onset = loaded
    if ftype != 'normal' and onset is None:
        print('skipped (no onset):', seq.name); continue
    data.append(windows(t, s, onset, ftype, seq.name))
D = pd.concat(data, ignore_index=True)
FEATS = [c for c in D.columns if '__' in c]
y = D.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
groups = D.flight.to_numpy()
summary = D.groupby('flight_type').flight.nunique().to_dict()
print('flights per type:', summary, '| windows per label:', D.label.value_counts().to_dict())

def candidates():
    return {'LogReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight='balanced')),
            'RandomForest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced',
                                                   n_jobs=1, random_state=0),
            'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight='balanced',
                                                     random_state=0)}

def outer_fold(seed, k, tr, te):
    inner = GroupKFold(n_splits=3)
    scores = {n: float(np.nanmean(cross_val_score(m, D.iloc[tr][FEATS], y[tr], groups=groups[tr], cv=inner,
                                                  scoring='balanced_accuracy'))) for n, m in candidates().items()}
    best = max(scores, key=scores.get)
    model = clone(candidates()[best]).fit(D.iloc[tr][FEATS], y[tr])
    proba = np.zeros((len(te), len(CLASSES)))
    proba[:, model.classes_] = model.predict_proba(D.iloc[te][FEATS])
    return seed, k, te, proba, best, scores

jobs = []
for seed in [1, 2, 3, 4, 5]:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    jobs += [(seed, k, tr, te) for k, (tr, te) in enumerate(sgkf.split(D[FEATS], y, groups))]
outs = Parallel(n_jobs=min(len(jobs), 10), verbose=0)(delayed(outer_fold)(*j) for j in jobs)

rows, chosen = [], []
for seed in [1, 2, 3, 4, 5]:
    proba = np.zeros((len(D), len(CLASSES)))
    for s_, k, te, p, best, scores in outs:
        if s_ == seed:
            proba[te] = p; chosen.append({'seed': seed, 'fold': k, 'model': best, **{f'cv_{n}': v for n, v in scores.items()}})
    pred = proba.argmax(1)
    det_true, det_pred = (y > 0).astype(int), (proba[:, 0] < 0.5).astype(int)
    normal = y == 0
    delays, missed = [], 0
    for fl, d in D[D.flight_type != 'normal'].groupby('flight'):
        idx = d.index.to_numpy()
        after = idx[d.t.to_numpy() >= d.onset.iloc[0]]
        flag = det_pred[after]
        hit = np.flatnonzero(flag[:-1] & flag[1:]) if len(flag) > 1 else np.array([], int)
        if len(hit):
            delays.append(D.t[after[hit[0] + 1]] - d.onset.iloc[0])
        else:
            missed += 1
    per_class = recall_score(y, pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    rows.append({'seed': seed, 'diagnosis_balanced_accuracy': balanced_accuracy_score(y, pred),
                 'detection_balanced_accuracy': balanced_accuracy_score(det_true, det_pred),
                 'false_alarm_rate_normal_windows': float(det_pred[normal].mean()),
                 'median_detection_delay_s': float(np.median(delays)) if delays else np.nan,
                 'failure_flights_detected': f'{len(delays)}/{len(delays) + missed}',
                 **{f'recall_{c}': r for c, r in zip(CLASSES, per_class)}})
    if seed == 1:
        cm = confusion_matrix(y, pred, labels=range(len(CLASSES)), normalize='true')
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).round(3).to_csv(OUT / 'confusion_seed1.csv')

res = pd.DataFrame(rows)
res.to_csv(OUT / 'alfa_per_seed.csv', index=False)
pd.DataFrame(chosen).to_csv(OUT / 'alfa_model_choice.csv', index=False)
num = res.select_dtypes('number').drop(columns='seed')
agg = pd.DataFrame({'mean': num.mean(), 'std': num.std(), 'min': num.min(), 'max': num.max()}).round(4)
agg.to_csv(OUT / 'alfa_summary.csv')
meta = {'flights_per_type': {k: int(v) for k, v in summary.items()}, 'windows': int(len(D)),
        'windows_per_label': {k: int(v) for k, v in D.label.value_counts().items()}, 'features': len(FEATS),
        'model_choice_counts': pd.DataFrame(chosen).model.value_counts().to_dict()}
(OUT / 'alfa_meta.json').write_text(json.dumps(meta, indent=2))
pd.set_option('display.width', 200)
print(res.round(3).to_string(index=False))
print('\nMean over seeds:\n', agg.to_string())
print('\nMeta:', json.dumps(meta))
