"""Hydraulic feature-set / model comparison using TRAINING-split cross-validation only.

Why: in round 3 the 30-bin profile features lowered accumulator diagnosis (5-seed mean 0.776 -> 0.747), and
critical-fault detection varies a lot between seeds. This script compares feature sets and model families for the
accumulator and detection tasks under exactly the pipeline's protocol (blocks of 50 cycles, GroupShuffleSplit with
the run seed, 5-fold GroupKFold inside the training split). The test split is NOT scored here, so the choice that
goes into the pipeline is made without looking at test data.

Usage (on the server, from the repo root):  python experiments/hydraulic_cv_experiment.py ivhm_data/hydraulic
Writes experiments/hydraulic_cv_results.csv and prints mean CV balanced accuracy per configuration.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else 'ivhm_data/hydraulic')
SEEDS = [1, 2, 3, 4, 5]
SENSORS = {'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100, 'EPS1': 100,
           'FS1': 10, 'FS2': 10, 'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1, 'VS1': 1, 'CE': 1, 'CP': 1, 'SE': 1}
PRESSURE_FLOW_POWER = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2']

def window_features(x, fs, prefix):          # identical to the pipeline (Section 5)
    x = x.astype(np.float64); T = x.shape[1]
    mean = x.mean(1); c = x - mean[:, None]; m2 = (c ** 2).mean(1) + 1e-12
    t = np.arange(T) / fs; tc = t - t.mean()
    spec = np.abs(np.fft.rfft(c, axis=1)); spec[:, 0] = 0; freqs = np.fft.rfftfreq(T, 1 / fs)
    f = {'mean': mean, 'std': x.std(1), 'min': x.min(1), 'max': x.max(1), 'rms': np.sqrt((x ** 2).mean(1)),
         'skew': (c ** 3).mean(1) / m2 ** 1.5, 'kurt': (c ** 4).mean(1) / m2 ** 2,
         'slope': (c * tc).sum(1) / (tc ** 2).sum(), 'p2p': x.max(1) - x.min(1),
         'diff_std': np.diff(x, axis=1).std(1), 'dom_freq': freqs[spec.argmax(1)],
         'spec_centroid': (spec * freqs).sum(1) / (spec.sum(1) + 1e-12)}
    return {f'{prefix}__{k}': v for k, v in f.items()}

def segment_features(x, n_seg, prefix):
    f = {}
    for k, s in enumerate(np.array_split(x.astype(np.float64), n_seg, axis=1)):
        f[f'{prefix}__seg{k}_mean'] = s.mean(1); f[f'{prefix}__seg{k}_std'] = s.std(1)
    return f

def profile_features(x, n_pts, prefix):
    prof = x.astype(np.float64).reshape(len(x), n_pts, -1).mean(2)
    return {f'{prefix}__prof{k:02d}': prof[:, k] for k in range(n_pts)}

raw = {s: pd.read_csv(DATA / f'{s}.txt', sep='\t', header=None).to_numpy(np.float32) for s in SENSORS}
prof = pd.read_csv(DATA / 'profile.txt', sep='\t', header=None, names=['cooler', 'valve', 'pump', 'accumulator', 'stable'])
W = pd.concat([pd.DataFrame(window_features(raw[s], fs, s)) for s, fs in SENSORS.items()], axis=1)
S = pd.concat([pd.DataFrame(segment_features(raw[s], 6, s)) for s in SENSORS], axis=1)
P = pd.concat([pd.DataFrame(profile_features(raw[s], 30, s)) for s in PRESSURE_FLOW_POWER], axis=1)
pfp_cols = [c for c in pd.concat([W, S], axis=1).columns if c.split('__')[0] in PRESSURE_FLOW_POWER]
FEATURE_SETS = {
    'round1_window': W,
    'round2_window+segment': pd.concat([W, S], axis=1),
    'round3_window+segment+profile': pd.concat([W, S, P], axis=1),
    'round2_pressure_flow_power_only': pd.concat([W, S], axis=1)[pfp_cols],
}
levels = sorted(prof.accumulator.unique())
crit = ((prof.cooler <= 3) | (prof.valve <= 80) | (prof.pump >= 2) | (prof.accumulator <= 100)).astype(int).to_numpy()
TASKS = {'accumulator': np.searchsorted(levels, prof.accumulator.values), 'critical_detection': crit}
groups = np.arange(len(prof)) // 50

def models():
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=0)
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=0)
    return {'LogReg': lr, 'RandomForest': rf, 'HistGB': hgb,
            'HistGB_regularised': HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                                                 min_samples_leaf=30, l2_regularization=1.0,
                                                                 early_stopping=False, random_state=0),
            'SoftVote(LogReg,RF,HistGB)': VotingClassifier([('lr', clone(lr)), ('rf', clone(rf)), ('hgb', clone(hgb))],
                                                           voting='soft')}

def run(task, fs_name, seed):
    X, y = FEATURE_SETS[fs_name], TASKS[task]
    tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, groups=groups))
    cv = GroupKFold(n_splits=5)
    rows = []
    for m_name, m in models().items():
        s = cross_val_score(m, X.iloc[tr], y[tr], groups=groups[tr], cv=cv, scoring='balanced_accuracy')
        rows.append({'task': task, 'feature_set': fs_name, 'model': m_name, 'seed': seed, 'cv_balanced_accuracy': float(np.nanmean(s))})
    return rows

jobs = [(t, f, s) for t in TASKS for f in FEATURE_SETS for s in SEEDS]
res = pd.DataFrame([r for rows in Parallel(n_jobs=min(20, os.cpu_count() or 1), verbose=5)(delayed(run)(*j) for j in jobs) for r in rows])
out = Path(__file__).with_name('hydraulic_cv_results.csv')
res.to_csv(out, index=False)
summary = res.groupby(['task', 'feature_set', 'model']).cv_balanced_accuracy.agg(['mean', 'std', 'min']).round(4)
pd.set_option('display.width', 200)
for task, d in summary.groupby(level=0):
    print(f'\n=== {task}: mean CV balanced accuracy over seeds (training split only) ===')
    print(d.sort_values('mean', ascending=False).to_string())
print(f'\nSaved {out}')
