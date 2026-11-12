"""Aggregate held-out IVHM results across seed runs: mean, std, min, max per task and metric.

Usage: python aggregate_seeds.py runs     (expects runs/<run>/all_results.csv)
Writes runs/seed_summary.csv and prints the accuracy-type metrics against the 85 % target.
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs')
files = sorted(root.glob('*/all_results.csv'))
if not files:
    sys.exit(f'no all_results.csv found under {root}/')

df = pd.concat([pd.read_csv(f).assign(run=f.parent.name) for f in files])
held = df[df.split.isin(['test', 'test_last_cycle', 'test_all_cycles', 'lobo'])].copy()
# The selected model can differ between seeds; keep the CNN separate because it is reported next to the selected model.
held['model_key'] = held.model.where(held.model == 'CNN1D', 'selected')
keys = ['subsystem', 'data_type', 'task', 'model_key', 'split', 'metric']
summary = held.groupby(keys).agg(mean=('value', 'mean'), std=('value', 'std'), min=('value', 'min'),
                                 max=('value', 'max'), runs=('run', 'nunique'),
                                 models=('model', lambda s: ','.join(sorted(set(s))))).reset_index()
summary.to_csv(root / 'seed_summary.csv', index=False)

acc = summary[summary.metric.isin(['balanced_accuracy', 'state_accuracy', 'soh_accuracy'])
              & summary.task.isin(['detection_selected', 'diagnosis', 'rul', 'rul_extended', 'soh'])
              & (summary.split != 'test_all_cycles')].copy()
acc['meets_85_on_mean'] = (acc['mean'] >= 0.85).map({True: 'YES', False: 'no'})
acc['meets_85_on_min'] = (acc['min'] >= 0.85).map({True: 'YES', False: 'no'})
pd.set_option('display.width', 220)
print(f'{len(files)} runs aggregated from {root}/')
print(acc.drop(columns=['models']).round(3).to_string(index=False))
print(f'\nFull table: {root / "seed_summary.csv"}')
