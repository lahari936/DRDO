"""Build the submission package from a multi-seed run.

Usage:  python make_submission.py runs          (after: bash run_seeds.sh)

Creates submission/ and IVHM_submission.zip containing:
  Results_Summary.md      final results: mean +/- std, min and max over seeds vs the >85 % figure of merit
  IVHM_Technical_Report.md, README.md, notebook, script, requirements
  models/, figures/       from the first seed run (all seeds use the identical pipeline)
  results/                seed_summary.csv plus each seed's all_results.csv and auto-generated report
Every number in Results_Summary.md is read from the run outputs; nothing is typed in by hand.
"""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TARGET = 0.85
root = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs')
seed_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / 'all_results.csv').exists())
if not seed_dirs:
    sys.exit(f'no seed runs with all_results.csv under {root}/ (run: bash run_seeds.sh)')

df = pd.concat([pd.read_csv(p / 'all_results.csv').assign(run=p.name) for p in seed_dirs])
held = df[df.split.isin(['test', 'test_last_cycle', 'lobo'])].copy()
held['model_key'] = held.model.where(held.model == 'CNN1D', 'selected')
keys = ['subsystem', 'data_type', 'task', 'model_key', 'split', 'metric']
summ = held.groupby(keys).agg(mean=('value', 'mean'), std=('value', 'std'), min=('value', 'min'), max=('value', 'max'),
                              runs=('run', 'nunique'), models=('model', lambda s: ', '.join(sorted(set(s))))).reset_index()
summ['std'] = summ['std'].fillna(0.0)

def headline_rows():
    is_cls = summ.task.isin(['detection_selected', 'diagnosis']) & (summ.metric == 'balanced_accuracy')
    is_rul = (summ.task == 'rul') & (summ.metric == 'state_accuracy')
    is_soh = (summ.task == 'soh') & (summ.metric == 'soh_accuracy')
    h = summ[is_cls | is_rul | is_soh].copy()
    h['task'] = h.task.str.replace('detection_selected', 'detection') + h.model_key.map({'CNN1D': ' (1D-CNN)', 'selected': ''})
    order = {'REAL': 0, 'SIMULATED-BENCHMARK': 1, 'SYNTHETIC': 2}
    return h.sort_values(['data_type', 'subsystem', 'task'], key=lambda c: c.map(order) if c.name == 'data_type' else c)

def md_table(d, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in d[cols].iterrows():
        out.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r) + ' |')
    return '\n'.join(out)

head = headline_rows()
head['mean ± std'] = head['mean'].map('{:.3f}'.format) + ' ± ' + head['std'].map('{:.3f}'.format)
head['meets 85 % (mean)'] = (head['mean'] >= TARGET).map({True: 'YES', False: 'no'})
head['meets 85 % (every seed)'] = (head['min'] >= TARGET).map({True: 'YES', False: 'no'})
head = head.rename(columns={'min': 'worst seed', 'models': 'model(s) selected'})

err = summ[summ.metric.isin(['rmse', 'mae', 'rmse_ah', 'roc_auc', 'macro_f1', 'tolerance_accuracy', 'nasa_score'])].copy()
err['task'] = err.task + err.model_key.map({'CNN1D': ' (1D-CNN)', 'selected': ''})
err['mean ± std'] = err['mean'].map('{:.3f}'.format) + ' ± ' + err['std'].map('{:.3f}'.format)

by_type = {dt: f"{int((d['mean'] >= TARGET).sum())}/{len(d)}" for dt, d in head.groupby('data_type')}
below = head[head['mean'] < TARGET]
fragile = head[(head['mean'] >= TARGET) & (head['worst seed'] < TARGET)]

lines = [
    '# IVHM for UAVs: Final Results Summary',
    f'Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from {len(seed_dirs)} independent runs '
    f'({", ".join(p.name for p in seed_dirs)}). Each run regenerates the synthetic fleet and re-draws every '
    'train/test split with its own seed. Splits are by unit, model choice uses training data only, and the test split is scored once.',
    '',
    '**Figure of merit (problem statement):** accuracy of the model > 85 %. Detection and diagnosis use balanced accuracy. '
    'RUL uses maintenance-state accuracy (critical / warning / healthy band). Battery state of health uses 1 − MAPE of capacity. '
    'Definitions are in the Technical Report, section 6.',
    '',
    '## 1. Headline results (held-out data, mean over seeds)',
    md_table(head, ['subsystem', 'data_type', 'task', 'mean ± std', 'worst seed', 'meets 85 % (mean)',
                    'meets 85 % (every seed)', 'model(s) selected']),
    '',
    f'Tasks meeting the target on the mean, by data type: {by_type}.',
    '',
    '**Below target on the mean:** ' + ('none.' if below.empty else
        '; '.join(f'{r.subsystem} / {r.task} = {r["mean"]:.3f}' for _, r in below.iterrows()) + '.'),
    '',
    '**Meets target on the mean but not in every seed:** ' + ('none.' if fragile.empty else
        '; '.join(f'{r.subsystem} / {r.task} (worst {r["worst seed"]:.3f})' for _, r in fragile.iterrows()) + '.'),
    '',
    '## 2. Error and secondary metrics (mean ± std over seeds)',
    'RUL errors are in flights (synthetic) or cycles (C-MAPSS, battery). Battery capacity errors are in Ah.',
    '',
    md_table(err.sort_values(['data_type', 'subsystem', 'task', 'metric']),
             ['subsystem', 'data_type', 'task', 'split', 'metric', 'mean ± std']),
    '',
    '## 3. How to read these numbers',
    '* **REAL** rows (UCI hydraulic rig, NASA Li-ion cells) are the evidence that the methods work on measured sensor data.',
    '* **SIMULATED-BENCHMARK** (NASA C-MAPSS) is a public, peer-reviewed turbofan simulation and is comparable with the published literature.',
    '* **SYNTHETIC** rows come from this project\'s physics-informed simulators. They show that the algorithms recover the modelled '
    'fault physics. They are **not** evidence of field performance and must be re-validated on DRDO/ADE rig or flight data.',
    '* The "every seed" column is the conservative reading. Quote that column, or the mean ± std, rather than a single best run.',
]
out = Path('submission')
if out.exists():
    shutil.rmtree(out)
(out / 'results').mkdir(parents=True)
(out / 'Results_Summary.md').write_text('\n'.join(lines), encoding='utf-8')
summ.to_csv(out / 'results' / 'seed_summary_heldout.csv', index=False)
for p in seed_dirs:
    shutil.copy(p / 'all_results.csv', out / 'results' / f'{p.name}_all_results.csv')
    rep = p / 'IVHM_Test_and_Analysis_Report.md'
    if rep.exists():
        shutil.copy(rep, out / 'results' / f'{p.name}_Test_and_Analysis_Report.md')
for sub in ('models', 'figures'):
    if (seed_dirs[0] / sub).exists():
        shutil.copytree(seed_dirs[0] / sub, out / sub)
for f in ('docs/IVHM_Technical_Report.md', 'README.md', 'IVHM_UAV_Colab.ipynb', 'ivhm_pipeline.py', 'requirements.txt',
          'run_seeds.sh', 'aggregate_seeds.py', 'make_submission.py'):
    if Path(f).exists():
        shutil.copy(f, out / Path(f).name)
archive = shutil.make_archive('IVHM_submission', 'zip', out)
print('\n'.join(lines[:len(lines)]))
print(f'\nSubmission folder: {out.resolve()}\nArchive: {Path(archive).resolve()}')
