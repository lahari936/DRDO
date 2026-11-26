"""Figures for the IVHM paper, drawn only from saved result files.

Usage (repo root): python paper/make_figures.py
Inputs : results/final_5seeds/seed_summary_heldout.csv, experiments/alfa_results/confusion_seed1.csv,
         experiments/robustness/shift_seed*.csv
Outputs: paper/fig1_architecture.jpeg ... paper/fig4_robustness.jpeg (column width, 300 dpi)
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

OUT = Path(__file__).parent
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 7,
                     'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 6.2, 'xtick.labelsize': 6.2,
                     'ytick.labelsize': 6.2, 'savefig.dpi': 300})
COLW = 3.4
COL = {'REAL': '#2e7d32', 'SIMULATED-BENCHMARK': '#ef6c00', 'SYNTHETIC': '#1f5fa8'}

# ---------- Fig. 1: architecture
fig, ax = plt.subplots(figsize=(COLW, 2.25)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis('off')
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.15', fc=fc, ec='#333', lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=4.6, linespacing=1.15)
def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=0.6, color='#333'))
box(0.1, 4.6, 2.2, 1.5, 'Sensors\n9 subsystems\nreal / bench. /\nsynthetic', '#eceff1')
box(2.8, 4.6, 2.0, 1.5, 'Cleaning\nunit-wise\nsplit', '#e3f2fd')
box(5.3, 4.6, 2.0, 1.5, 'Window +\nphysics +\ncausal trend\nfeatures', '#e3f2fd')
box(7.8, 5.45, 2.1, 0.8, 'Detection P(fault)', '#e8f5e9')
box(7.8, 4.45, 2.1, 0.8, 'Diagnosis (mode)', '#e8f5e9')
box(7.8, 3.45, 2.1, 0.8, 'Prognosis (RUL)', '#e8f5e9')
box(3.2, 1.5, 3.6, 1.3, 'IVHMSystem.assess()\nper-subsystem status\nvehicle: NORMAL /\nCAUTION / CRITICAL', '#fff3e0')
box(0.1, 1.5, 2.6, 1.3, 'Model choice:\nGroupKFold CV\non training units', '#fce4ec')
box(7.3, 1.5, 2.6, 1.3, 'Outputs: models,\n5-seed metrics,\nreports', '#eceff1')
arrow(2.3, 5.35, 2.8, 5.35); arrow(4.8, 5.35, 5.3, 5.35)
for y in (5.85, 4.85, 3.85):
    arrow(7.3, 5.35, 7.8, y)
arrow(8.85, 3.45, 6.8, 2.5); arrow(2.7, 2.15, 3.2, 2.15); arrow(6.8, 2.15, 7.3, 2.15)
ax.text(5, 0.7, 'offline: training & evaluation  |  online: assess() per flight', ha='center', fontsize=4.8, style='italic')
fig.savefig(OUT / 'fig1_architecture.jpeg', bbox_inches='tight', pil_kwargs={'quality': 95}); plt.close(fig)

# ---------- Fig. 2: headline results
s = pd.read_csv('results/final_5seeds/seed_summary_heldout.csv')
m = ((s.task.isin(['detection_selected', 'diagnosis']) & (s.metric == 'balanced_accuracy')) |
     ((s.task == 'rul') & (s.metric == 'state_accuracy') & (s.split != 'test_all_cycles')) |
     ((s.task == 'soh') & (s.metric == 'soh_accuracy')))
h = s[m].copy()
h = h[~((h.subsystem == 'propulsion_cmapss') & (h.model_key == 'selected'))]          # deployed C-MAPSS model = 1D-CNN
h['label'] = (h.subsystem.str.replace('hydraulic:', 'hyd. ').str.replace('_', ' ') + ' / ' +
              h.task.str.replace('detection_selected', 'det.').str.replace('diagnosis', 'diag.').str.replace('rul', 'RUL').str.replace('soh', 'SOH'))
order = {'REAL': 0, 'SIMULATED-BENCHMARK': 1, 'SYNTHETIC': 2}
h = h.sort_values(['data_type', 'label'], key=lambda c: c.map(order) if c.name == 'data_type' else c).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(COLW, 4.6))
yy = np.arange(len(h))[::-1]
ax.barh(yy, h['mean'], xerr=h['std'], color=h.data_type.map(COL), height=0.7, error_kw=dict(lw=0.5, capsize=1.2))
ax.scatter(h['min'], yy, marker='|', color='k', s=12, lw=0.8, zorder=3)
ax.axvline(0.85, color='red', ls='--', lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels(h.label); ax.set_xlim(0.6, 1.0); ax.set_xlabel('held-out score (mean ± std, 5 seeds)')
handles = [Patch(color=c, label=dt.title().replace('-', ' ')) for dt, c in COL.items()] + [
    Line2D([], [], color='red', ls='--', lw=0.8, label='85 % target'),
    Line2D([], [], color='k', marker='|', ls='', label='worst of 5 seeds')]
ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.35, 1.0), ncol=3, frameon=False, fontsize=5.6)
fig.savefig(OUT / 'fig2_headline.jpeg', bbox_inches='tight', pil_kwargs={'quality': 95}); plt.close(fig)

# ---------- Fig. 3: ALFA confusion matrix (real UAV flights)
cm = pd.read_csv('experiments/alfa_results/confusion_seed1.csv', index_col=0)
fig, ax = plt.subplots(figsize=(COLW * 0.75, 2.3))
im = ax.imshow(cm.values, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=30); ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f'{cm.values[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if cm.values[i, j] > 0.55 else 'black')
ax.set_xlabel('predicted'); ax.set_ylabel('true')
fig.colorbar(im, fraction=0.046, pad=0.04)
fig.savefig(OUT / 'fig3_alfa_confusion.jpeg', bbox_inches='tight', pil_kwargs={'quality': 95}); plt.close(fig)

# ---------- Fig. 4: robustness under distribution shift
r = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/robustness/shift_seed*.csv')])
scen = ['nominal (fresh vehicles)', 'sensor noise x2', 'tolerance x2 (+/-8 %)', 'noise x2 + tolerance x2 + 1 % dropouts']
agg = r.groupby('scenario')[['detection_balanced_accuracy', 'diagnosis_balanced_accuracy', 'rul_state_accuracy']].agg(['mean', 'std']).reindex(scen)
fig, ax = plt.subplots(figsize=(COLW, 1.9))
x = np.arange(len(scen)); w = 0.26
for k, (col, lab, c) in enumerate([('detection_balanced_accuracy', 'detection', '#1f5fa8'),
                                   ('diagnosis_balanced_accuracy', 'diagnosis', '#5e92d1'),
                                   ('rul_state_accuracy', 'RUL state', '#ef6c00')]):
    ax.bar(x + (k - 1) * w, agg[(col, 'mean')], w, yerr=agg[(col, 'std')], color=c, label=lab, error_kw=dict(lw=0.5, capsize=1.2))
ax.axhline(0.85, color='red', ls='--', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['nominal', 'noise x2', 'tolerance x2', 'combined'])
ax.set_ylim(0.5, 1.0); ax.set_ylabel('mean over 8 subsystems')
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.savefig(OUT / 'fig4_robustness.jpeg', bbox_inches='tight', pil_kwargs={'quality': 95}); plt.close(fig)
print('figures written to', OUT)
