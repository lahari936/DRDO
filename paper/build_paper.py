"""Build the IVHM research paper (.docx) on top of the user's IEEE-style template.

Usage (repo root): python paper/build_paper.py "<template.docx>" paper/IVHM_Research_Paper.docx
Keeps the template's title block, author tables, styles, page setup and two-column section; replaces the title text,
the body and the four figure images. Table values are read from the result files, not typed by hand.
"""
import glob
import json
import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

TEMPLATE, OUT = sys.argv[1], sys.argv[2]
HERE = Path(__file__).parent

# ------------------------------------------------------------------ results
S = pd.read_csv('results/final_5seeds/seed_summary_heldout.csv')
def val(sub, task, metric, key='selected', split=None):
    d = S[(S.subsystem == sub) & (S.task == task) & (S.metric == metric) & (S.model_key == key)]
    if split:
        d = d[d.split == split]
    r = d.iloc[0]
    return r['mean'], r['std'], r['min']
def pm(sub, task, metric, key='selected', split=None, nd=3):
    m, s, _ = val(sub, task, metric, key, split)
    return f'{m:.{nd}f} ± {s:.{nd}f}'
ALFA = pd.read_csv('experiments/alfa_results/alfa_summary.csv', index_col=0)
ALFA_META = json.loads(Path('experiments/alfa_results/alfa_meta.json').read_text())
ROB = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/robustness/shift_seed*.csv')])
DEP = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/robustness/deployment_seed*.csv')])
VEH = pd.concat([pd.read_csv(f) for f in glob.glob('experiments/robustness/vehicle_seed*.csv')])
def alfa(k):
    return f"{ALFA.loc[k, 'mean']:.3f} ± {ALFA.loc[k, 'std']:.3f}"

# ------------------------------------------------------------------ XML helpers
SP = '<w:spacing w:before="240" w:after="120"/>'
def run(text, b=False, i=False, sz=None, color=None):
    rpr = ''.join(['<w:b/><w:bCs/>' if b else '', '<w:i/><w:iCs/>' if i else '',
                   f'<w:color w:val="{color}"/>' if color else '', f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else ''])
    return f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
def rich(text, **kw):
    # **bold** and *italic* inline markup
    out = []
    for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text):
        if part.startswith('**'):
            out.append(run(part[2:-2], b=True, **kw))
        elif part.startswith('*') and len(part) > 1:
            out.append(run(part[1:-1], i=True, **kw))
        elif part:
            out.append(run(part, **kw))
    return ''.join(out)
def para(content, jc='both', spacing=SP, ind=''):
    return f'<w:p><w:pPr>{spacing}{ind}<w:jc w:val="{jc}"/></w:pPr>{content}</w:p>'
def H(t):      return para(run(t), jc='center')
def SH(t):     return f'<w:p><w:pPr>{SP}<w:rPr><w:i/><w:iCs/></w:rPr></w:pPr>{run(t, i=True)}</w:p>'
def P(t):      return para(rich(t))
def B(t):      return para(run('• ') + rich(t))
def ABS(label, t):
    return para(run(label, b=True, i=True) + run(t, b=True, i=True), spacing='')
def REF(t):    return para(rich(t), spacing='<w:spacing w:after="60"/>', ind='<w:ind w:left="288" w:hanging="288"/>')
def CAP(t):    return para(run(t, i=True, sz=16), jc='center', spacing='<w:spacing w:before="60" w:after="160"/>')
def TCAP(t):   return para(run(t, sz=16), jc='center', spacing='<w:keepNext/><w:spacing w:before="200" w:after="80"/>')

def jpeg_size(path):
    data = Path(path).read_bytes(); i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1; continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack('>HH', data[i + 5:i + 9]); return w, h
        i += 2 + struct.unpack('>H', data[i + 2:i + 4])[0]
    raise ValueError(path)
FIG_ID = [0]
def FIG(rid, path, caption, width_in=3.3):
    w, h = jpeg_size(path); cx = int(width_in * 914400); cy = int(cx * h / w); FIG_ID[0] += 1; n = FIG_ID[0]
    drawing = (f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
               f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{900 + n}" name="Figure {n}"/>'
               '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
               '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{n}" name="Figure {n}"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
               '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
    return para(drawing, jc='center', spacing='<w:spacing w:before="160" w:after="0"/>') + CAP(caption)

def TABLE(header, rows, widths):
    total = sum(widths)
    border = ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="auto"/>' for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
    x = [f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>{border}</w:tblBorders>'
         '<w:tblCellMar><w:left w:w="40" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>']
    x += [f'<w:gridCol w:w="{w}"/>' for w in widths] + ['</w:tblGrid>']
    def cell(t, w, head=False, first=False):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="1F2937"/>' if head else ''
        jc = 'left' if (first and not head) else 'center'
        r = run(t, b=head, sz=15, color='FFFFFF' if head else None)
        return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}</w:tcPr><w:p><w:pPr><w:spacing w:after="0"/><w:jc w:val="{jc}"/></w:pPr>{r}</w:p></w:tc>'
    x.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>' + ''.join(cell(h, w, head=True) for h, w in zip(header, widths)) + '</w:tr>')
    for row in rows:
        x.append('<w:tr>' + ''.join(cell(str(c), w, first=(k == 0)) for k, (c, w) in enumerate(zip(row, widths))) + '</w:tr>')
    x.append('</w:tbl>')
    return ''.join(x)

# ------------------------------------------------------------------ numbers used in text
det_h = pm('hydraulic', 'detection_selected', 'balanced_accuracy')
acc_h = pm('hydraulic:accumulator', 'diagnosis', 'balanced_accuracy')
bat_rul = pm('li_ion_battery', 'rul', 'state_accuracy', split='lobo')
bat_soh = pm('li_ion_battery', 'soh', 'soh_accuracy', split='lobo')
cnn_acc = pm('propulsion_cmapss', 'rul', 'state_accuracy', key='CNN1D', split='test_last_cycle')
cnn_rmse = pm('propulsion_cmapss', 'rul', 'rmse', key='CNN1D', split='test_last_cycle', nd=1)
hgb_rmse = pm('propulsion_cmapss', 'rul', 'rmse', split='test_last_cycle', nd=1)
syn = S[S.data_type == 'SYNTHETIC']
def rng_(task, metric):
    d = syn[(syn.task == task) & (syn.metric == metric)]['mean']; return f'{d.min():.3f}–{d.max():.3f}'
heads = S[((S.task.isin(['detection_selected', 'diagnosis'])) & (S.metric == 'balanced_accuracy')) |
          ((S.task == 'rul') & (S.metric == 'state_accuracy') & (S.split != 'test_all_cycles')) |
          ((S.task == 'soh') & (S.metric == 'soh_accuracy'))]
heads = heads[~((heads.subsystem == 'propulsion_cmapss') & (heads.model_key == 'selected'))]
n_pass, n_all = int((heads['mean'] >= 0.85).sum()), len(heads)
rob = ROB.groupby('scenario')[['detection_balanced_accuracy', 'diagnosis_balanced_accuracy', 'rul_state_accuracy']].mean()
nom, nz, tol, comb = ('nominal (fresh vehicles)', 'sensor noise x2', 'tolerance x2 (+/-8 %)', 'noise x2 + tolerance x2 + 1 % dropouts')
veh = f"{VEH.vehicle_status_accuracy.mean():.3f} ± {VEH.vehicle_status_accuracy.std():.3f}"
lat_d = DEP[DEP.task == 'diagnosis'].groupby('subsystem').latency_ms_median.median()
lat_r = DEP[DEP.task == 'rul'].groupby('subsystem').latency_ms_median.median()

# ------------------------------------------------------------------ body
b = []
b.append(ABS('Abstract—',
    'Integrated Vehicle Health Management (IVHM) for unmanned aerial vehicles (UAVs) needs algorithms that detect faults, '
    'isolate the failing component and estimate remaining useful life (RUL) across many subsystems, yet public UAV fault data '
    'cover only a few of them. This paper presents a leakage-controlled IVHM pipeline for the nine subsystems named in a DRDO '
    'research problem: airframe, environment control, hydraulic, landing gear, fuel, Li-ion battery, electronic, actuator and '
    'electrical systems. Real data (a hydraulic test rig and NASA Li-ion aging cells), a public turbofan benchmark (C-MAPSS) and '
    'physics-informed simulators for the remaining subsystems feed one pipeline with unit-wise splits, training-only model '
    'selection and causal prognostic features. Over five independent seeds, '
    f'{n_pass} of {n_all} headline tasks exceed the 85 % accuracy figure of merit; battery RUL reaches {bat_rul} and hydraulic '
    f'fault detection {det_h}, while accumulator diagnosis ({acc_h}) stays below target. Three further tests probe the limits: '
    f'real fixed-wing UAV flights from the ALFA dataset (fault detection {alfa("detection_balanced_accuracy")}, median detection '
    f'delay {ALFA.loc["median_detection_delay_s", "mean"]:.1f} s), a distribution-shift study in which doubled sensor noise lowers '
    f'synthetic diagnosis from {rob.loc[nom, "diagnosis_balanced_accuracy"]:.2f} to {rob.loc[nz, "diagnosis_balanced_accuracy"]:.2f}, '
    f'and a vehicle-level evaluation with {veh} status accuracy. All code, data links and per-seed results are public.'))
b.append(ABS('Keywords—', 'integrated vehicle health management, UAV, fault detection and diagnosis, remaining useful life, '
             'prognostics, data leakage, C-MAPSS, ALFA dataset, physics-informed simulation'))

b.append(H('I. INTRODUCTION'))
b.append(P('A UAV that loses an actuator, a fuel pump or battery capacity in flight has little margin to recover, and a fleet '
           'operator who replaces parts on a fixed schedule pays for both early removals and missed failures. IVHM addresses this '
           'by monitoring sensor data on every flight, diagnosing which subsystem is degrading and predicting how many flights or '
           'cycles remain before maintenance is needed [1], [2].'))
b.append(P('Building such a system for a complete UAV is hard for a practical reason: labelled fault data exist for very few '
           'aircraft subsystems. Public prognostics datasets cover turbofan engines [3], batteries [4] and industrial hydraulics [5], '
           'and one public dataset records real UAV control-surface and engine failures [6]. Nothing comparable exists for UAV '
           'airframes, environment-control loops, landing gear, fuel systems, power electronics or electrical distribution. Studies '
           'that report high accuracy on such data also often split windows from the same machine across training and test sets, '
           'which inflates results [7].'))
b.append(P('This work was carried out for a DRDO (ADE) research problem that asks for diagnostic and prognostic algorithms for nine '
           'UAV subsystems with a model accuracy above 85 %. Rather than report one favourable number per subsystem, we built a '
           'single pipeline whose evaluation protocol is fixed in advance and applied to every dataset. Each result is labelled as '
           'real, benchmark or synthetic, and we report where the method falls short.'))
b.append(P('The contributions of this paper are:'))
b.append(B('An IVHM pipeline that covers all nine subsystems with fault detection, fault diagnosis and RUL prediction, integrated '
           'into one vehicle-level health assessment.'))
b.append(B('A leakage-controlled protocol: unit-wise splits, model selection by cross-validation on training units only, causal '
           'features, and five independent seeds with mean, standard deviation and worst-seed reporting.'))
b.append(B('Physics-informed simulators for the seven subsystems that lack public data, with physics health indicators and '
           'causal degradation-trend features for prognostics.'))
b.append(B('An honest stress test: validation on real UAV flights (ALFA), a distribution-shift study, inference cost and '
           'vehicle-level accuracy, including the cases that miss the 85 % target.'))

b.append(H('II. LITERATURE REVIEW'))
b.append(P('IVHM grew out of aerospace condition-based maintenance, where diagnostics identify the present fault and prognostics '
           'estimate the time to failure [1], [8], [9]. Data-driven prognostics now dominate the literature, from feature-based '
           'regression to deep networks [9]. The NASA C-MAPSS turbofan simulation [3] became the standard RUL benchmark; '
           'convolutional networks over sliding windows of sensor history [11], [12] reduced test RMSE well below classical '
           'regressors, and evaluation guidelines for prognostics stress metrics beyond RMSE [10].'))
b.append(P('For batteries, the NASA Ames aging dataset [4] is widely used for state-of-health (SOH) and RUL estimation. For '
           'hydraulics, Helwig et al. [5] released a rig with four independently degraded components and showed that the '
           'accumulator is the hardest to classify. For UAVs specifically, the ALFA dataset [6] provides real flights with '
           'injected engine and control-surface failures. Tree ensembles [14], gradient boosting [15] and isolation forests [16] '
           'remain strong baselines on tabular sensor features. Kaufman et al. [7] describe how leakage between training and test '
           'data invalidates reported accuracy, a risk that is acute when consecutive windows from one machine are split randomly.'))
b.append(P('Most published work therefore addresses one subsystem with one dataset. This paper instead evaluates all nine '
           'subsystems requested for a UAV under one protocol, and states for each result whether it rests on real, benchmark or '
           'simulated data.'))

b.append(H('III. METHODOLOGY'))
b.append(SH('A. Subsystems and Data'))
b.append(P('Table I maps each subsystem to its data source. Real data come from the UCI hydraulic rig (2205 load cycles, 17 sensors, '
           'four component conditions) [5] and NASA Li-ion cells B0005, B0006, B0007 and B0018 [4]. Propulsion RUL uses C-MAPSS '
           'FD001 [3]. The seven remaining subsystems use physics-informed simulators written for this project, and real UAV flights '
           f'from ALFA [6] ({sum(ALFA_META["flights_per_type"].values())} labelled flights) validate the actuator and engine-failure case.'))
b.append(TCAP('TABLE I. SUBSYSTEM COVERAGE AND DATA'))
b.append(TABLE(['Subsystem', 'Data source', 'Type'], [
    ['Hydraulic', 'UCI #447 test rig', 'Real'], ['Li-ion battery', 'NASA cells B0005/6/7/18', 'Real'],
    ['Actuators, engine', 'ALFA UAV flights (validation)', 'Real'], ['Propulsion RUL', 'C-MAPSS FD001', 'Benchmark'],
    ['Airframe', 'strain + modal vibration model', 'Synthetic'], ['Environment control', 'avionics cooling loop', 'Synthetic'],
    ['Landing gear', 'oleo strut, tyre, brake', 'Synthetic'], ['Fuel', 'pump, filter, tank', 'Synthetic'],
    ['Battery pack faults', '6S pack model', 'Synthetic'], ['Electronics', 'MOSFET, DC link, solder', 'Synthetic'],
    ['Actuators (EMA)', 'servo with friction, backlash', 'Synthetic'], ['Electrical', '28 V distribution', 'Synthetic']],
    [1500, 2300, 1000]))
b.append(SH('B. Physics-Informed Simulators'))
b.append(P('Each simulator is a lumped-parameter model of its subsystem: modal frequency loss and breathing-crack asymmetry for the '
           'airframe, a heat balance Q = UA·ΔT for the cooling loop, a polytropic gas spring with second-order damping for the '
           'landing-gear strut, a quadratic filter loss and tank mass balance for fuel, an OCV–resistance cell model for the battery '
           'pack, conduction loss and thermal resistance for power electronics, a servo loop with a backlash operator for the '
           'electro-mechanical actuator, and Ohm\'s law with current balance for electrical distribution. Every simulated vehicle has '
           'its own ±4 % manufacturing tolerance, every window its own mission load and ambient temperature, and every channel '
           'Gaussian noise and 0.3 % telemetry dropouts. Fault severity is continuous between 0.2 and 1.0, so mild faults overlap '
           'with normal variation. Run-to-failure fleets follow a damage law d = (k/L)^p with random life L (80–200 flights) and '
           'exponent p (1.5–3), with failure at d = 1.'))
b.append(SH('C. Features'))
b.append(P('For each sensor and window the pipeline computes twelve statistical and spectral features (mean, standard deviation, '
           'extremes, RMS, skewness, kurtosis, slope, peak-to-peak, difference spread, dominant frequency and spectral centroid). '
           'Subsystem-specific physics indicators are added where standard practice defines them: least-squares identification of '
           'inertia, viscous and Coulomb friction for actuators; flow-corrected filter resistance ΔP/Q² and a tank mass-balance '
           'mismatch for fuel; ESR and temperature-corrected on-resistance for electronics. Prognostic tables add causal features '
           'computed only from past and present flights of the same unit: 5- and 15-flight rolling means, drift from the unit\'s '
           'own early-life baseline, and the change over the last 10 flights. Battery RUL uses the capacity measured on completed '
           'discharges and a trend-extrapolated time to end of life (EOL); battery SOH uses only the first 1000 s of each discharge.'))
b.append(SH('D. Models and Selection'))
b.append(P('Detection and diagnosis compare logistic regression, random forest [14] and histogram gradient boosting [15]; the '
           'hydraulic tasks also consider a soft-voting ensemble. RUL compares ridge regression, random forest and gradient boosting; '
           'C-MAPSS adds an MLP and a 1D convolutional network over 30-cycle windows [11], [12]. For every task the model is chosen by '
           'five-fold GroupKFold cross-validation on the training units only, then refit and scored once on held-out units. '
           'Detection can also be derived from the diagnoser as P(fault) = 1 − P(normal); the variant with the better training-CV '
           'score becomes the vehicle detector. An isolation forest [16] trained on healthy data only is reported as an unsupervised '
           'baseline. All models are implemented in scikit-learn [13] and PyTorch [17].'))
b.append(SH('E. Evaluation Protocol'))
b.append(P('Splits are made by unit: simulated vehicle, engine, battery cell, block of 50 consecutive rig cycles, or ALFA flight, '
           'and an assertion checks that no unit appears on both sides. Detection and diagnosis report balanced accuracy. RUL reports '
           'RMSE and MAE together with maintenance-state accuracy, the share of predictions that fall in the correct band (critical, '
           'warning or healthy). Battery SOH reports 1 − MAPE of capacity, using nested leave-one-battery-out. The full pipeline '
           'runs for five seeds; each seed regenerates the synthetic fleets and redraws every split. Results are summarised as mean ± '
           'standard deviation and worst seed.'))

b.append(H('IV. SYSTEM ARCHITECTURE'))
b.append(P('Fig. 1 shows the pipeline. Sensor streams are cleaned, split by unit and converted to features; three model families per '
           'subsystem produce P(fault), the diagnosed mode and RUL. An IVHMSystem object registers every subsystem model and, for a '
           'new flight, returns a per-subsystem table and a vehicle status: CRITICAL when any RUL is at or below its critical band, '
           'CAUTION when a fault is detected or a non-nominal mode is diagnosed, and NORMAL otherwise. Everything runs in one Google '
           'Colab notebook on a free CPU runtime; only the C-MAPSS network benefits from a GPU.'))
b.append(FIG('rId7', HERE / 'fig1_architecture.jpeg', 'Fig. 1. IVHM pipeline: unit-wise preprocessing, per-subsystem detection, diagnosis and prognosis, and vehicle-level integration.'))

b.append(H('V. RESULTS AND ANALYSIS'))
b.append(SH('A. Headline Results'))
b.append(P(f'Fig. 2 and Table II summarise the held-out results over five seeds. {n_pass} of {n_all} headline tasks reach the 85 % '
           f'figure of merit on the mean. On real data, hydraulic cooler, valve and pump diagnosis score 0.95–1.00, critical-fault '
           f'detection {det_h}, battery SOH {bat_soh} and battery RUL {bat_rul} (RMSE {pm("li_ion_battery", "rul", "rmse", split="lobo", nd=1)} cycles). '
           f'On C-MAPSS the 1D-CNN reaches {cnn_acc} state accuracy with test RMSE {cnn_rmse} cycles, against {hgb_rmse} for '
           f'gradient boosting; it was chosen because its validation error on held-out training engines was lower. On the synthetic '
           f'subsystems detection ranges {rng_("detection_selected", "balanced_accuracy")}, diagnosis '
           f'{rng_("diagnosis", "balanced_accuracy")} and RUL state accuracy {rng_("rul", "state_accuracy")}.'))
b.append(FIG('rId8', HERE / 'fig2_headline.jpeg', 'Fig. 2. Held-out score per subsystem and task (mean ± std over five seeds; tick: worst seed). Dashed line: 85 % target.'))
rows = []
for sub, label in [('hydraulic', 'Hydraulic (R)'), ('li_ion_battery', 'Li-ion cell (R)'), ('propulsion_cmapss', 'C-MAPSS (B)'),
                   ('airframe', 'Airframe (S)'), ('ecs', 'ECS (S)'), ('landing_gear', 'Landing gear (S)'), ('fuel', 'Fuel (S)'),
                   ('battery_pack', 'Battery pack (S)'), ('electronics', 'Electronics (S)'), ('actuators', 'Actuators (S)'),
                   ('electrical', 'Electrical (S)')]:
    def get(task, metric, key='selected', split=None):
        try:
            m, s, _ = val(sub, task, metric, key, split); return f'{m:.3f}±{s:.3f}'
        except IndexError:
            return '—'
    if sub == 'hydraulic':
        diag = f'{val("hydraulic:accumulator", "diagnosis", "balanced_accuracy")[0]:.3f}–{val("hydraulic:valve", "diagnosis", "balanced_accuracy")[0]:.3f}*'
        rows.append([label, get('detection_selected', 'balanced_accuracy'), diag, '—'])
    elif sub == 'li_ion_battery':
        rows.append([label, '—', f'SOH {get("soh", "soh_accuracy", split="lobo")}', get('rul', 'state_accuracy', split='lobo')])
    elif sub == 'propulsion_cmapss':
        rows.append([label, '—', '—', get('rul', 'state_accuracy', 'CNN1D', 'test_last_cycle')])
    else:
        rows.append([label, get('detection_selected', 'balanced_accuracy'), get('diagnosis', 'balanced_accuracy'), get('rul', 'state_accuracy')])
b.append(TCAP('TABLE II. HELD-OUT RESULTS, MEAN ± STD OVER FIVE SEEDS'))
b.append(TABLE(['Subsystem', 'Detection', 'Diagnosis', 'RUL state'], rows, [1350, 1150, 1250, 1100]))
b.append(para(run('R: real, B: benchmark, S: synthetic. *Range over the four hydraulic components (accumulator lowest).', i=True, sz=15),
              spacing='<w:spacing w:before="60" w:after="120"/>'))
b.append(SH('B. Development History and Ablation'))
b.append(P('The final design came from four rounds, each judged on training cross-validation and multi-seed means rather than on a '
           'single test score. Deriving detection from the diagnoser raised fuel detection from 0.807 to above 0.94. Actuator '
           'system-identification features lifted actuator diagnosis from 0.780 to 0.948. Causal baseline-drift and degradation-rate '
           'features raised electronics RUL from 0.817 to 0.892 and fuel RUL from 0.843 to 0.901, and capacity-history features '
           'raised battery RUL from 0.782 to 0.873. One change failed: 30-bin hydraulic cycle profiles lowered accumulator accuracy, '
           'and a dedicated cross-validation experiment instead selected the fast pressure, flow and power sensors with a '
           'soft-voting ensemble, which also raised hydraulic detection from 0.883 to 0.915.'))
b.append(SH('C. Validation on Real UAV Flights'))
b.append(P(f'ALFA [6] provides {sum(ALFA_META["flights_per_type"].values())} usable flights of a Carbon-Z fixed-wing UAV '
           f'({ALFA_META["flights_per_type"]["engine"]} engine, {ALFA_META["flights_per_type"]["aileron"]} aileron, '
           f'{ALFA_META["flights_per_type"]["rudder"]} rudder, {ALFA_META["flights_per_type"]["elevator"]} elevator failures and '
           f'{ALFA_META["flights_per_type"]["normal"]} normal flights). Using 2 s windows of attitude tracking error, airspeed, '
           f'throttle, climb rate, inertial data and actuator outputs, with leave-flights-out cross-validation repeated over five '
           f'seeds, fault detection reaches {alfa("detection_balanced_accuracy")} balanced accuracy with a false-alarm rate of '
           f'{ALFA.loc["false_alarm_rate_normal_windows", "mean"]:.3f} per normal window. The median detection delay after failure '
           f'onset is {ALFA.loc["median_detection_delay_s", "mean"]:.1f} s, and 33–34 of 35 failure flights are detected. Five-class '
           f'diagnosis reaches {alfa("diagnosis_balanced_accuracy")}: engine failures are recognised almost perfectly '
           f'(recall {ALFA.loc["recall_engine", "mean"]:.3f}), whereas aileron failures are often mistaken for normal flight '
           f'(recall {ALFA.loc["recall_aileron", "mean"]:.3f}, Fig. 3) because the autopilot partly compensates for a stuck '
           'aileron. Rudder and elevator have only three and two flights, so their recall is uncertain. On real UAV data the '
           'detection result falls just short of the 85 % target.'))
b.append(FIG('rId9', HERE / 'fig3_alfa_confusion.jpeg', 'Fig. 3. ALFA real-flight diagnosis, row-normalised confusion matrix (leave-flights-out, seed 1).', width_in=2.7))
b.append(SH('D. Robustness to Distribution Shift'))
b.append(P('To test whether the synthetic models only memorise their simulator, the trained models were applied to new vehicles '
           'generated with settings never seen in training (Fig. 4, Table III). Doubling manufacturing tolerance costs little '
           f'(diagnosis {rob.loc[tol, "diagnosis_balanced_accuracy"]:.3f}, RUL {rob.loc[tol, "rul_state_accuracy"]:.3f}). Doubling '
           f'sensor noise does not: diagnosis falls from {rob.loc[nom, "diagnosis_balanced_accuracy"]:.3f} to '
           f'{rob.loc[nz, "diagnosis_balanced_accuracy"]:.3f}, with the airframe, environment-control and battery-pack diagnosers '
           f'near chance, while RUL degrades only to {rob.loc[nz, "rul_state_accuracy"]:.3f} because its rolling features average '
           'noise out. The diagnosers therefore rely on noise-sensitive features, and noise-randomised training is the clear next step.'))
b.append(FIG('rId10', HERE / 'fig4_robustness.jpeg', 'Fig. 4. Synthetic-subsystem accuracy under distribution shift (mean ± std over 8 subsystems and 5 seeds).'))
b.append(TCAP('TABLE III. MEAN ACCURACY UNDER DISTRIBUTION SHIFT'))
b.append(TABLE(['Scenario', 'Detection', 'Diagnosis', 'RUL state'],
               [[lab, f'{rob.loc[k, "detection_balanced_accuracy"]:.3f}', f'{rob.loc[k, "diagnosis_balanced_accuracy"]:.3f}',
                 f'{rob.loc[k, "rul_state_accuracy"]:.3f}'] for lab, k in
                [('Nominal (new vehicles)', nom), ('Noise ×2', nz), ('Tolerance ×2', tol), ('Combined + 1 % dropouts', comb)]],
               [1850, 1000, 1000, 1000]))
b.append(SH('E. Deployment Cost and Vehicle-Level Accuracy'))
b.append(P(f'Single-sample inference on one server CPU core takes {lat_d.min():.1f}–{lat_d.max():.1f} ms per diagnosis model and '
           f'{lat_r.min():.1f}–{lat_r.max():.1f} ms per RUL model, well inside a once-per-flight or once-per-second health update. '
           f'Random-forest RUL models are the largest at about 7 MB each and are the obvious candidates for compression on an '
           f'embedded computer. Across 500 virtual vehicles assembled from fresh simulated data, the IVHM status rule matches the '
           f'true vehicle status in {veh} of cases and the per-subsystem status in '
           f'{VEH.subsystem_status_accuracy.mean():.3f} of cases. With eight subsystems per vehicle nearly every vehicle carries at '
           'least one fault or near-end-of-life part, so this test exercises CAUTION and CRITICAL far more than NORMAL.'))
b.append(SH('F. Does the Work Meet the Problem Criteria?'))
b.append(P('The problem asks for algorithms, data generation, trained and tested models, source code and test and analysis reports '
           f'for nine subsystems with accuracy above 85 %. All deliverables exist and all nine subsystems are covered. On held-out '
           f'data the target is met for {n_pass} of {n_all} headline tasks. It is not met for hydraulic accumulator diagnosis, for '
           'real-UAV fault detection on ALFA (0.835) and diagnosis (0.737), or for synthetic diagnosis under doubled sensor noise. '
           'The 24 synthetic passes show that the algorithms recover the simulated physics; they are not evidence of field performance.'))
b.append(SH('G. Limitations'))
b.append(B('Seven subsystems are validated only on simulators whose constants are engineering estimates, not vendor or flight data.'))
b.append(B('Real data come from an industrial rig, lab-cycled 18650 cells, a turbofan simulation and one small fixed-wing UAV; '
           'battery RUL rests on three cells that reach end of life.'))
b.append(B('Accumulator diagnosis scores 0.86 in training cross-validation but 0.77 on held-out blocks, because the rig changed the '
           'accumulator in long runs and held-out blocks contain rare operating combinations.'))
b.append(B('Maintenance-state accuracy depends on the chosen bands; RMSE and MAE are therefore always reported alongside.'))

b.append(H('VI. DISCUSSION AND FUTURE WORK'))
b.append(P('Two results matter more than the headline count. First, the protocol changed conclusions: random window splits on the '
           'hydraulic rig would have reported higher accumulator accuracy, and single-run test scores near 85 % flipped between pass '
           'and fail across seeds. Second, the stress tests located the real weaknesses: diagnosis under unseen sensor noise, and '
           'control-surface failures that a closed-loop autopilot masks. Future work follows directly:'))
b.append(B('Train with randomised sensor noise and tolerance (domain randomisation) and re-run the shift study.'))
b.append(B('Calibrate the simulators against DRDO/ADE rig or flight logs and re-run the pipeline unchanged on those tables.'))
b.append(B('Add control-residual features, such as the gap between commanded and achieved roll rate, to separate aileron failures '
           'from autopilot corrections on ALFA.'))
b.append(B('Quantify RUL uncertainty with quantile models and compress the random-forest RUL models for embedded hardware.'))

b.append(H('VII. CONCLUSION'))
b.append(P('This paper presented a leakage-controlled IVHM pipeline that delivers fault detection, diagnosis and RUL prediction for '
           f'all nine UAV subsystems in the problem statement and combines them into one vehicle-level assessment. Over five seeds, '
           f'{n_pass} of {n_all} headline tasks exceed 85 %, including battery RUL ({bat_rul}), hydraulic fault detection ({det_h}) '
           f'and C-MAPSS RUL with a 1D-CNN ({cnn_acc}). Real UAV flights and a distribution-shift study show where the method falls '
           'short and what to fix next. Every number in this paper is generated from public code and data.'))

b.append(H('REFERENCES'))
refs = [
    '[1] I. K. Jennions, Ed., *Integrated Vehicle Health Management: Perspectives on an Emerging Field*. Warrendale, PA, USA: SAE International, 2011.',
    '[2] K. Goebel, M. Daigle, A. Saxena, I. Roychoudhury, S. Sankararaman, and J. R. Celaya, *Prognostics: The Science of Making Predictions*. CreateSpace, 2017.',
    '[3] A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in *Proc. Int. Conf. Prognostics and Health Management (PHM)*, 2008.',
    '[4] B. Saha and K. Goebel, "Battery data set," NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.',
    '[5] N. Helwig, E. Pignanelli, and A. Schütze, "Condition monitoring of a complex hydraulic system using multivariate statistics," in *Proc. IEEE Int. Instrumentation and Measurement Technology Conf. (I2MTC)*, 2015.',
    '[6] A. Keipour, M. Mousaei, and S. Scherer, "ALFA: A dataset for UAV fault and anomaly detection," *Int. J. Robotics Research*, vol. 40, no. 2–3, pp. 515–520, 2021.',
    '[7] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowledge Discovery from Data*, vol. 6, no. 4, 2012.',
    '[8] A. K. S. Jardine, D. Lin, and D. Banjevic, "A review on machinery diagnostics and prognostics implementing condition-based maintenance," *Mechanical Systems and Signal Processing*, vol. 20, no. 7, pp. 1483–1510, 2006.',
    '[9] Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," *Mechanical Systems and Signal Processing*, vol. 104, pp. 799–834, 2018.',
    '[10] A. Saxena, J. Celaya, E. Balaban, K. Goebel, B. Saha, S. Saha, and M. Schwabacher, "Metrics for evaluating performance of prognostic techniques," in *Proc. Int. Conf. Prognostics and Health Management (PHM)*, 2008.',
    '[11] G. Sateesh Babu, P. Zhao, and X.-L. Li, "Deep convolutional neural network based regression approach for estimation of remaining useful life," in *Proc. Int. Conf. Database Systems for Advanced Applications (DASFAA)*, 2016.',
    '[12] X. Li, Q. Ding, and J.-Q. Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks," *Reliability Engineering & System Safety*, vol. 172, pp. 1–11, 2018.',
    '[13] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
    '[14] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.',
    '[15] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.',
    '[16] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2008.',
    '[17] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.',
]
b += [REF(r) for r in refs]

# ------------------------------------------------------------------ assemble
zin = zipfile.ZipFile(TEMPLATE)
doc = zin.read('word/document.xml').decode('utf-8')
head, body_and_tail = doc.split('<w:body>', 1)
body, tail = body_and_tail.rsplit('</w:body>', 1)
elems = re.findall(r'(<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>|<w:sectPr.*?</w:sectPr>)', body, re.S)
assert 'w:sectPr' in elems[8] and 'Abstract' in ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', elems[9]))
final_sect = elems[-1]
assert final_sect.startswith('<w:sectPr') and 'w:num="2"' in final_sect
title_run = lambda t: ('<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>'
                       f'<w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>')
new_body = (title_run('Integrated Vehicle Health Management for UAVs: Leakage-Controlled')
            + title_run('Fault Diagnosis and RUL Prediction Across Nine Subsystems')
            + ''.join(elems[2:9]) + ''.join(b) + final_sect)
doc = head + '<w:body>' + new_body + '</w:body>' + tail

media = {'word/media/image1.jpeg': HERE / 'fig1_architecture.jpeg', 'word/media/image2.jpeg': HERE / 'fig2_headline.jpeg',
         'word/media/image3.jpeg': HERE / 'fig3_alfa_confusion.jpeg', 'word/media/image4.jpeg': HERE / 'fig4_robustness.jpeg'}
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename.endswith('/'):
            continue
        if item.filename == 'word/document.xml':
            zout.writestr(item, doc.encode('utf-8'))
        elif item.filename in media:
            zout.writestr(item, Path(media[item.filename]).read_bytes())
        elif item.filename == 'docProps/core.xml':
            core = zin.read(item.filename).decode('utf-8')
            core = re.sub(r'<dc:title>.*?</dc:title>', '<dc:title>IVHM for UAVs</dc:title>', core)
            zout.writestr(item, core.encode('utf-8'))
        else:
            zout.writestr(item, zin.read(item.filename))
print('wrote', OUT, f'({n_pass}/{n_all} headline tasks >= 85 %)')
