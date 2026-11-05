# IVHM for UAVs - script version of IVHM_UAV_Colab.ipynb (generated from the same cells).
# Run: python ivhm_pipeline.py   (outputs -> ./ivhm_outputs, data -> ./ivhm_data)
import matplotlib
matplotlib.use("Agg")

# %% [2]
import importlib.util, subprocess, sys, os
# Cap OpenMP threads before numpy/sklearn load: oversubscription makes gradient boosting very slow on many-core CPUs.
os.environ.setdefault('OMP_NUM_THREADS', str(min(4, os.cpu_count() or 1)))
PIP_NAMES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn',
             'matplotlib': 'matplotlib', 'joblib': 'joblib'}
missing = [pip for mod, pip in PIP_NAMES.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
print('Packages installed by this cell:', missing or 'none')

# %% [3]
import os, io, json, time, zipfile, platform, warnings, urllib.request
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              IsolationForest)
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)

warnings.filterwarnings('ignore', category=ConvergenceWarning)

IN_COLAB = 'google.colab' in sys.modules
SEED = 42
np.random.seed(SEED)
QUICK_MODE = os.environ.get('IVHM_QUICK', '0') == '1'   # smaller synthetic fleet for a fast smoke test

BASE_DIR = Path('/content') if IN_COLAB else Path.cwd()
DATA_DIR = Path(os.environ.get('IVHM_DATA_DIR', str(BASE_DIR / 'ivhm_data')))
OUT_DIR = Path(os.environ.get('IVHM_OUT_DIR', str(BASE_DIR / 'ivhm_outputs')))
FIG_DIR, MODEL_DIR = OUT_DIR / 'figures', OUT_DIR / 'models'
for d in (DATA_DIR, OUT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

T_START = time.time()
ENV_INFO = {
    'run_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
    'in_colab': IN_COLAB, 'quick_mode': QUICK_MODE, 'python': platform.python_version(),
    'cpu_count': os.cpu_count(), 'numpy': np.__version__, 'pandas': pd.__version__,
    'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
    'data_dir': str(DATA_DIR), 'out_dir': str(OUT_DIR), 'seed': SEED,
}
print(json.dumps(ENV_INFO, indent=2))

# %% [4]
# Run-wide bookkeeping: every metric goes through record() so the final report is built from computed values only.
RESULTS = []
ACC_TARGET = 0.85
HELDOUT_SPLITS = ('test', 'test_last_cycle', 'test_all_cycles', 'lobo')

def record(subsystem, data_type, task, model, metric, value, split='test'):
    RESULTS.append({'subsystem': subsystem, 'data_type': data_type, 'task': task, 'model': model,
                    'split': split, 'metric': metric, 'value': float(value)})

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.show()
    plt.close('all')

# %% [6]
HYD_SENSORS = {'PS1': 100, 'PS2': 100, 'PS3': 100, 'PS4': 100, 'PS5': 100, 'PS6': 100, 'EPS1': 100,
               'FS1': 10, 'FS2': 10, 'TS1': 1, 'TS2': 1, 'TS3': 1, 'TS4': 1, 'VS1': 1,
               'CE': 1, 'CP': 1, 'SE': 1}          # sensor -> sampling rate (Hz), from UCI documentation
BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']

REAL_DATASETS = {
    'cmapss': {
        'url': 'https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip',
        'file': 'cmapss.zip', 'wanted': ['train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt'],
        'data_type': 'SIMULATED-BENCHMARK',
        'source': 'NASA PCoE C-MAPSS turbofan degradation simulation, FD001 (Saxena et al., PHM 2008)'},
    'battery': {
        'url': 'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip',
        'file': 'battery.zip', 'wanted': [f'{b}.mat' for b in BATTERIES],
        'data_type': 'REAL',
        'source': 'NASA PCoE Li-ion 18650 aging tests B0005/6/7/18 (Saha & Goebel, 2007)'},
    'hydraulic': {
        'url': 'https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip',
        'file': 'hydraulic.zip', 'wanted': [f'{s}.txt' for s in HYD_SENSORS] + ['profile.txt'],
        'data_type': 'REAL',
        'source': 'UCI #447, ZeMA hydraulic test rig, 2205 cycles (Helwig et al., 2015)'},
}

def download(url, dest, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f'  cached: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)')
        return True
    tmp = dest.with_suffix('.part')
    for attempt in range(1, retries + 1):
        try:
            print(f'  downloading {dest.name} (attempt {attempt}) ...')
            req = urllib.request.Request(url, headers={'User-Agent': 'ivhm-colab-notebook'})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            print(f'  done: {dest.stat().st_size / 1e6:.1f} MB')
            return True
        except Exception as e:
            print('  failed:', repr(e))
            time.sleep(3 * attempt)
    return False

def extract_matching(zip_src, wanted, out_dir):
    # Recursively walk (nested) zips and extract members whose file name is in `wanted`.
    found = {}
    with zipfile.ZipFile(zip_src) as zf:
        for member in zf.namelist():
            if len(found) == len(wanted):
                break
            base = member.rsplit('/', 1)[-1]
            if base in wanted and base not in found:
                target = out_dir / base
                if not target.exists():
                    target.write_bytes(zf.read(member))
                found[base] = target
            elif member.lower().endswith('.zip'):
                for k, v in extract_matching(io.BytesIO(zf.read(member)), wanted, out_dir).items():
                    found.setdefault(k, v)
    return found

DATA_OK, DATA_FILES = {}, {}
for key, cfg in REAL_DATASETS.items():
    print(f'[{key}] {cfg["source"]}')
    out = DATA_DIR / key
    out.mkdir(exist_ok=True)
    existing = {w: out / w for w in cfg['wanted'] if (out / w).exists()}
    if len(existing) == len(cfg['wanted']):
        DATA_FILES[key], DATA_OK[key] = existing, True
        print('  already extracted')
        continue
    ok = download(cfg['url'], DATA_DIR / cfg['file'])
    if ok:
        try:
            files = extract_matching(DATA_DIR / cfg['file'], set(cfg['wanted']), out)
            miss = set(cfg['wanted']) - set(files)
            if miss:
                print('  missing after extraction:', sorted(miss))
            ok = not miss
            DATA_FILES[key] = files
        except zipfile.BadZipFile as e:
            print('  corrupt archive, deleting so the next run re-downloads:', e)
            (DATA_DIR / cfg['file']).unlink(missing_ok=True)
            ok = False
    DATA_OK[key] = ok
print('\nDataset availability:', DATA_OK)

# %% [7]
# ---- Load C-MAPSS FD001 (SIMULATED-BENCHMARK)
CMAPSS_COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

def read_cmapss(path):
    return pd.read_csv(path, sep=r'\s+', header=None, names=CMAPSS_COLS)

if DATA_OK['cmapss']:
    cm_train = read_cmapss(DATA_FILES['cmapss']['train_FD001.txt'])
    cm_test = read_cmapss(DATA_FILES['cmapss']['test_FD001.txt'])
    cm_rul_final = pd.read_csv(DATA_FILES['cmapss']['RUL_FD001.txt'], sep=r'\s+', header=None, names=['RUL'])
    cm_rul_final['unit'] = np.arange(1, len(cm_rul_final) + 1)
    print('C-MAPSS train', cm_train.shape, '| test', cm_test.shape, '| test RUL labels', cm_rul_final.shape)

# ---- Load NASA Li-ion battery discharge cycles (REAL)
def load_nasa_battery(path):
    name = Path(path).stem
    m = loadmat(str(path), simplify_cells=True)[name]
    cycles, k = [], 0
    for c in m['cycle']:
        if c['type'] != 'discharge':
            continue
        d = c['data']
        k += 1
        cycles.append({'battery': name, 'discharge_idx': k,
                       'ambient_temp': float(c['ambient_temperature']),
                       'capacity_ah': float(np.atleast_1d(d.get('Capacity', np.nan))[0]),
                       't': np.asarray(d['Time'], float), 'v': np.asarray(d['Voltage_measured'], float),
                       'i': np.asarray(d['Current_measured'], float),
                       'temp': np.asarray(d['Temperature_measured'], float)})
    return cycles

if DATA_OK['battery']:
    bat_cycles = [c for b in BATTERIES for c in load_nasa_battery(DATA_FILES['battery'][f'{b}.mat'])]
    print('Battery discharge cycles loaded:', pd.Series([c['battery'] for c in bat_cycles]).value_counts().to_dict())

# ---- Load UCI hydraulic rig (REAL)
if DATA_OK['hydraulic']:
    hyd_raw = {s: pd.read_csv(DATA_FILES['hydraulic'][f'{s}.txt'], sep='\t', header=None).to_numpy(np.float32)
               for s in HYD_SENSORS}
    hyd_profile = pd.read_csv(DATA_FILES['hydraulic']['profile.txt'], sep='\t', header=None,
                              names=['cooler', 'valve', 'pump', 'accumulator', 'stable'])
    print('Hydraulic cycles:', len(hyd_profile), '| sensor shapes:', {k: v.shape for k, v in list(hyd_raw.items())[:3]}, '...')

# %% [9]
def _col(sev, i):
    return sev[:, i:i + 1]

def mission_load(n, T, fs, rng):
    # Normalised load/throttle profile in [0.05, 1] for n windows of T samples.
    t = np.arange(T) / fs
    dur = T / fs
    base = rng.uniform(0.3, 0.9, (n, 1))
    f = rng.uniform(0.5, 3.0, (n, 1)) / dur
    ph = rng.uniform(0, 2 * np.pi, (n, 1))
    L = base + 0.12 * np.sin(2 * np.pi * f * t + ph) + 0.02 * rng.standard_normal((n, T))
    return t, np.clip(L, 0.05, 1.0)

def meas(x, sigma, rng):
    return x + sigma * rng.standard_normal(x.shape)

def sim_airframe(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng)
    cr, dl, lf = _col(sev, 0), _col(sev, 1), _col(sev, 2)
    turb = rng.uniform(0.6, 1.4, (n, 1))
    f1 = 8.0 * uf[:, 0:1] * (1 - 0.10 * cr - 0.02 * dl)      # 1st bending mode (Hz)
    f2 = 23.0 * uf[:, 1:2] * (1 - 0.03 * cr - 0.08 * dl)     # 1st torsion mode (Hz)
    p1, p2 = rng.uniform(0, 2 * np.pi, (2, n, 1))
    m1, m2 = np.sin(2 * np.pi * f1 * t + p1), np.sin(2 * np.pi * f2 * t + p2)
    tip = turb * L * (2.0 * m1 + 0.6 * (1 - 0.5 * dl) * m2) * (1 - 0.2 * dl)
    tip = tip + 0.8 * cr * np.maximum(tip, 0)                 # breathing crack: bilinear stiffness
    spikes = (rng.random((n, T)) < 0.03 * lf) * rng.uniform(1, 3, (n, T)) * rng.choice([-1, 1], (n, T))
    fus = turb * L * 0.5 * m1 + spikes
    s_root = 500 * L * uf[:, 2:3] * (1 + 0.15 * cr) + 40 * turb * m1
    s_fus = 300 * L * uf[:, 3:4] * (1 + 0.08 * lf) + 15 * turb * m1
    return np.stack([meas(s_root, 8, rng), meas(s_fus, 6, rng), meas(tip, 0.15, rng), meas(fus, 0.15, rng)], -1)

def sim_ecs(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng)
    hx, fd, cl, tb = (_col(sev, i) for i in range(4))
    amb = rng.uniform(-10, 40, (n, 1)) + 0 * L
    Q = 150 + 350 * L                                         # heat load (W)
    rpm = 6000 * uf[:, 0:1] * (1 - 0.25 * fd) * (0.6 + 0.4 * L)
    flow = uf[:, 1:2] * (1 - 0.5 * cl)
    UA = 25 * uf[:, 2:3] * (1 - 0.45 * hx) * (0.5 + 0.5 * rpm / 6000) * (0.4 + 0.6 * flow)
    bay = amb + Q / UA
    cin = amb + 0.15 * Q / UA
    cout = cin + Q / (60 * flow)
    cur = 1.2 * (rpm / 6000) ** 2 * uf[:, 3:4] + 0.6 * fd
    dp = 2.0 * flow ** 2 * (1 + 1.5 * hx) + 0 * L
    return np.stack([meas(bay + 4.0 * tb, 0.3, rng), meas(cin, 0.3, rng), meas(cout, 0.3, rng), meas(rpm, 40, rng),
                     meas(cur, 0.03, rng), meas(dp, 0.05, rng), meas(amb, 0.3, rng)], -1)

def sim_landing_gear(sev, uf, rng, T, fs):
    n = len(sev); t = np.arange(T) / fs
    n2, ol, tu, bw = (_col(sev, i) for i in range(4))
    W = rng.uniform(0.6, 1.0, (n, 1)); sink = rng.uniform(0.5, 2.5, (n, 1)); amb = rng.uniform(-10, 40, (n, 1))
    t0 = rng.uniform(0.5, 1.5, (n, 1)); tau = np.clip(t - t0, 0, None); on = (t >= t0)
    P0 = 1200 * uf[:, 0:1] * (1 - 0.35 * n2)                 # strut gas precharge (kPa)
    s_st = 0.06 * (0.6 + 0.4 * W) * (1200 / P0)                # static stroke (m)
    z = np.clip(0.45 * uf[:, 1:2] * (1 - 0.6 * ol), 0.05, 0.95)
    wn = 2 * np.pi * 1.5; wd = wn * np.sqrt(1 - z ** 2); env = np.exp(-z * wn * tau)
    s = on * (s_st * (1 - env * (np.cos(wd * tau) + z / np.sqrt(1 - z ** 2) * np.sin(wd * tau)))
              + 0.03 * sink * env * np.sin(wd * tau))
    s = np.clip(s, 0, 0.2)
    P = P0 * (0.25 / (0.25 - s)) ** 1.3                       # polytropic gas spring
    v0 = 55 * (0.9 + 0.2 * W)
    a = 2.5 * uf[:, 3:4] * (1 - 0.4 * bw)                     # braking deceleration (m/s^2)
    v = np.clip(v0 - a * tau, 0, None)
    wheel = on * v / (0.30 * (1 - 0.05 * tu))
    decel = on * a * (v > 0)
    p_tyre = 1000 * uf[:, 2:3] * (1 - 0.30 * tu) * (1 + 0.05 * W * on)
    brake = amb + 150 * (0.7 + 0.3 * W) * (1 + 0.6 * bw) * (v0 ** 2 - v ** 2) / v0 ** 2
    return np.stack([meas(P, 10, rng), meas(s * 1000, 1.0, rng), meas(p_tyre, 5, rng), meas(brake, 1.0, rng),
                     meas(wheel, 1.0, rng), meas(decel, 0.1, rng)], -1)

def sim_fuel(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng)
    fc, pw, lk, fd = (_col(sev, i) for i in range(4))
    amb = rng.uniform(-10, 40, (n, 1))
    demand = (20 + 60 * L) * uf[:, 0:1]                        # engine fuel draw (L/h)
    leak = 10 * lk                                             # external leak (L/h)
    dp = 15 * (demand / 50) ** 2 * (1 + 3 * fc) * uf[:, 1:2]   # filter dP (kPa)
    p_pump = 300 * uf[:, 2:3] * (1 - 0.25 * pw) - 0.4 * dp - 25 * lk
    cur = (3 + 0.03 * demand * (1 + 0.3 * fc)) * (1 + 0.4 * pw) * uf[:, 3:4]
    level = rng.uniform(40, 90, (n, 1)) - np.cumsum((demand + leak) / 3600 / fs, axis=1) * 100 / 50  # 50 L tank, %
    ftemp = amb + 5 + 3 * pw + 0 * L
    return np.stack([meas(p_pump, 2, rng), meas(demand * (1 + 0.15 * fd), 0.5, rng), meas(dp, 0.3, rng),
                     meas(level, 0.05, rng), meas(cur, 0.05, rng), meas(ftemp, 0.3, rng)], -1)

def sim_battery_pack(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng)
    ci, hr, isc, tsf = (_col(sev, i) for i in range(4))
    amb = rng.uniform(-10, 40, (n, 1))
    I = 20 * L * uf[:, 0:1]                                    # pack current (A), 6S 10 Ah pack
    soc = rng.uniform(0.5, 1.0, (n, 1)) - np.cumsum(I, axis=1) / (3600 * 10.0 * fs)
    ocv = lambda s: 3.4 + 0.8 * np.clip(s, 0, 1)
    R0 = 0.015 * uf[:, 1:2]
    soc_w = soc - 0.15 * ci - isc * (0.02 + 0.0004 * t)       # weakest cell: imbalance / internal short
    R_w = R0 * (1 + 2.0 * hr)
    v_nom, v_w = ocv(soc) - I * R0, ocv(soc_w) - I * R_w
    temp = amb + 0.25 * (5 * I ** 2 * R0 + I ** 2 * R_w) + 6 * isc
    ph = rng.uniform(0, 2 * np.pi, (n, 1))
    temp_meas = temp + tsf * 3 * np.sin(2 * np.pi * t / 40 + ph) + (0.2 + 1.5 * tsf) * rng.standard_normal((n, T))
    spread = 0.5 + 3.0 * isc + 2.0 * hr * L
    return np.stack([meas(5 * v_nom + v_w, 0.02, rng), meas(I, 0.05, rng), meas(np.maximum(v_nom, v_w), 0.005, rng),
                     meas(np.minimum(v_nom, v_w), 0.005, rng), temp_meas, meas(spread, 0.1, rng)], -1)

def sim_electronics(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng)
    ma, cd, sf = (_col(sev, i) for i in range(3))
    amb = rng.uniform(-10, 40, (n, 1))
    I = 15 * L * uf[:, 0:1]
    Rds = 0.010 * uf[:, 1:2] * (1 + 1.0 * ma)                  # on-resistance (ohm)
    Rth = 2.0 * uf[:, 2:3] * (1 + 0.8 * sf)                    # junction-case thermal resistance (K/W)
    case = amb + 8 + 6 * I ** 2 * Rds * Rth
    rds_meas = 1000 * Rds * (1 + 0.004 * (case - 25))
    esr = 0.02 * uf[:, 3:4] * (1 + 2.0 * cd)
    ripple = I * esr * (1 + 0.1 * np.sin(2 * np.pi * 0.3 * t))
    board = amb + 5 + 0.3 * (case - amb)
    glitch = rng.random((n, T)) < 0.01 * sf                    # intermittent solder-joint opens
    v_sup = 28 - 0.05 * I - 2.0 * glitch
    i_sup = I * (1 + 0.02 * ma) + 1.5 * glitch
    return np.stack([meas(case, 0.3, rng), meas(rds_meas, 0.2, rng), meas(ripple, 0.01, rng), meas(board, 0.3, rng),
                     meas(i_sup, 0.1, rng), meas(v_sup, 0.05, rng)], -1)

def sim_actuator(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng); dt = 1 / fs
    fr, bl, wd, pb = (_col(sev, i) for i in range(4))
    amb = rng.uniform(-10, 40, (n, 1))
    A1, f1 = rng.uniform(5, 15, (n, 1)), rng.uniform(0.2, 1.0, (n, 1))
    A2, f2 = rng.uniform(1, 4, (n, 1)), rng.uniform(1, 2, (n, 1))
    cmd = A1 * np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi, (n, 1))) + A2 * np.sin(2 * np.pi * f2 * t)
    tau = (0.04 * uf[:, 0] * (1 + 0.8 * fr[:, 0]))             # servo time constant (s)
    half_b = 0.75 * bl[:, 0]                                    # half backlash (deg)
    act = np.empty((n, T)); out = cmd[:, 0].copy(); pos = cmd[:, 0].copy()
    for k in range(T):
        diff = cmd[:, k] - out
        out = np.where(diff > half_b, cmd[:, k] - half_b, np.where(diff < -half_b, cmd[:, k] + half_b, out))
        pos = pos + (dt / tau) * (out - pos)
        act[:, k] = pos
    vel = np.gradient(act, dt, axis=1); acc = np.gradient(vel, dt, axis=1)
    kt = 0.1 * uf[:, 1:2] * (1 - 0.35 * wd)                    # torque constant
    torque = 0.0005 * acc + 0.01 * vel * (1 + 2 * fr) + 0.05 * L * act + 0.05 * fr * np.sign(vel)
    current = torque / kt
    m_temp = amb + 0.15 * np.mean(current ** 2, axis=1, keepdims=True) * (1 + 0.3 * wd) + 0 * t
    return np.stack([meas(cmd, 0.01, rng), meas(act + 2.0 * pb, 0.05, rng), meas(current, 0.2, rng),
                     meas(m_temp, 0.3, rng), meas(vel * 100 / 6, 10, rng)], -1)

def sim_electrical(sev, uf, rng, T, fs):
    n = len(sev); t, L = mission_load(n, T, fs, rng)
    cr, io_, il, rd = (_col(sev, i) for i in range(4))
    amb = rng.uniform(-10, 40, (n, 1))
    i_av = 4 * uf[:, 0:1] + 0.3 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi, (n, 1)))
    i_pl = 10 * L * uf[:, 1:2]
    Rc = 0.01 * (1 + 4 * cr)                                    # payload connector resistance (ohm)
    v_bus = 28 * (1 - 0.05 * rd) + 0.4 * rd * np.sin(2 * np.pi * 2.0 * t)
    drop = rng.random((n, T)) < 0.02 * io_
    drop = drop | np.roll(drop, 1, axis=1) | np.roll(drop, 2, axis=1)
    i_pl_eff = i_pl * (~drop)
    v_br = (v_bus - i_pl_eff * Rc) * (~drop)
    i_main = i_av + i_pl_eff + 0.8 * il                         # insulation leakage adds unmetered current
    c_temp = amb + 2 + 15 * i_pl ** 2 * Rc
    return np.stack([meas(v_bus, 0.05, rng), meas(i_main, 0.05, rng), meas(i_av, 0.05, rng),
                     meas(i_pl_eff, 0.05, rng), meas(c_temp, 0.3, rng), meas(v_br, 0.05, rng)], -1)

SYN_SPECS = {
    'airframe': dict(fn=sim_airframe, fs=200, T=512, progressive='crack',
                     sensors=['strain_wing_root', 'strain_fuselage', 'accel_wing_tip', 'accel_fuselage'],
                     faults=['crack', 'delamination', 'loose_fastener']),
    'ecs': dict(fn=sim_ecs, fs=1, T=256, progressive='hx_fouling',
                sensors=['bay_temp', 'coolant_in_temp', 'coolant_out_temp', 'fan_rpm', 'fan_current', 'hx_dp', 'ambient_temp'],
                faults=['hx_fouling', 'fan_degradation', 'coolant_leak', 'temp_sensor_bias']),
    'landing_gear': dict(fn=sim_landing_gear, fs=50, T=500, progressive='brake_wear',
                         sensors=['strut_pressure', 'strut_stroke_mm', 'tyre_pressure', 'brake_temp', 'wheel_speed', 'decel'],
                         faults=['n2_leak', 'oil_leak', 'tyre_underinflation', 'brake_wear']),
    'fuel': dict(fn=sim_fuel, fs=1, T=256, progressive='filter_clog',
                 sensors=['pump_pressure', 'fuel_flow', 'filter_dp', 'tank_level', 'pump_current', 'fuel_temp'],
                 faults=['filter_clog', 'pump_wear', 'leak', 'flow_sensor_drift']),
    'battery_pack': dict(fn=sim_battery_pack, fs=1, T=256, progressive='high_resistance',
                         sensors=['pack_voltage', 'pack_current', 'cell_v_max', 'cell_v_min', 'pack_temp', 'cell_temp_spread'],
                         faults=['cell_imbalance', 'high_resistance', 'internal_short', 'thermal_sensor_fault']),
    'electronics': dict(fn=sim_electronics, fs=10, T=256, progressive='mosfet_aging',
                        sensors=['switch_case_temp', 'rds_on_mohm', 'dc_link_ripple', 'board_temp', 'supply_current', 'supply_voltage'],
                        faults=['mosfet_aging', 'capacitor_degradation', 'solder_fatigue']),
    'actuators': dict(fn=sim_actuator, fs=50, T=512, progressive='friction_increase',
                      sensors=['cmd_pos', 'act_pos', 'motor_current', 'motor_temp', 'motor_speed'],
                      faults=['friction_increase', 'backlash', 'winding_degradation', 'position_sensor_bias']),
    'electrical': dict(fn=sim_electrical, fs=20, T=256, progressive='connector_resistance',
                       sensors=['bus_voltage', 'main_current', 'avionics_current', 'payload_current', 'connector_temp', 'branch_voltage'],
                       faults=['connector_resistance', 'intermittent_open', 'insulation_leakage', 'regulator_drift']),
}

# %% [10]
def simulate(name, sev, uf, rng, chunk=1000):
    sp = SYN_SPECS[name]
    X = np.concatenate([sp['fn'](sev[i:i + chunk], uf[i:i + chunk], rng, sp['T'], sp['fs']).astype(np.float32)
                        for i in range(0, len(sev), chunk)])
    X[rng.random(X.shape) < 0.003] = np.nan        # telemetry dropouts, repaired in Section 4
    return X

def make_fault_dataset(name, n_units, win_per_unit, rng, p_normal=0.4):
    # Labelled windows: class 0 = normal, class k = fault k with severity U(0.2, 1.0).
    F = len(SYN_SPECS[name]['faults'])
    unit = np.repeat(np.arange(n_units), win_per_unit)
    n = len(unit)
    uf = rng.normal(1.0, 0.04, (n_units, 4))[unit]
    label = np.where(rng.random(n) < p_normal, 0, rng.integers(1, F + 1, n))
    severity = np.where(label > 0, rng.uniform(0.2, 1.0, n), 0.0)
    sev = np.zeros((n, F))
    idx = np.flatnonzero(label > 0)
    sev[idx, label[idx] - 1] = severity[idx]
    return {'X': simulate(name, sev, uf, rng), 'label': label, 'severity': severity, 'unit': unit}

def make_rtf_dataset(name, n_units, rng, life_range=(80, 200)):
    # Run-to-failure fleet: progressive fault grows d = (k/L)^p; failure when d reaches 1.0.
    sp = SYN_SPECS[name]
    pidx = sp['faults'].index(sp['progressive'])
    units, cycles, ruls, health = [], [], [], []
    for u in range(n_units):
        life = int(rng.integers(*life_range)); p = rng.uniform(1.5, 3.0)
        k = np.arange(1, life + 1)
        d = np.clip((k / life) ** p * (1 + 0.05 * rng.standard_normal(life)), 0, 1)
        units.append(np.full(life, u)); cycles.append(k); ruls.append(life - k); health.append(d)
    unit, cycle, rul, d = map(np.concatenate, (units, cycles, ruls, health))
    sev = np.zeros((len(unit), len(sp['faults']))); sev[:, pidx] = d
    uf = rng.normal(1.0, 0.04, (n_units, 4))[unit]
    return {'X': simulate(name, sev, uf, rng), 'unit': unit, 'cycle': cycle, 'rul': rul, 'damage': d}

N_UNITS_FAULT, WIN_PER_UNIT, N_UNITS_RTF = (16, 20, 12) if QUICK_MODE else (48, 50, 40)
syn_fault_raw, syn_rtf_raw = {}, {}
t0 = time.time()
for i, name in enumerate(SYN_SPECS):
    syn_fault_raw[name] = make_fault_dataset(name, N_UNITS_FAULT, WIN_PER_UNIT, np.random.default_rng(SEED + i))
    syn_rtf_raw[name] = make_rtf_dataset(name, N_UNITS_RTF, np.random.default_rng(SEED + 100 + i))
    print(f'{name:13s} fault windows {syn_fault_raw[name]["X"].shape} | run-to-failure flights {syn_rtf_raw[name]["X"].shape}')
print(f'Synthetic generation took {time.time() - t0:.1f} s')

# %% [12]
if DATA_OK['cmapss']:
    life = cm_train.groupby('unit').cycle.max()
    print('C-MAPSS FD001 engine life (cycles):'); print(life.describe().round(1).to_string())
    tmp = cm_train.copy(); tmp['RUL'] = tmp.groupby('unit').cycle.transform('max') - tmp.cycle
    corr = tmp[[f's{i}' for i in range(1, 22)] + ['RUL']].corr()['RUL'].drop('RUL').fillna(0)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].hist(life, bins=20, color='tab:blue'); ax[0].set_title('Engine life distribution (train)'); ax[0].set_xlabel('cycles')
    for u in [1, 2, 3, 4]:
        d = cm_train[cm_train.unit == u]; ax[1].plot(d.cycle, d.s11, label=f'unit {u}')
    ax[1].set_title('Sensor s11 (HPC static pressure) to failure'); ax[1].set_xlabel('cycle'); ax[1].legend()
    ax[2].bar(corr.index, corr.values, color=['tab:red' if v < 0 else 'tab:green' for v in corr.values])
    ax[2].set_title('Correlation of each sensor with RUL'); ax[2].tick_params(axis='x', rotation=90)
    savefig('eda_cmapss')

# %% [13]
if DATA_OK['battery']:
    bdf = pd.DataFrame([{k: c[k] for k in ('battery', 'discharge_idx', 'capacity_ah', 'ambient_temp')} | {'duration_s': c['t'].max()}
                        for c in bat_cycles])
    print(bdf.groupby('battery').agg(cycles=('discharge_idx', 'max'), cap_first=('capacity_ah', 'first'),
                                     cap_min=('capacity_ah', 'min'), ambient=('ambient_temp', 'mean')).round(3).to_string())
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for b, d in bdf.groupby('battery'):
        ax[0].plot(d.discharge_idx, d.capacity_ah, label=b)
    ax[0].axhline(1.4, ls='--', c='k', label='EOL 1.4 Ah (30% fade)'); ax[0].set_xlabel('discharge cycle'); ax[0].set_ylabel('Ah')
    ax[0].set_title('NASA Li-ion capacity fade (REAL)'); ax[0].legend()
    b5 = [c for c in bat_cycles if c['battery'] == 'B0005']
    for c in (b5[0], b5[len(b5) // 2], b5[-1]):
        ax[1].plot(c['t'], c['v'], label=f'cycle {c["discharge_idx"]}')
    ax[1].axvline(1000, ls=':', c='grey', label='feature window end'); ax[1].set_xlabel('s'); ax[1].set_ylabel('V')
    ax[1].set_title('B0005 discharge voltage, early vs late life'); ax[1].legend()
    savefig('eda_battery')

# %% [14]
if DATA_OK['hydraulic']:
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.5))
    for a, comp in zip(ax[:4], ['cooler', 'valve', 'pump', 'accumulator']):
        hyd_profile[comp].value_counts().sort_index().plot.bar(ax=a, color='tab:purple'); a.set_title(f'{comp} condition counts')
    for lvl in sorted(hyd_profile.cooler.unique()):
        ax[4].plot(np.arange(60), hyd_raw['TS4'][hyd_profile.cooler.values == lvl].mean(0), label=f'cooler {lvl}%')
    ax[4].set_title('Mean TS4 (cooler temp) per cycle (REAL)'); ax[4].set_xlabel('s'); ax[4].legend()
    savefig('eda_hydraulic')

# %% [15]
# Synthetic signals: the same mission profile replayed with each fault at severity 0.8 (SYNTHETIC)
for name, sp in SYN_SPECS.items():
    F, S = len(sp['faults']), len(sp['sensors'])
    t = np.arange(sp['T']) / sp['fs']
    fig, ax = plt.subplots(1, S, figsize=(3.2 * S, 3))
    for k, lab in enumerate(['normal'] + sp['faults']):
        sev = np.zeros((1, F))
        if k:
            sev[0, k - 1] = 0.8
        x = sp['fn'](sev, np.ones((1, 4)), np.random.default_rng(0), sp['T'], sp['fs'])[0]
        for j in range(S):
            ax[j].plot(t, x[:, j], lw=0.8, label=lab)
    for j in range(S):
        ax[j].set_title(sp['sensors'][j], fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle(f'{name} (SYNTHETIC): normal vs faults', y=1.03)
    savefig(f'eda_synthetic_{name}')

fig, ax = plt.subplots(2, 4, figsize=(18, 6))
for a, name in zip(ax.ravel(), SYN_SPECS):
    r = syn_rtf_raw[name]
    for u in range(5):
        m = r['unit'] == u; a.plot(r['cycle'][m], r['damage'][m], lw=1)
    a.set_title(f'{name}: {SYN_SPECS[name]["progressive"]} damage'); a.set_xlabel('flight')
savefig('eda_synthetic_degradation_paths')

print(pd.DataFrame({n: pd.Series(syn_fault_raw[n]['label']).map(dict(enumerate(['normal'] + SYN_SPECS[n]['faults']))).value_counts()
                    for n in SYN_SPECS}).fillna(0).astype(int).to_string())

# %% [17]
def clean_windows(X):
    bad = ~np.isfinite(X)
    n_bad = int(bad.sum())
    if n_bad:
        X = X.copy(); X[bad] = np.nan
        X = np.where(np.isnan(X), np.nanmedian(X, axis=1, keepdims=True), X)
    return X, n_bad

clean_log = []
for name in SYN_SPECS:
    for tag, store in (('fault', syn_fault_raw), ('rtf', syn_rtf_raw)):
        store[name]['X'], nb = clean_windows(store[name]['X'])
        clean_log.append({'dataset': f'{name}/{tag}', 'type': 'SYNTHETIC', 'values_repaired': nb,
                          'remaining_nan': int(np.isnan(store[name]['X']).sum())})

PARTIAL_T = 1000.0      # seconds of each discharge used for battery features
if DATA_OK['battery']:
    n0 = len(bat_cycles)
    bat_cycles = [c for c in bat_cycles
                  if c['t'].max() >= PARTIAL_T and 0.5 < c['capacity_ah'] < 2.5
                  and np.all(np.isfinite(c['v'])) and np.all(np.isfinite(c['temp'])) and np.all(np.diff(c['t']) > 0)]
    clean_log.append({'dataset': 'nasa_battery', 'type': 'REAL', 'values_repaired': 0, 'remaining_nan': 0,
                      'records_dropped': n0 - len(bat_cycles)})

if DATA_OK['cmapss']:
    sensor_cols = [f's{i}' for i in range(1, 22)]
    CM_SENSORS = [c for c in sensor_cols if cm_train[c].std() > 1e-3]    # decided on TRAIN only
    clean_log.append({'dataset': 'cmapss_fd001', 'type': 'SIMULATED-BENCHMARK',
                      'values_repaired': int(cm_train.isna().sum().sum() + cm_test.isna().sum().sum()), 'remaining_nan': 0,
                      'records_dropped': 0, 'constant_sensors_dropped': ','.join(sorted(set(sensor_cols) - set(CM_SENSORS)))})

if DATA_OK['hydraulic']:
    nb = sum(int((~np.isfinite(v)).sum()) for v in hyd_raw.values())
    clean_log.append({'dataset': 'uci_hydraulic', 'type': 'REAL', 'values_repaired': nb, 'remaining_nan': nb})

print(pd.DataFrame(clean_log).fillna('').to_string(index=False))

def split_by_group(groups, test_size=0.25, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(groups)), groups=groups))
    assert not set(groups[tr]) & set(groups[te]), 'group leakage'
    return tr, te

SPLITS = {name: split_by_group(syn_fault_raw[name]['unit']) for name in SYN_SPECS}
SPLITS_RTF = {name: split_by_group(syn_rtf_raw[name]['unit']) for name in SYN_SPECS}
if DATA_OK['hydraulic']:
    HYD_GROUPS = np.arange(len(hyd_profile)) // 50     # blocks of 50 consecutive rig cycles (adjacent cycles are correlated)
    SPLITS['hydraulic'] = split_by_group(HYD_GROUPS)
print({k: (len(v[0]), len(v[1])) for k, v in SPLITS.items()})

# %% [19]
def window_features(x, fs, prefix):
    x = x.astype(np.float64)
    T = x.shape[1]
    mean = x.mean(1); c = x - mean[:, None]
    m2 = (c ** 2).mean(1) + 1e-12
    t = np.arange(T) / fs; tc = t - t.mean()
    spec = np.abs(np.fft.rfft(c, axis=1)); spec[:, 0] = 0
    freqs = np.fft.rfftfreq(T, 1 / fs)
    f = {'mean': mean, 'std': x.std(1), 'min': x.min(1), 'max': x.max(1), 'rms': np.sqrt((x ** 2).mean(1)),
         'skew': (c ** 3).mean(1) / m2 ** 1.5, 'kurt': (c ** 4).mean(1) / m2 ** 2,
         'slope': (c * tc).sum(1) / (tc ** 2).sum(), 'p2p': x.max(1) - x.min(1),
         'diff_std': np.diff(x, axis=1).std(1), 'dom_freq': freqs[spec.argmax(1)],
         'spec_centroid': (spec * freqs).sum(1) / (spec.sum(1) + 1e-12)}
    return {f'{prefix}__{k}': v for k, v in f.items()}

def tensor_features(X, sensors, fs):
    feats = {}
    for j, s in enumerate(sensors):
        feats.update(window_features(X[:, :, j], fs, s))
    return pd.DataFrame(feats)

def causal_rolling(df, group_col, cols, w, stat='mean'):
    r = df.groupby(group_col)[cols].rolling(w, min_periods=1 if stat == 'mean' else 2)
    r = (r.mean() if stat == 'mean' else r.std()).reset_index(level=0, drop=True).sort_index()
    return r.fillna(0.0).add_suffix(f'_{stat}{w}')

# ---- Synthetic fault tables and run-to-failure tables
FAULT_TASKS, RTF_TABLES, RTF_FEATS = {}, {}, {}
for name, sp in SYN_SPECS.items():
    fr = syn_fault_raw[name]
    Xf = tensor_features(fr['X'], sp['sensors'], sp['fs'])
    FAULT_TASKS[name] = {'X': Xf, 'y': fr['label'], 'groups': fr['unit'], 'classes': ['normal'] + sp['faults'],
                         'nominal': 'normal', 'data_type': 'SYNTHETIC'}
    rr = syn_rtf_raw[name]
    base = tensor_features(rr['X'], sp['sensors'], sp['fs'])
    meta = pd.DataFrame({'unit': rr['unit'], 'age': rr['cycle'], 'rul': rr['rul']})
    tab = pd.concat([meta, base], axis=1)
    tab = pd.concat([tab, causal_rolling(tab, 'unit', list(base.columns), 5)], axis=1)
    RTF_TABLES[name] = tab
    RTF_FEATS[name] = [c for c in tab.columns if c not in ('unit', 'rul')]
    fr['X'] = rr['X'] = None          # free raw tensors
    print(f'{name:13s} fault features {Xf.shape} | RUL features {tab[RTF_FEATS[name]].shape}')

# ---- Hydraulic (REAL)
HYD_COMPONENTS = {'cooler': 100, 'valve': 100, 'pump': 0, 'accumulator': 130}     # nominal (healthy) values
# UCI documentation: levels at or beyond these are 'severe' / 'close to total failure'.
# Only ~1 % of rig cycles are fully nominal, so detection targets critical faults (balanced, operationally meaningful).
HYD_CRITICAL = {'cooler': lambda v: v <= 3, 'valve': lambda v: v <= 80, 'pump': lambda v: v >= 2, 'accumulator': lambda v: v <= 100}
if DATA_OK['hydraulic']:
    hyd_X = pd.concat([pd.DataFrame(window_features(hyd_raw[s], fs, s)) for s, fs in HYD_SENSORS.items()], axis=1)
    hyd_any_fault = np.zeros(len(hyd_profile), int)
    for comp, is_crit in HYD_CRITICAL.items():
        hyd_any_fault |= is_crit(hyd_profile[comp].values).astype(int)
    FAULT_TASKS['hydraulic'] = {'X': hyd_X, 'y': hyd_any_fault, 'groups': HYD_GROUPS,
                                'classes': ['no_critical_fault', 'critical_fault'], 'nominal': 'no_critical_fault', 'data_type': 'REAL'}
    del hyd_raw
    print('hydraulic features', hyd_X.shape, '| cycles with a critical fault:', int(hyd_any_fault.sum()), 'of', len(hyd_any_fault))

# ---- Battery (REAL): partial-discharge health indicators
def battery_features(c):
    t, v, temp, i = c['t'], c['v'], c['temp'], c['i']
    m = t <= PARTIAL_T
    grid = np.array([100.0, 250.0, 500.0, 750.0, 1000.0])
    vg, tg = np.interp(grid, t, v), np.interp(grid, t, temp)
    f = {f'v_{int(g)}s': x for g, x in zip(grid, vg)}
    f.update({'v_mean': v[m].mean(), 'v_slope': np.polyfit(t[m], v[m], 1)[0], 'temp_rise': tg[-1] - temp[0],
              'temp_mean': temp[m].mean(), 'i_mean': i[m].mean(), 'ambient_temp': c['ambient_temp']})
    return f

if DATA_OK['battery']:
    bat_tab = pd.DataFrame([{'battery': c['battery'], 'age': c['discharge_idx'], 'capacity_ah': c['capacity_ah']} | battery_features(c)
                            for c in bat_cycles]).reset_index(drop=True)
    BAT_BASE = [c for c in bat_tab.columns if c not in ('battery', 'age', 'capacity_ah')]
    bat_tab = pd.concat([bat_tab, causal_rolling(bat_tab, 'battery', BAT_BASE, 5)], axis=1)
    BAT_FEATS = BAT_BASE + [f'{c}_mean5' for c in BAT_BASE]
    print('battery feature table', bat_tab.shape)

# ---- C-MAPSS (SIMULATED-BENCHMARK)
RUL_CAP_CM = 125
def cmapss_table(df):
    df = df.sort_values(['unit', 'cycle']).reset_index(drop=True)
    out = pd.concat([df[['unit', 'cycle'] + CM_SENSORS], causal_rolling(df, 'unit', CM_SENSORS, 10, 'mean'),
                     causal_rolling(df, 'unit', CM_SENSORS, 10, 'std')], axis=1)
    return out.rename(columns={'cycle': 'age'})

if DATA_OK['cmapss']:
    cm_tr_tab = cmapss_table(cm_train)
    cm_tr_tab['rul'] = cm_tr_tab.groupby('unit').age.transform('max') - cm_tr_tab.age
    cm_te_tab = cmapss_table(cm_test)
    last = cm_te_tab.groupby('unit').age.transform('max')
    cm_te_tab['rul'] = cm_te_tab.unit.map(cm_rul_final.set_index('unit').RUL) + (last - cm_te_tab.age)
    CM_FEATS = [c for c in cm_tr_tab.columns if c not in ('unit', 'rul')]
    print('C-MAPSS train table', cm_tr_tab.shape, '| test table', cm_te_tab.shape)

# %% [21]
def clf_candidates():
    return {'LogReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            'RandomForest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
            'HistGB': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=SEED)}

def reg_candidates(with_mlp=False):
    c = {'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
         'RandomForest': RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
         'HistGB': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)}
    if with_mlp:
        c['MLP'] = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                max_iter=300, random_state=SEED))
    return c

def select_and_fit(cands, X, y, groups, scoring, n_splits=5):
    # Model selection by GroupKFold CV on the training data only, then refit the winner on all training data.
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    cv_scores = {n: float(np.nanmean(cross_val_score(m, X, y, groups=groups, cv=cv, scoring=scoring))) for n, m in cands.items()}
    best = max(cv_scores, key=cv_scores.get)
    return best, clone(cands[best]).fit(X, y), cv_scores

DET_MODELS = {}
t0 = time.time()
for name, task in FAULT_TASKS.items():
    X, g, dt = task['X'], task['groups'], task['data_type']
    y = (task['y'] > 0).astype(int)
    tr, te = SPLITS[name]
    best, model, cvs = select_and_fit(clf_candidates(), X.iloc[tr], y[tr], g[tr], 'balanced_accuracy')
    for m_name, s in cvs.items():
        record(name, dt, 'detection', m_name, 'balanced_accuracy', s, split='cv_train')
    p = model.predict_proba(X.iloc[te])[:, 1]; yhat = (p >= 0.5).astype(int)
    for metric, val in [('accuracy', accuracy_score(y[te], yhat)), ('balanced_accuracy', balanced_accuracy_score(y[te], yhat)),
                        ('precision', precision_score(y[te], yhat, zero_division=0)), ('recall', recall_score(y[te], yhat)),
                        ('f1', f1_score(y[te], yhat))]:
        record(name, dt, 'detection', best, metric, val)
    if len(np.unique(y[te])) > 1:
        record(name, dt, 'detection', best, 'roc_auc', roc_auc_score(y[te], p))
    healthy_tr = tr[y[tr] == 0]
    if len(healthy_tr) < 10:
        print(f'{name}: only {len(healthy_tr)} healthy training windows, Isolation Forest baseline skipped')
        DET_MODELS[name] = {'model': model, 'best': best, 'iso': None, 'iso_thr': None, 'cv': cvs}
        continue
    iso = make_pipeline(StandardScaler(), IsolationForest(n_estimators=200, random_state=SEED)).fit(X.iloc[healthy_tr])
    thr = np.quantile(-iso.decision_function(X.iloc[healthy_tr]), 0.95)
    s_te = -iso.decision_function(X.iloc[te]); y_iso = (s_te > thr).astype(int)
    record(name, dt, 'detection_unsupervised', 'IsolationForest', 'accuracy', accuracy_score(y[te], y_iso))
    record(name, dt, 'detection_unsupervised', 'IsolationForest', 'balanced_accuracy', balanced_accuracy_score(y[te], y_iso))
    if len(np.unique(y[te])) > 1:
        record(name, dt, 'detection_unsupervised', 'IsolationForest', 'roc_auc', roc_auc_score(y[te], s_te))
    DET_MODELS[name] = {'model': model, 'best': best, 'iso': iso, 'iso_thr': float(thr), 'cv': cvs}
    print(f'{name:13s} [{dt}] best={best:12s} test acc={accuracy_score(y[te], yhat):.3f} '
          f'bal_acc={balanced_accuracy_score(y[te], yhat):.3f} | IsolationForest bal_acc={balanced_accuracy_score(y[te], y_iso):.3f}')
print(f'Detection took {time.time() - t0:.0f} s')

# %% [23]
DIAG_MODELS, DIAG_CM = {}, {}
DIAG_TASKS = {n: t for n, t in FAULT_TASKS.items() if t['data_type'] == 'SYNTHETIC'}
if DATA_OK['hydraulic']:
    for comp, nominal in HYD_COMPONENTS.items():
        levels = sorted(hyd_profile[comp].unique())
        DIAG_TASKS[f'hydraulic:{comp}'] = {'X': hyd_X, 'y': np.searchsorted(levels, hyd_profile[comp].values),
                                           'groups': HYD_GROUPS, 'classes': [f'{comp}={l}' for l in levels],
                                           'nominal': f'{comp}={nominal}', 'data_type': 'REAL', 'split_key': 'hydraulic'}

t0 = time.time()
for name, task in DIAG_TASKS.items():
    X, y, g, dt = task['X'], task['y'], task['groups'], task['data_type']
    tr, te = SPLITS[task.get('split_key', name)]
    best, model, cvs = select_and_fit(clf_candidates(), X.iloc[tr], y[tr], g[tr], 'balanced_accuracy')
    for m_name, s in cvs.items():
        record(name, dt, 'diagnosis', m_name, 'balanced_accuracy', s, split='cv_train')
    yhat = model.predict(X.iloc[te])
    for metric, val in [('accuracy', accuracy_score(y[te], yhat)), ('balanced_accuracy', balanced_accuracy_score(y[te], yhat)),
                        ('macro_f1', f1_score(y[te], yhat, average='macro'))]:
        record(name, dt, 'diagnosis', best, metric, val)
    DIAG_MODELS[name] = {'model': model, 'best': best, 'classes': task['classes'], 'nominal': task['nominal'], 'cv': cvs}
    DIAG_CM[name] = (y[te], yhat, task['classes'])
    print(f'{name:22s} [{dt}] best={best:12s} test acc={accuracy_score(y[te], yhat):.3f} '
          f'bal_acc={balanced_accuracy_score(y[te], yhat):.3f} macroF1={f1_score(y[te], yhat, average="macro"):.3f}')
print(f'Diagnosis took {time.time() - t0:.0f} s')

# %% [24]
names = list(DIAG_CM)
cols = 4; rows = int(np.ceil(len(names) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows))
for a in axes.ravel()[len(names):]:
    a.axis('off')
for a, name in zip(axes.ravel(), names):
    yt, yp, classes = DIAG_CM[name]
    labels = list(range(len(classes)))
    ConfusionMatrixDisplay(confusion_matrix(yt, yp, labels=labels, normalize='true'),
                           display_labels=[c[:12] for c in classes]).plot(ax=a, colorbar=False, values_format='.2f')
    a.set_title(f'{name} [{DIAG_TASKS[name]["data_type"]}]', fontsize=9); a.tick_params(axis='x', rotation=60, labelsize=7)
    a.tick_params(axis='y', labelsize=7)
savefig('diagnosis_confusion_matrices')

# %% [26]
def health_state(rul, crit, warn):
    return np.digitize(rul, [crit + 1e-9, warn + 1e-9])      # 0 critical, 1 warning, 2 healthy

def rul_metrics(y, yhat, tol, crit, warn):
    return {'rmse': float(np.sqrt(mean_squared_error(y, yhat))), 'mae': float(mean_absolute_error(y, yhat)),
            'r2': float(r2_score(y, yhat)), 'tolerance_accuracy': float(np.mean(np.abs(yhat - y) <= tol)),
            'state_accuracy': float(accuracy_score(health_state(y, crit, warn), health_state(yhat, crit, warn)))}

def nasa_score(y, yhat):
    d = yhat - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

RUL_CAP_SYN, RUL_TOL_SYN, CRIT_SYN, WARN_SYN = 100, 15, 20, 50
RUL_MODELS, RUL_PRED = {}, {}
t0 = time.time()
for name in SYN_SPECS:
    tab, feats = RTF_TABLES[name], RTF_FEATS[name]
    y = np.minimum(tab['rul'].to_numpy(float), RUL_CAP_SYN); g = tab['unit'].to_numpy(); tr, te = SPLITS_RTF[name]
    best, model, cvs = select_and_fit(reg_candidates(), tab.iloc[tr][feats], y[tr], g[tr], 'neg_root_mean_squared_error')
    for m_name, s in cvs.items():
        record(name, 'SYNTHETIC', 'rul', m_name, 'rmse', -s, split='cv_train')
    yhat = np.clip(model.predict(tab.iloc[te][feats]), 0, RUL_CAP_SYN)
    for k, v in rul_metrics(y[te], yhat, RUL_TOL_SYN, CRIT_SYN, WARN_SYN).items():
        record(name, 'SYNTHETIC', 'rul', best, k, v)
    RUL_MODELS[name] = {'model': model, 'best': best, 'feats': feats, 'unit': 'flights', 'crit': CRIT_SYN}
    RUL_PRED[name] = (g[te], tab.iloc[te]['age'].to_numpy(), y[te], yhat)
    m = rul_metrics(y[te], yhat, RUL_TOL_SYN, CRIT_SYN, WARN_SYN)
    print(f'{name:13s} best={best:12s} RMSE={m["rmse"]:.1f} MAE={m["mae"]:.1f} R2={m["r2"]:.3f} '
          f'state_acc={m["state_accuracy"]:.3f} tol_acc={m["tolerance_accuracy"]:.3f}')
print(f'Synthetic RUL took {time.time() - t0:.0f} s')

fig, ax = plt.subplots(2, 4, figsize=(18, 7))
for a, name in zip(ax.ravel(), SYN_SPECS):
    u, age, yt, yp = RUL_PRED[name]
    for uu in np.unique(u)[:3]:
        m = u == uu
        a.plot(age[m], yt[m], 'k-', lw=1); a.plot(age[m], yp[m], '.', ms=3, label=f'unit {uu}')
    a.set_title(f'{name} (SYNTHETIC) test units'); a.set_xlabel('flight'); a.set_ylabel('RUL (capped)')
    a.legend(fontsize=7)
savefig('rul_synthetic_test_units')

# %% [27]
# ---- C-MAPSS FD001 (SIMULATED-BENCHMARK): train on official train set, test on official test set
CRIT_CM, WARN_CM, TOL_CM = 30, 75, 15
if DATA_OK['cmapss']:
    t0 = time.time()
    y_tr = np.minimum(cm_tr_tab['rul'].to_numpy(float), RUL_CAP_CM)
    best, cm_model, cvs = select_and_fit(reg_candidates(with_mlp=True), cm_tr_tab[CM_FEATS], y_tr,
                                         cm_tr_tab['unit'].to_numpy(), 'neg_root_mean_squared_error')
    for m_name, s in cvs.items():
        record('propulsion_cmapss', 'SIMULATED-BENCHMARK', 'rul', m_name, 'rmse', -s, split='cv_train')
    print('CV RMSE (train units):', {k: round(-v, 2) for k, v in cvs.items()}, '-> best', best)
    yhat_all = np.clip(cm_model.predict(cm_te_tab[CM_FEATS]), 0, RUL_CAP_CM)
    y_all = np.minimum(cm_te_tab['rul'].to_numpy(float), RUL_CAP_CM)
    is_last = (cm_te_tab.age == cm_te_tab.groupby('unit').age.transform('max')).to_numpy()
    for split, msk in (('test_last_cycle', is_last), ('test_all_cycles', np.ones_like(is_last))):
        for k, v in rul_metrics(y_all[msk], yhat_all[msk], TOL_CM, CRIT_CM, WARN_CM).items():
            record('propulsion_cmapss', 'SIMULATED-BENCHMARK', 'rul', best, k, v, split=split)
    record('propulsion_cmapss', 'SIMULATED-BENCHMARK', 'rul', best, 'nasa_score', nasa_score(y_all[is_last], yhat_all[is_last]),
           split='test_last_cycle')
    m = rul_metrics(y_all[is_last], yhat_all[is_last], TOL_CM, CRIT_CM, WARN_CM)
    print(f'Official test (last cycle, 100 engines): RMSE={m["rmse"]:.2f} MAE={m["mae"]:.2f} '
          f'state_acc={m["state_accuracy"]:.3f} tol_acc={m["tolerance_accuracy"]:.3f} '
          f'NASA score={nasa_score(y_all[is_last], yhat_all[is_last]):.0f}  ({time.time() - t0:.0f} s)')
    RUL_MODELS['propulsion_cmapss'] = {'model': cm_model, 'best': best, 'feats': CM_FEATS, 'unit': 'cycles', 'crit': CRIT_CM}

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    order = np.argsort(y_all[is_last])
    ax[0].plot(y_all[is_last][order], 'k-', label='true RUL'); ax[0].plot(yhat_all[is_last][order], 'o', ms=3, label='predicted')
    ax[0].set_title(f'C-MAPSS FD001 test engines, last cycle ({best})'); ax[0].set_xlabel('engine (sorted by true RUL)'); ax[0].legend()
    for u in [1, 20, 31]:
        msk = (cm_te_tab.unit == u).to_numpy()
        ax[1].plot(cm_te_tab.age[msk], y_all[msk], 'k-', lw=1); ax[1].plot(cm_te_tab.age[msk], yhat_all[msk], '.', ms=3, label=f'engine {u}')
    ax[1].set_title('Full test trajectories'); ax[1].set_xlabel('cycle'); ax[1].legend()
    savefig('rul_cmapss')
else:
    print('C-MAPSS not available: propulsion RUL NOT RUN')

# %% [28]
# ---- NASA Li-ion battery (REAL): SOH estimation and RUL with nested leave-one-battery-out
EOL_AH, CRIT_BAT, WARN_BAT, TOL_BAT = 1.4, 15, 40, 10
BAT_FOLD_MODELS = {}
if DATA_OK['battery']:
    t0 = time.time()
    soh_true, soh_pred = [], []
    for held in BATTERIES:
        tr = (bat_tab.battery != held).to_numpy(); te = ~tr
        best, model, _ = select_and_fit(reg_candidates(), bat_tab.loc[tr, BAT_FEATS + ['age']], bat_tab.capacity_ah[tr],
                                        bat_tab.battery[tr].to_numpy(), 'neg_root_mean_squared_error')
        pred = model.predict(bat_tab.loc[te, BAT_FEATS + ['age']])
        soh_true.append(bat_tab.capacity_ah[te].to_numpy()); soh_pred.append(pred)
        mape = np.mean(np.abs(pred - bat_tab.capacity_ah[te]) / bat_tab.capacity_ah[te])
        print(f'SOH  held-out {held}: model={best:12s} RMSE={np.sqrt(mean_squared_error(bat_tab.capacity_ah[te], pred)):.4f} Ah '
              f'capacity accuracy (1-MAPE)={1 - mape:.3f}')
    yt, yp = np.concatenate(soh_true), np.concatenate(soh_pred)
    record('li_ion_battery', 'REAL', 'soh', 'nested-LOBO', 'rmse_ah', np.sqrt(mean_squared_error(yt, yp)), split='lobo')
    record('li_ion_battery', 'REAL', 'soh', 'nested-LOBO', 'soh_accuracy', 1 - np.mean(np.abs(yp - yt) / yt), split='lobo')
    record('li_ion_battery', 'REAL', 'soh', 'nested-LOBO', 'r2', r2_score(yt, yp), split='lobo')

    # RUL: only batteries whose measured capacity crosses the EOL threshold have a defined RUL
    eol = {b: d.loc[d.capacity_ah < EOL_AH, 'age'].min() for b, d in bat_tab.groupby('battery')}
    rul_bats = [b for b in BATTERIES if pd.notna(eol[b])]
    print('Batteries reaching EOL (1.4 Ah):', {b: (int(eol[b]) if pd.notna(eol[b]) else 'never') for b in BATTERIES})
    rt = bat_tab[bat_tab.battery.isin(rul_bats)].copy()
    rt['rul'] = rt.battery.map(eol) - rt.age
    rt = rt[rt.rul >= 0]
    bat_rul_pred = {}
    if len(rul_bats) >= 3:
        for held in rul_bats:
            tr = (rt.battery != held).to_numpy(); te = ~tr
            best, model, _ = select_and_fit(reg_candidates(), rt.loc[tr, BAT_FEATS + ['age']], rt.rul[tr],
                                            rt.battery[tr].to_numpy(), 'neg_root_mean_squared_error')
            pred = np.clip(model.predict(rt.loc[te, BAT_FEATS + ['age']]), 0, None)
            BAT_FOLD_MODELS[held] = model
            bat_rul_pred[held] = (rt.age[te].to_numpy(), rt.rul[te].to_numpy(float), pred)
            m = rul_metrics(rt.rul[te].to_numpy(float), pred, TOL_BAT, CRIT_BAT, WARN_BAT)
            print(f'RUL  held-out {held}: model={best:12s} RMSE={m["rmse"]:.1f} cycles, state_acc={m["state_accuracy"]:.3f}, '
                  f'tol_acc={m["tolerance_accuracy"]:.3f}')
        yt = np.concatenate([v[1] for v in bat_rul_pred.values()]); yp = np.concatenate([v[2] for v in bat_rul_pred.values()])
        for k, v in rul_metrics(yt, yp, TOL_BAT, CRIT_BAT, WARN_BAT).items():
            record('li_ion_battery', 'REAL', 'rul', 'nested-LOBO', k, v, split='lobo')
        final_best, bat_final, _ = select_and_fit(reg_candidates(), rt[BAT_FEATS + ['age']], rt.rul, rt.battery.to_numpy(),
                                                  'neg_root_mean_squared_error')
        RUL_MODELS['li_ion_battery'] = {'model': bat_final, 'best': final_best, 'feats': BAT_FEATS + ['age'], 'unit': 'cycles', 'crit': CRIT_BAT}
    else:
        print('Fewer than 3 batteries reach EOL: battery RUL NOT RUN (cannot do nested leave-one-out).')
    print(f'Battery modelling took {time.time() - t0:.0f} s')

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for b in BATTERIES:
        ax[0].plot(bat_tab.age[bat_tab.battery == b], soh_true[BATTERIES.index(b)], 'k-', lw=1)
        ax[0].plot(bat_tab.age[bat_tab.battery == b], soh_pred[BATTERIES.index(b)], '.', ms=3, label=f'{b} predicted')
    ax[0].set_title('Capacity estimation, leave-one-battery-out (REAL)'); ax[0].set_xlabel('discharge cycle'); ax[0].set_ylabel('Ah'); ax[0].legend()
    for b, (age, yt_b, yp_b) in bat_rul_pred.items():
        ax[1].plot(age, yt_b, 'k-', lw=1); ax[1].plot(age, yp_b, '.', ms=3, label=f'{b} predicted')
    ax[1].set_title('RUL to 1.4 Ah, leave-one-battery-out (REAL)'); ax[1].set_xlabel('discharge cycle'); ax[1].set_ylabel('cycles'); ax[1].legend()
    savefig('rul_battery')
else:
    print('Battery data not available: SOH/RUL NOT RUN')

# %% [30]
class IVHMSystem:
    STATUS_ORDER = {'NORMAL': 0, 'CAUTION': 1, 'CRITICAL': 2}

    def __init__(self):
        self.entries = {}

    def register(self, name, data_type, detector=None, diagnoser=None, classes=None, nominal=None,
                 rul_model=None, rul_feats=None, rul_unit=None, rul_crit=None):
        self.entries[name] = dict(data_type=data_type, detector=detector, diagnoser=diagnoser, classes=classes,
                                  nominal=nominal, rul_model=rul_model, rul_feats=rul_feats, rul_unit=rul_unit, rul_crit=rul_crit)

    def assess(self, inputs):
        # inputs: {entry_name: {'fault': 1-row DataFrame of window features, 'rul': 1-row DataFrame of prognostic features}}
        rows = []
        for name, x in inputs.items():
            e = self.entries[name]
            row = {'subsystem': name, 'data_type': e['data_type']}
            status = 'NORMAL'
            if e['detector'] is not None and 'fault' in x:
                row['p_fault'] = float(e['detector'].predict_proba(x['fault'])[0, 1])
                if row['p_fault'] >= 0.5:
                    status = 'CAUTION'
            if e['diagnoser'] is not None and 'fault' in x:
                pr = e['diagnoser'].predict_proba(x['fault'])[0]
                k = int(np.argmax(pr))
                row['diagnosis'] = e['classes'][int(e['diagnoser'].classes_[k])]
                row['diag_conf'] = float(pr[k])
                if row['diagnosis'] != e['nominal']:
                    status = 'CAUTION'
            if e['rul_model'] is not None and 'rul' in x:
                row['rul'] = max(0.0, float(e['rul_model'].predict(x['rul'][e['rul_feats']])[0]))
                row['rul_unit'] = e['rul_unit']
                if row['rul'] <= e['rul_crit']:
                    status = 'CRITICAL'
            row['status'] = status
            rows.append(row)
        report = pd.DataFrame(rows)
        vehicle = max(report['status'], key=self.STATUS_ORDER.get)
        return report, vehicle

ivhm = IVHMSystem()
for name in SYN_SPECS:
    ivhm.register(name, 'SYNTHETIC', detector=DET_MODELS[name]['model'], diagnoser=DIAG_MODELS[name]['model'],
                  classes=DIAG_MODELS[name]['classes'], nominal='normal', rul_model=RUL_MODELS[name]['model'],
                  rul_feats=RUL_MODELS[name]['feats'], rul_unit='flights', rul_crit=CRIT_SYN)
if DATA_OK['hydraulic']:
    ivhm.register('hydraulic', 'REAL', detector=DET_MODELS['hydraulic']['model'])
    for comp in HYD_COMPONENTS:
        d = DIAG_MODELS[f'hydraulic:{comp}']
        ivhm.register(f'hydraulic:{comp}', 'REAL', diagnoser=d['model'], classes=d['classes'], nominal=d['nominal'])
if 'li_ion_battery' in RUL_MODELS:
    demo_bat = list(BAT_FOLD_MODELS)[-1]      # model trained without this battery, demo sample taken from it
    ivhm.register('li_ion_battery', 'REAL', rul_model=BAT_FOLD_MODELS[demo_bat], rul_feats=BAT_FEATS + ['age'],
                  rul_unit='cycles', rul_crit=CRIT_BAT)
if 'propulsion_cmapss' in RUL_MODELS:
    ivhm.register('propulsion_cmapss', 'SIMULATED-BENCHMARK', rul_model=cm_model, rul_feats=CM_FEATS, rul_unit='cycles', rul_crit=CRIT_CM)

# Assemble a virtual vehicle from held-out samples only
demo_rng = np.random.default_rng(7)
inputs, truth = {}, {}
for name in SYN_SPECS:
    te = SPLITS[name][1]; i = int(demo_rng.choice(te))
    tab = RTF_TABLES[name]; j = int(demo_rng.choice(SPLITS_RTF[name][1]))
    inputs[name] = {'fault': FAULT_TASKS[name]['X'].iloc[[i]], 'rul': tab.iloc[[j]]}
    truth[name] = {'true_condition': FAULT_TASKS[name]['classes'][FAULT_TASKS[name]['y'][i]],
                   'true_rul': min(float(tab['rul'].iloc[j]), RUL_CAP_SYN)}
if DATA_OK['hydraulic']:
    i = int(demo_rng.choice(SPLITS['hydraulic'][1]))
    inputs['hydraulic'] = {'fault': hyd_X.iloc[[i]]}
    truth['hydraulic'] = {'true_condition': 'critical_fault' if hyd_any_fault[i] else 'no_critical_fault'}
    for comp in HYD_COMPONENTS:
        inputs[f'hydraulic:{comp}'] = {'fault': hyd_X.iloc[[i]]}
        truth[f'hydraulic:{comp}'] = {'true_condition': f'{comp}={hyd_profile[comp].iloc[i]}'}
if 'li_ion_battery' in ivhm.entries:
    rows_b = rt[rt.battery == demo_bat]
    j = int(demo_rng.integers(len(rows_b)))
    inputs['li_ion_battery'] = {'rul': rows_b.iloc[[j]]}
    truth['li_ion_battery'] = {'true_rul': float(rows_b.rul.iloc[j])}
if 'propulsion_cmapss' in ivhm.entries:
    j = int(demo_rng.integers(len(cm_te_tab)))
    inputs['propulsion_cmapss'] = {'rul': cm_te_tab.iloc[[j]]}
    truth['propulsion_cmapss'] = {'true_rul': min(float(cm_te_tab.rul.iloc[j]), RUL_CAP_CM)}

vehicle_report, vehicle_status = ivhm.assess(inputs)
vehicle_report = vehicle_report.merge(pd.DataFrame(truth).T.rename_axis('subsystem').reset_index(), on='subsystem', how='left')
pd.set_option('display.width', 200)
print(vehicle_report.round(3).to_string(index=False))
print('\nVEHICLE STATUS:', vehicle_status)

# %% [32]
res = pd.DataFrame(RESULTS)
ACC_METRICS = ['accuracy', 'balanced_accuracy', 'state_accuracy', 'tolerance_accuracy', 'soh_accuracy']

cv_table = (res[res.split == 'cv_train'].pivot_table(index=['subsystem', 'task', 'metric'], columns='model', values='value')
            .round(3))
print('Cross-validation on TRAINING units only (balanced accuracy for classification, RMSE for RUL):')
print(cv_table.to_string())

held = res[res.split.isin(HELDOUT_SPLITS)]
summary = held[held.metric.isin(ACC_METRICS + ['macro_f1', 'roc_auc', 'rmse', 'mae', 'r2'])].pivot_table(
    index=['subsystem', 'data_type', 'task', 'model', 'split'], columns='metric', values='value').round(3)
print('\nHeld-out results:')
print(summary.to_string())

# %% [33]
headline = held[((held.task.isin(['detection', 'diagnosis'])) & (held.metric == 'balanced_accuracy')) |
                ((held.task == 'rul') & (held.metric == 'state_accuracy') & (held.split != 'test_all_cycles')) |
                ((held.task == 'soh') & (held.metric == 'soh_accuracy'))].copy()
headline['label'] = headline.subsystem + ' / ' + headline.task
colors = {'REAL': 'tab:green', 'SIMULATED-BENCHMARK': 'tab:orange', 'SYNTHETIC': 'tab:blue'}
headline = headline.sort_values(['data_type', 'label'])
fig, ax = plt.subplots(figsize=(10, 0.32 * len(headline) + 1.5))
ax.barh(headline.label, headline.value, color=headline.data_type.map(colors))
ax.axvline(ACC_TARGET, c='r', ls='--', label='85 % target')
for dt, c in colors.items():
    ax.barh([], [], color=c, label=dt)
ax.set_xlim(0, 1.0); ax.set_xlabel('held-out score (balanced acc / state acc / SOH acc)'); ax.legend(loc='lower right')
ax.set_title('Headline held-out scores by subsystem and data type')
savefig('summary_headline_scores')

det_cmp = held[(held.metric == 'balanced_accuracy') & held.task.isin(['detection', 'detection_unsupervised'])].pivot_table(
    index='subsystem', columns='task', values='value')
det_cmp.plot.bar(figsize=(12, 4), title='Supervised vs unsupervised (Isolation Forest) detection, balanced accuracy')
plt.axhline(ACC_TARGET, c='r', ls='--'); plt.ylim(0, 1.05)
savefig('summary_detection_supervised_vs_unsupervised')

# %% [35]
imp_names = [n for n in DIAG_MODELS]
cols = 3; rows = int(np.ceil(len(imp_names) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.3 * rows))
for a in axes.ravel()[len(imp_names):]:
    a.axis('off')
TOP_FEATURES = {}
for a, name in zip(axes.ravel(), imp_names):
    task = DIAG_TASKS[name]; te = SPLITS[task.get('split_key', name)][1]
    Xte = task['X'].iloc[te]
    pi = permutation_importance(DIAG_MODELS[name]['model'], Xte, task['y'][te], scoring='balanced_accuracy',
                                n_repeats=3, random_state=SEED, n_jobs=1)
    s = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False).head(8)
    TOP_FEATURES[name] = s.index.tolist()
    a.barh(s.index[::-1], s.values[::-1], color='tab:gray'); a.set_title(f'{name} [{task["data_type"]}]', fontsize=9)
    a.tick_params(axis='y', labelsize=7)
savefig('diagnosis_permutation_importance')

# %% [37]
for name, d in DET_MODELS.items():
    joblib.dump({'model': d['model'], 'iso': d['iso'], 'iso_thr': d['iso_thr']}, MODEL_DIR / f'detect_{name.replace(":", "_")}.joblib')
for name, d in DIAG_MODELS.items():
    joblib.dump({'model': d['model'], 'classes': d['classes']}, MODEL_DIR / f'diagnose_{name.replace(":", "_")}.joblib')
for name, d in RUL_MODELS.items():
    joblib.dump({'model': d['model'], 'features': d['feats']}, MODEL_DIR / f'rul_{name}.joblib')
joblib.dump(ivhm, MODEL_DIR / 'ivhm_system.joblib')

model_card = {
    'environment': ENV_INFO,
    'datasets': {k: {'source': v['source'], 'data_type': v['data_type'], 'url': v['url'], 'available': DATA_OK[k]}
                 for k, v in REAL_DATASETS.items()},
    'synthetic_subsystems': {n: {'sensors': s['sensors'], 'faults': s['faults'], 'progressive': s['progressive'],
                                 'fs_hz': s['fs'], 'window_samples': s['T']} for n, s in SYN_SPECS.items()},
    'detection': {n: {'model': d['best'], 'cv_balanced_accuracy': d['cv']} for n, d in DET_MODELS.items()},
    'diagnosis': {n: {'model': d['best'], 'classes': d['classes'], 'cv_balanced_accuracy': d['cv']} for n, d in DIAG_MODELS.items()},
    'rul': {n: {'model': d['best'], 'unit': d['unit'], 'n_features': len(d['feats'])} for n, d in RUL_MODELS.items()},
}
(MODEL_DIR / 'model_card.json').write_text(json.dumps(model_card, indent=2, default=str))

# quick reload check
reloaded = joblib.load(MODEL_DIR / 'ivhm_system.joblib')
assert reloaded.assess(inputs)[1] == vehicle_status
print('Saved', len(list(MODEL_DIR.glob('*.joblib'))), 'model files to', MODEL_DIR, '| reload check passed')

# %% [39]
def md_table(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v) for v in r) + ' |')
    return '\n'.join(out)

res = pd.DataFrame(RESULTS)
res.to_csv(OUT_DIR / 'all_results.csv', index=False)
held = res[res.split.isin(HELDOUT_SPLITS)]
acc = held[held.metric.isin(ACC_METRICS)].copy()
acc['meets_85pct'] = np.where(acc.value >= ACC_TARGET, 'YES', 'no')
acc = acc[['subsystem', 'data_type', 'task', 'model', 'split', 'metric', 'value', 'meets_85pct']].sort_values(['data_type', 'subsystem', 'task', 'metric'])
err = held[held.metric.isin(['rmse', 'mae', 'r2', 'rmse_ah', 'nasa_score', 'macro_f1', 'roc_auc'])][
    ['subsystem', 'data_type', 'task', 'model', 'split', 'metric', 'value']].sort_values(['data_type', 'subsystem', 'task', 'metric'])

head = headline[['subsystem', 'data_type', 'task', 'metric', 'value']].copy()
head['meets_85pct'] = np.where(head.value >= ACC_TARGET, 'YES', 'no')
by_type = {dt: f'{int((d.value >= ACC_TARGET).sum())}/{len(d)}' for dt, d in head.groupby('data_type')}
below = head[head.value < ACC_TARGET]

def most_confused(name):
    yt, yp, classes = DIAG_CM[name]
    cm = confusion_matrix(yt, yp, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    if cm.sum() == 0:
        return 'no errors on test split'
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    return f'{classes[i]} -> {classes[j]} ({cm[i, j]} windows)'

not_run = [k for k, v in DATA_OK.items() if not v]
elapsed = (time.time() - T_START) / 60
lines = [
    '# IVHM for UAVs - Test and Analysis Report',
    f'Generated {ENV_INFO["run_utc"]} | Colab: {IN_COLAB} | quick mode: {QUICK_MODE} | sklearn {sklearn.__version__} | runtime {elapsed:.1f} min',
    '',
    '## 1. Data provenance',
    md_table(pd.DataFrame([{'dataset': k, 'type': v['data_type'], 'available': DATA_OK[k], 'source': v['source']}
                           for k, v in REAL_DATASETS.items()])),
    '',
    f'Synthetic subsystems ({len(SYN_SPECS)}): {", ".join(SYN_SPECS)}. Fleet size per subsystem: '
    f'{N_UNITS_FAULT} units x {WIN_PER_UNIT} labelled windows, {N_UNITS_RTF} run-to-failure units.',
    f'Datasets that could not be downloaded (their tasks were NOT run): {not_run or "none"}.',
    '',
    '## 2. Headline held-out results vs the >85 % figure of merit',
    md_table(head),
    '',
    f'Tasks meeting the target, by data type: {by_type}.',
    '',
    '## 3. All held-out accuracy-type metrics',
    md_table(acc),
    '',
    '## 4. Error metrics (RUL in flights/cycles, battery capacity in Ah)',
    md_table(err),
    '',
    '## 5. Analysis',
    '* Splits are by unit (vehicle, engine, battery, block of rig cycles); model choice used GroupKFold CV on training units only.',
    '* Most frequent diagnosis confusion per subsystem (held-out):',
] + [f'  * {n}: {most_confused(n)}' for n in DIAG_CM] + [
    '* Top features driving each diagnosis model (permutation importance):',
] + [f'  * {n}: {", ".join(f[:4])}' for n, f in TOP_FEATURES.items()] + [
    '* Tasks below 85 %: ' + ('none' if below.empty else '; '.join(f'{r.subsystem}/{r.task} = {r.value:.3f}' for r in below.itertuples())),
    '',
    '## 6. Limitations and risks (apply to every number above)',
    '* SYNTHETIC scores measure recovery of faults from this project\'s simulators. The simulators encode assumed physics, so high scores can be optimistic: real faults may have signatures the models never saw. They must not be cited as field performance.',
    '* Real-data coverage is limited to the hydraulic rig (industrial, not a UAV) and Li-ion cells tested in a lab. C-MAPSS is a turbofan simulation; small-UAV engines differ.',
    '* Battery RUL rests on very few cells (leave-one-battery-out over the cells that reach 1.4 Ah); treat it as a feasibility result.',
    '* Hydraulic rig cycles are correlated in time; block-grouped splits reduce, but do not remove, optimism.',
    '* RUL accuracy figures depend on the stated state thresholds and tolerances; RMSE/MAE are reported alongside.',
    '* Next step: validate on DRDO/ADE flight logs or rig data, calibrate simulator constants against it, and re-run this notebook unchanged with the real tables.',
]
report_path = OUT_DIR / 'IVHM_Test_and_Analysis_Report.md'
report_path.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))

# %% [40]
import shutil
archive = shutil.make_archive(str(BASE_DIR / 'ivhm_outputs'), 'zip', OUT_DIR)
print('Outputs archived to', archive, f'({Path(archive).stat().st_size / 1e6:.1f} MB)')
print(f'Total runtime: {(time.time() - T_START) / 60:.1f} min')
if IN_COLAB:
    from google.colab import files
    files.download(archive)
