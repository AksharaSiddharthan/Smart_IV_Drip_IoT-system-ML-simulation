"""
Smart IV Drip Failure & Backflow Detection System
==================================================
REAL-TIME SIMULATION
====================
Just simulates sensor data coming in one by one, like how the ESP32 would
actually read it: grab data → do math → check rules → ML classifier → alert.

Run:
    python simulation.py

Options:
    python simulation.py --scenario all        # Run all scenarios (default)
    python simulation.py --scenario normal
    python simulation.py --scenario empty
    python simulation.py --scenario blockage
    python simulation.py --scenario backflow
    python simulation.py --speed fast          # No pauses
    python simulation.py --speed slow          # Half second delay (realistic)
"""

import argparse
import time
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
np.random.seed(None)   # Different noise every run

WINDOW = 10   # How many readings we keep around

def _noise(signal, std=1.0, spike_prob=0.005):
    """Add some random noise + occasional spikes (cuz sensors are messy)"""
    n = len(signal)
    noisy = signal + np.random.normal(0, std, n)
    mask  = np.random.random(n) < spike_prob
    noisy[mask] += np.random.normal(0, std * 8, mask.sum())
    return noisy

def _smooth(s, w=5):
    """Smooth out the wiggles"""
    return np.convolve(s, np.ones(w)/w, mode='same')

def _nab_flatmiddle(n):
    """Sine wave that goes flat in the middle"""
    t = np.linspace(0, 4*np.pi, n)
    s = 50 + 20*np.sin(t) + np.random.normal(0, 2, n)
    a, b = int(n*0.35), int(n*0.65)
    s[a:b] = np.random.normal(0, 0.5, b-a)
    return s

def _nab_jumpsdown(n):
    """Normal, then suddenly drops"""
    s = np.random.normal(50, 3, n)
    j = int(n*0.4)
    s[j:] = np.random.normal(8, 2, n-j)
    return s

def _nab_jumpsup(n):
    """Normal, then suddenly spikes"""
    s = np.random.normal(50, 3, n)
    j = int(n*0.4)
    s[j:] = np.random.normal(95, 4, n-j)
    return s

def _nab_machine_temp(n):
    """Simulates a gradual heat-up"""
    s  = np.random.normal(50, 3, n)
    ds = int(n*0.5); de = int(n*0.75)
    s[ds:de] += np.linspace(0, 30, de-ds)
    s[de:]    = np.random.normal(95, 6, n-de)
    return s

# ── Scenario data generators ────────────────────────────────────────────────
def scenario_normal(n=120):
    """Everything's fine, drip flowing normally"""
    d = np.clip(_noise(np.full(n, 20.0), 1.5), 10, 35)
    p = _smooth(_noise(np.full(n, 50.0), 3.0), 7)
    w = _noise(np.linspace(500, 440, n), 0.8)
    return d, p, w, ['normal']*n

def scenario_empty(n=100):
    """Bag's empty, everything drops"""
    nab = _nab_flatmiddle(n)
    d   = np.clip(np.interp(nab, (nab.min(), nab.max()), (0, 20)), 0, 25)
    p   = _smooth(_noise(np.full(n, 20.0), 2.5) * np.linspace(1.0, 0.4, n), 5)
    w   = np.clip(_noise(np.linspace(80, 20, n), 1.0), 0, 100)
    return d, p, w, ['empty']*n

def scenario_blockage(n=100):
    """Line gets clogged, pressure builds up"""
    nd  = _nab_jumpsdown(n)
    np_ = _nab_jumpsup(n)
    d   = np.interp(nd,  (nd.min(),  nd.max()),  (1, 22))
    p   = _smooth(np.interp(np_, (np_.min(), np_.max()), (50, 100)), 5)
    nm  = _nab_machine_temp(n)
    w   = _smooth(np.interp(nm, (nm.min(), nm.max()), (280, 360)), 5)
    return d, p, w, ['blockage']*n

def scenario_backflow(n=80):
    """Blood's flowing backwards — bad sign"""
    d  = np.clip(_noise(np.zeros(n), 0.3), 0, 2)
    pb = np.linspace(-5, -25, n)
    p  = _smooth(_noise(pb, 3.0), 5)
    w  = np.clip(_noise(np.linspace(35, 18, n), 1.0), 0, 50)
    return d, p, w, ['backflow']*n

# ── Keep a rolling window of recent readings ────────────────────────────────
class RollingBuffer:
    """Keeps the last 10 readings so we can calc averages"""
    def __init__(self, w=WINDOW):
        self.w = w
        self.d = deque(maxlen=w)
        self.p = deque(maxlen=w)
        self.wt = deque(maxlen=w)

    def push(self, drip, pressure, weight):
        self.d.append(drip)
        self.p.append(pressure)
        self.wt.append(weight)

    def stats(self):
        """Get the rolling averages and std devs"""
        return {
            'drip_rolling_mean':     np.mean(self.d),
            'drip_rolling_std':      np.std(self.d) if len(self.d) > 1 else 0,
            'pressure_rolling_mean': np.mean(self.p),
            'pressure_rolling_std':  np.std(self.p) if len(self.p) > 1 else 0,
            'weight_rolling_mean':   np.mean(self.wt),
        }

# Just the numbers we care about
THRESHOLDS = {
    'drip_low':       3.0,
    'pressure_high':  80.0,
    'pressure_neg':   0.0,
    'weight_low':     60.0,
    'weight_critical':30.0,
}

def rule_classify(stats):
    """Simple rules — if X and Y and Z, then it's probably this"""
    d  = stats['drip_rolling_mean']
    p  = stats['pressure_rolling_mean']
    w  = stats['weight_rolling_mean']

    if d < THRESHOLDS['drip_low'] and p < THRESHOLDS['pressure_neg'] and w < THRESHOLDS['weight_low']:
        return 'backflow',  'RULE'
    if d < THRESHOLDS['drip_low'] and w < THRESHOLDS['weight_critical']:
        return 'empty',     'RULE'
    if d < THRESHOLDS['drip_low'] and p > THRESHOLDS['pressure_high']:
        return 'blockage',  'RULE'
    return 'uncertain', 'RULE'

# ── ML model (trained once at the start) ──────────────────────────────────
FEATURE_COLS = [
    'drip_rate', 'pressure', 'weight',
    'drip_rolling_mean', 'drip_rolling_std', 'drip_delta',
    'pressure_rolling_mean', 'pressure_rolling_std', 'pressure_delta',
    'weight_delta', 'weight_rolling_mean',
    'pres_drip_ratio', 'pressure_negative', 'low_weight_zero_drip'
]

def train_rf():
    """Train the random forest on a bunch of fake data"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Generate a bunch of training examples
    segs = [scenario_normal(600), scenario_normal(300),
            scenario_empty(500), scenario_blockage(500),
            scenario_backflow(400)]

    all_d, all_p, all_w, all_l = [], [], [], []
    for d, p, w, l in segs:
        all_d.extend(d); all_p.extend(p)
        all_w.extend(w); all_l.extend(l)

    df = pd.DataFrame({'drip_rate': all_d, 'pressure': all_p,
                       'weight': all_w, 'label': all_l})

    # Calc all the features
    win = 10
    df['drip_rolling_mean']     = df['drip_rate'].rolling(win, min_periods=1).mean()
    df['drip_rolling_std']      = df['drip_rate'].rolling(win, min_periods=1).std().fillna(0)
    df['drip_delta']            = df['drip_rate'].diff().fillna(0)
    df['pressure_rolling_mean'] = df['pressure'].rolling(win, min_periods=1).mean()
    df['pressure_rolling_std']  = df['pressure'].rolling(win, min_periods=1).std().fillna(0)
    df['pressure_delta']        = df['pressure'].diff().fillna(0)
    df['weight_delta']          = df['weight'].diff().fillna(0)
    df['weight_rolling_mean']   = df['weight'].rolling(win, min_periods=1).mean()
    df['pres_drip_ratio']       = df['pressure'] / (df['drip_rate'] + 1e-3)
    df['pressure_negative']     = (df['pressure'] < 0).astype(int)
    df['low_weight_zero_drip']  = ((df['weight'] < 60) & (df['drip_rate'] < 2)).astype(int)

    le  = LabelEncoder()
    le.fit(['normal', 'empty', 'blockage', 'backflow'])
    y   = le.transform(df['label'])

    # Train it up
    clf = RandomForestClassifier(n_estimators=150, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    clf.fit(df[FEATURE_COLS], y)
    return clf, le

# Terminal colors and icons
COLORS = {
    'normal':   '\033[92m',   # green
    'empty':    '\033[93m',   # yellow
    'blockage': '\033[91m',   # red
    'backflow': '\033[95m',   # magenta
    'uncertain':'\033[94m',   # blue
    'RESET':    '\033[0m',
    'BOLD':     '\033[1m',
    'DIM':      '\033[2m',
    'CYAN':     '\033[96m',
    'WHITE':    '\033[97m',
}

ALERT_ICONS = {
    'normal':   '✓',
    'empty':    '⚠',
    'blockage': '✖',
    'backflow': '🔴',
}

STATE_DESCRIPTIONS = {
    'normal':   'Infusion proceeding normally',
    'empty':    'ALERT: Bottle empty — replace immediately',
    'blockage': 'ALERT: Line occlusion detected — check IV line',
    'backflow': 'CRITICAL: Blood backflow detected — urgent intervention',
}

def bar(val, lo, hi, width=20, fill='█', empty='░'):
    """Draw a little progress bar thing"""
    pct  = max(0, min(1, (val - lo) / (hi - lo)))
    filled = int(pct * width)
    return fill * filled + empty * (width - filled)

def color_for(state):
    """Get the right color code for a state"""
    return COLORS.get(state, COLORS['RESET'])

def print_header():
    """Fancy header at the start"""
    c = COLORS
    print(f"\n{c['BOLD']}{c['CYAN']}{'═'*70}")
    print(f"   SMART IV DRIP FAILURE & BACKFLOW DETECTION — REAL-TIME SIMULATION")
    print(f"{'═'*70}{c['RESET']}")
    print(f"{c['DIM']}   Hybrid Engine: Rule-Based Pre-filter → Random Forest Classifier")
    print(f"   Sensor loop: 1 Hz  |  Rolling window: {WINDOW} samples{c['RESET']}\n")

def print_reading(t, sample, drip, pressure, weight, stats,
                  rule_result, rf_result, final_state, confidence, source):
    """Print out one reading"""
    c     = COLORS
    col   = color_for(final_state)
    icon  = ALERT_ICONS.get(final_state, '?')
    desc  = STATE_DESCRIPTIONS.get(final_state, '')
    ts    = datetime.now().strftime('%H:%M:%S.%f')[:12]

    # Sensor display with bars
    drip_bar  = bar(drip,     0,  40)
    pres_bar  = bar(pressure, -30, 110)
    wt_bar    = bar(weight,   0,  520)

    drip_col  = c['RESET'] if drip > 5   else c['empty'] if drip > 0 else c['blockage']
    pres_col  = c['backflow'] if pressure < 0 else c['blockage'] if pressure > 80 else c['RESET']
    wt_col    = c['empty'] if weight < 60 else c['RESET']

    print(f"{c['DIM']}[{ts}] Sample #{sample:04d}{c['RESET']}")
    print(f"  {c['WHITE']}Drip  {drip_col}{drip:6.1f} dpm  {drip_bar}{c['RESET']}")
    print(f"  {c['WHITE']}Pres  {pres_col}{pressure:6.1f} mmHg {pres_bar}{c['RESET']}")
    print(f"  {c['WHITE']}Wt    {wt_col}{weight:6.1f} g    {wt_bar}{c['RESET']}")

    # Show what the engine decided
    src_tag  = f"[{source}]"
    conf_str = f"  conf={confidence:.2f}" if source == 'RF' else ''
    print(f"  {c['DIM']}Rule→{c['RESET']} {rule_result:<10}  "
          f"{c['DIM']}RF→{c['RESET']} {rf_result:<10}")
    print(f"  {col}{c['BOLD']}{icon} {final_state.upper():<10}{src_tag}{conf_str}  "
          f"{desc}{c['RESET']}")
    print(f"  {c['DIM']}{'─'*64}{c['RESET']}")

def print_scenario_banner(name, n):
    """Big banner for a new scenario"""
    c = COLORS
    col = color_for(name)
    print(f"\n{c['BOLD']}{col}{'▶'*3} SCENARIO: {name.upper()} ({n} samples) {'◀'*3}{c['RESET']}\n")

def print_scenario_summary(name, counts, total):
    """Summary of how we did on this scenario"""
    c = COLORS
    col = color_for(name)
    correct = counts.get(name, 0)
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n{c['BOLD']}  Scenario Summary — {name.upper()}{c['RESET']}")
    print(f"  Total readings : {total}")
    print(f"  {col}Correct detections: {correct} ({acc:.1f}%){c['RESET']}")
    for state, count in sorted(counts.items()):
        tag = ' ← correct' if state == name else ''
        print(f"    {color_for(state)}{state:<12} {count:>4} ({count/total*100:5.1f}%){tag}{c['RESET']}")

def print_final_summary(all_counts, all_totals):
    """Big summary at the end"""
    c = COLORS
    states = ['normal', 'empty', 'blockage', 'backflow']
    print(f"\n{c['BOLD']}{c['CYAN']}{'═'*70}")
    print(f"   FULL RUN SUMMARY")
    print(f"{'═'*70}{c['RESET']}")
    total_correct = 0
    total_all = 0
    for s in states:
        tot = all_totals.get(s, 0)
        cor = all_counts.get(s, {}).get(s, 0)
        if tot == 0: continue
        acc = cor / tot * 100
        total_correct += cor
        total_all += tot
        col = color_for(s)
        b = bar(acc, 0, 100, width=25)
        print(f"  {col}{s:<12} {cor:>4}/{tot:<4} = {acc:5.1f}%  {b}{c['RESET']}")
    if total_all > 0:
        oa = total_correct / total_all * 100
        print(f"\n  {c['BOLD']}Overall accuracy : {total_correct}/{total_all} = {oa:.1f}%{c['RESET']}")
    print(f"{c['CYAN']}{'═'*70}{c['RESET']}\n")

def run_scenario(name, drips, pressures, weights, labels,
                 clf, le, delay, show_every=1):
    """Run through one scenario"""
    buf = RollingBuffer(WINDOW)
    prev_d = prev_p = prev_w = 0.0
    counts = {}
    total  = 0

    print_scenario_banner(name, len(drips))

    for i, (d, p, w, true_lbl) in enumerate(zip(drips, pressures, weights, labels)):
        buf.push(d, p, w)
        stats = buf.stats()

        # Calculate some derived stuff
        drip_delta    = d - prev_d
        pres_delta    = p - prev_p
        wt_delta      = w - prev_w
        pres_drip_r   = p / (d + 1e-3)
        pres_neg      = int(p < 0)
        low_wt_zd     = int(w < 60 and d < 2)

        # Try the rule engine first
        rule_result, _ = rule_classify(stats)

        # Build features for the ML model
        feat_row = pd.DataFrame([{
            'drip_rate':            d,
            'pressure':             p,
            'weight':               w,
            'drip_rolling_mean':    stats['drip_rolling_mean'],
            'drip_rolling_std':     stats['drip_rolling_std'],
            'drip_delta':           drip_delta,
            'pressure_rolling_mean':stats['pressure_rolling_mean'],
            'pressure_rolling_std': stats['pressure_rolling_std'],
            'pressure_delta':       pres_delta,
            'weight_delta':         wt_delta,
            'weight_rolling_mean':  stats['weight_rolling_mean'],
            'pres_drip_ratio':      pres_drip_r,
            'pressure_negative':    pres_neg,
            'low_weight_zero_drip': low_wt_zd,
        }])

        # Get the RF prediction
        rf_probs   = clf.predict_proba(feat_row)[0]
        rf_pred_id = np.argmax(rf_probs)
        rf_result  = le.classes_[rf_pred_id]
        confidence = rf_probs[rf_pred_id]

        # Decide: rules get priority for obvious cases
        if rule_result != 'uncertain':
            final_state = rule_result
            source      = 'RULE'
        else:
            final_state = rf_result
            source      = 'RF'

        counts[final_state] = counts.get(final_state, 0) + 1
        total += 1

        if i % show_every == 0:
            print_reading(datetime.now(), i+1, d, p, w, stats,
                          rule_result, rf_result, final_state,
                          confidence, source)

        if delay > 0:
            time.sleep(delay)

        prev_d, prev_p, prev_w = d, p, w

    print_scenario_summary(name, counts, total)
    return counts, total

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='IV Drip Simulation')
    parser.add_argument('--scenario', default='all',
                        choices=['all', 'normal', 'empty', 'blockage', 'backflow'])
    parser.add_argument('--speed', default='normal',
                        choices=['fast', 'normal', 'slow'])
    parser.add_argument('--show_every', type=int, default=3,
                        help='Print every Nth sample (default 3)')
    args = parser.parse_args()

    delays = {'fast': 0, 'normal': 0.08, 'slow': 0.5}
    delay  = delays[args.speed]

    print_header()
    print(f"  Training RF classifier on synthetic dataset...")
    clf, le = train_rf()
    print(f"  RF ready. Starting simulation (speed={args.speed})...\n")

    scenarios = {
        'normal':   scenario_normal(80),
        'empty':    scenario_empty(80),
        'blockage': scenario_blockage(80),
        'backflow': scenario_backflow(60),
    }

    all_counts = {}
    all_totals = {}

    if args.scenario == 'all':
        order = ['normal', 'empty', 'blockage', 'backflow']
    else:
        order = [args.scenario]

    for name in order:
        d, p, w, l = scenarios[name]
        counts, total = run_scenario(name, d, p, w, l, clf, le,
                                     delay, args.show_every)
        all_counts[name] = counts
        all_totals[name] = total

        if args.scenario == 'all' and name != order[-1]:
            input(f"\n  {COLORS['DIM']}  Press Enter to continue to next scenario...{COLORS['RESET']}")

    if args.scenario == 'all':
        print_final_summary(all_counts, all_totals)

if __name__ == '__main__':
    main()
