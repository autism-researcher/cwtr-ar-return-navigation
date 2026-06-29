"""
Real-GPS trace evaluation for CWTR  (Campaign A)
================================================
Feeds RECORDED GPS fixes into the same CWTR pipeline and baselines used in the
simulation study, and reports RMSE-to-ground-truth-path, origin-return error,
and jitter on real data. No values are invented: every number is computed from
the CSV files you place in the traces directory.

USAGE
-----
1) Put each recorded trace as a CSV in   ./real_traces/<name>.csv
   Required columns (header names auto-detected, case-insensitive):
       time | timestamp | utc      (any parseable time, or seconds)
       lat  | latitude
       lon  | lng | longitude
       acc  | accuracy | hAccuracy  (horizontal accuracy in METERS)
   Optional: hdop, satellites.
2) Provide ground truth for each trace, either:
     (a) a polyline file  ./real_traces/<name>.truth.csv  with columns lat,lon
         (trace the true walked line in Google Earth and export the vertices), OR
     (b) two markers      ./real_traces/<name>.markers.csv with rows:
              role,lat,lon
              start,...
              dest,...
   If a polyline exists it is used for RMSE; markers (if present) give origin
   and destination end-point error.
3) Run:   python3 analyze_real_traces.py
   Test the pipeline first with a synthetic trace:
          python3 analyze_real_traces.py --make-demo   (writes one demo trace)
          python3 analyze_real_traces.py               (analyses it)
   Demo output is labelled SYNTHETIC and must NOT be used in the paper.

METHOD (identical constants to cwtr_simulation.py)
"""
import os, sys, glob, math
import numpy as np

A0, VMAX, SIGMA0, TAU, GATE, Q_ACC = 8.0, 2.5, 4.0, 0.30, 1e4, 0.05
TRACE_DIR = "real_traces"

# ---------------- IO + projection ----------------
def _find(cols, *cands):
    import re
    low = {re.sub(r'[ _]+', '', c.lower().strip()): c for c in cols}
    for cand in cands:
        c2 = re.sub(r'[ _]+', '', cand)
        for k, orig in low.items():
            if c2 in k:
                return orig
    return None

def load_gps_csv(path):
    import csv
    rows = list(csv.DictReader(open(path, newline='')))
    if not rows:
        raise ValueError(f"{path}: empty")
    cols = rows[0].keys()
    clat = _find(cols, 'latitude', 'lat')
    clon = _find(cols, 'longitude', 'lng', 'lon')
    cacc = _find(cols, 'horizontalaccuracy', 'haccuracy', 'accuracy', 'acc', 'hdop')
    ctime = _find(cols, 'secondselapsed', 'elapsed', 'timestamp', 'utc', 'datetime', 'date', 'time')
    if clat is None or clon is None:
        raise ValueError(f"{path}: could not find lat/lon columns in {list(cols)}")
    lat = np.array([float(r[clat]) for r in rows])
    lon = np.array([float(r[clon]) for r in rows])
    acc = (np.array([float(r[cacc]) for r in rows]) if cacc else np.full(len(rows), 5.0))
    # time -> seconds (handles seconds_elapsed, epoch s/ms/ns, or ISO datetime)
    t = np.arange(len(rows), dtype=float)
    if ctime:
        raw = [str(r[ctime]).strip() for r in rows]
        def _isnum(x):
            try: float(x); return True
            except Exception: return False
        if raw and all(_isnum(x) for x in raw if x):
            t = np.array([float(x) for x in raw], float); t = t - t[0]
            d = np.diff(t); d = d[d != 0]
            md = float(np.median(np.abs(d))) if len(d) else 1.0
            if md > 1e6:    t = t / 1e9      # nanoseconds  -> s (e.g. Sensor Logger 'time')
            elif md > 50:   t = t / 1e3      # milliseconds -> s
        else:
            try:
                from dateutil import parser as dp
                ts = [dp.parse(x).timestamp() for x in raw]
                t = np.array(ts) - ts[0]
            except Exception:
                t = np.arange(len(rows), dtype=float)
    return t, lat, lon, np.clip(acc, 1.0, None)

def load_polyline(path):
    import csv
    rows = list(csv.DictReader(open(path, newline='')))
    cols = rows[0].keys()
    clat = _find(cols, 'latitude', 'lat'); clon = _find(cols, 'longitude', 'lng', 'lon')
    return (np.array([float(r[clat]) for r in rows]),
            np.array([float(r[clon]) for r in rows]))

def load_markers(path):
    import csv
    out = {}
    for r in csv.DictReader(open(path, newline='')):
        role = r.get('role') or r.get('Role')
        cols = r.keys(); clat = _find(cols, 'latitude', 'lat'); clon = _find(cols, 'longitude', 'lng', 'lon')
        out[role.strip().lower()] = (float(r[clat]), float(r[clon]))
    return out

def to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat - lat0) * 110540.0
    return np.column_stack([x, y])

# ---------------- geometry ----------------
def pt_seg_dist(p, a, b):
    ab = b - a; t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = max(0.0, min(1.0, t)); proj = a + t * ab
    return np.linalg.norm(p - proj)

def pt_polyline_dist(p, poly):
    return min(pt_seg_dist(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1))

# ---------------- CWTR pipeline (variable dt) ----------------
def kalman_rts_vardt(z, Rseq, dts):
    n = len(z); H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
    xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4)); xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    Fs = []
    x = np.array([z[0, 0], 0, z[0, 1], 0.0]); P = np.eye(4) * 100.0
    for t in range(n):
        dt = dts[t] if t > 0 else 1.0
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], float); Fs.append(F)
        G = np.array([[dt * dt / 2, 0], [dt, 0], [0, dt * dt / 2], [0, dt]], float); Q = G @ G.T * Q_ACC
        if t > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xp[t] = x; Pp[t] = P
        S = H @ P @ H.T + Rseq[t]; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z[t] - H @ x); P = (np.eye(4) - K @ H) @ P
        xs[t] = x; Ps[t] = P
    xsm = xs.copy()
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ Fs[t + 1].T @ np.linalg.inv(Pp[t + 1])
        xsm[t] = xs[t] + C @ (xsm[t + 1] - xp[t + 1])
    return xsm[:, [0, 2]]

CHI2_99 = 9.21   # chi-square 99% quantile, 2 dof

def kalman_rts_chi2_vardt(z, acc, dts, chi2=CHI2_99, inflate=GATE):
    """Adaptive robust-KF baseline: per-fix R from the reported accuracy plus
    chi-square innovation gating (99%, 2 dof) with RTS smoothing. Standard
    robust-filtering practice; differs from CWTR in using only the innovation
    statistic (no kinematic/temporal confidence, no continuous weighting)."""
    n = len(z); H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
    xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4)); xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    Fs = []
    x = np.array([z[0, 0], 0, z[0, 1], 0.0]); P = np.eye(4) * 100.0
    for t in range(n):
        dt = dts[t] if t > 0 else 1.0
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], float); Fs.append(F)
        G = np.array([[dt * dt / 2, 0], [dt, 0], [0, dt * dt / 2], [0, dt]], float); Q = G @ G.T * Q_ACC
        if t > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xp[t] = x; Pp[t] = P
        R = np.eye(2) * float(acc[t]) ** 2
        S = H @ P @ H.T + R; v = z[t] - H @ x
        if t > 0 and float(v @ np.linalg.solve(S, v)) > chi2:
            R = R * inflate; S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ v; P = (np.eye(4) - K @ H) @ P
        xs[t] = x; Ps[t] = P
    xsm = xs.copy()
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ Fs[t + 1].T @ np.linalg.inv(Pp[t + 1])
        xsm[t] = xs[t] + C @ (xsm[t + 1] - xp[t + 1])
    return xsm[:, [0, 2]]

def confidence(z, acc, dts):
    n = len(z); c = np.ones(n)
    nominal = np.median(dts[1:]) if n > 1 else 1.0
    for t in range(n):
        cacc = 1 / (1 + (acc[t] / A0) ** 2)
        if t > 0:
            dt = max(dts[t], 1e-3); s = np.linalg.norm(z[t] - z[t - 1]) / dt
            ckin = 1 / (1 + (max(0, s - VMAX)) ** 2)
            ctmp = 1 / (1 + (max(0, dt / nominal - 1.5)) ** 2)   # penalise long gaps
        else:
            ckin = ctmp = 1.0
        c[t] = np.clip(cacc * ckin * ctmp, 1e-3, 1)
    return c

def decim(z, mind):
    keep = [0]; last = z[0]
    for t in range(1, len(z)):
        if np.linalg.norm(z[t] - last) >= mind: keep.append(t); last = z[t]
    keep = np.array(keep); est = np.zeros_like(z)
    est[:, 0] = np.interp(np.arange(len(z)), keep, z[keep, 0])
    est[:, 1] = np.interp(np.arange(len(z)), keep, z[keep, 1])
    return est

def jitter(est):
    d = np.diff(est, axis=0); ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs(np.diff(ang)); turn = np.minimum(turn, 2 * np.pi - turn)
    return float(np.degrees(np.mean(turn))) if len(turn) else 0.0

def rmse_to_path(est, poly):
    return float(np.sqrt(np.mean([pt_polyline_dist(p, poly) ** 2 for p in est])))

def wilcoxon_signedrank(x, y):
    from math import erf, sqrt
    d = np.asarray(x, float) - np.asarray(y, float); d = d[d != 0]; n = len(d)
    if n == 0: return 0.0, 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    W = float(np.sum(ranks[d > 0])); mu = n * (n + 1) / 4.0
    sd = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sd if sd > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return W, p

# ---------------- per-trace processing ----------------
def process(csv_path):
    t, lat, lon, acc = load_gps_csv(csv_path)
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    z = to_xy(lat, lon, lat0, lon0)
    dts = np.diff(t, prepend=t[0]); dts[dts <= 0] = np.median(dts[dts > 0]) if np.any(dts > 0) else 1.0
    base = csv_path[:-4]
    poly = None
    for suf in ('.truth.csv', '_truth.csv'):
        if os.path.exists(base + suf):
            plat, plon = load_polyline(base + suf); poly = to_xy(plat, plon, lat0, lon0); break
    markers = None
    for suf in ('.markers.csv', '_markers.csv'):
        if os.path.exists(base + suf):
            m = load_markers(base + suf)
            markers = {k: to_xy(np.array([v[0]]), np.array([v[1]]), lat0, lon0)[0] for k, v in m.items()}
            break
    if poly is None and markers and 'start' in markers and 'dest' in markers:
        poly = np.array([markers['start'], markers['dest']])
    if poly is None:
        raise ValueError(f"{csv_path}: no ground truth (need .truth.csv polyline or .markers.csv)")

    # methods
    ests = {}
    ests['raw'] = z
    ests['dec'] = decim(z, 5.0)
    mask = acc <= 30
    zacc = z.copy()
    if mask.sum() >= 2:
        idx = np.where(mask)[0]
        zacc[:, 0] = np.interp(np.arange(len(z)), idx, z[idx, 0])
        zacc[:, 1] = np.interp(np.arange(len(z)), idx, z[idx, 1])
    ests['accdec'] = decim(zacc, 5.0)
    Rk = np.eye(2) * float(np.mean(acc ** 2))
    ests['kalman'] = kalman_rts_vardt(z, [Rk] * len(z), dts)
    ests['chikf'] = kalman_rts_chi2_vardt(z, acc, dts)
    c = confidence(z, acc, dts)
    Rc = [np.eye(2) * ((SIGMA0 ** 2 / c[i]) * (GATE if c[i] < TAU else 1.0)) for i in range(len(z))]
    ests['cwtr'] = kalman_rts_vardt(z, Rc, dts)

    res = {}
    for k, e in ests.items():
        row = {'rmse': rmse_to_path(e, poly), 'jitter': jitter(e)}
        if markers and 'start' in markers:
            row['origin'] = float(np.linalg.norm(e[0] - markers['start']))
        if markers and 'dest' in markers:
            row['dest'] = float(np.linalg.norm(e[-1] - markers['dest']))
        res[k] = row
    return res, len(z)

# ---------------- demo ----------------
def make_demo():
    os.makedirs(TRACE_DIR, exist_ok=True)
    rng = np.random.default_rng(0); T = 250
    lat0, lon0 = 24.4710, 39.6110         # Madinah-ish base
    h = 0.3; p = np.zeros(2); truth = []
    for _ in range(T):
        h += rng.normal(0, 0.12); p = p + 1.3 * np.array([np.cos(h), np.sin(h)]); truth.append(p.copy())
    truth = np.array(truth)
    sigma = 4.0 + 3.0 * np.abs(np.sin(np.linspace(0, 6, T)))
    z = truth + rng.normal(0, 1, (T, 2)) * sigma[:, None]
    for i in range(T):
        if rng.random() < 0.10:
            z[i] = truth[i] + rng.uniform(15, 55) * np.array([np.cos(rng.uniform(0, 6.28)), np.sin(rng.uniform(0, 6.28))])
    def xy2ll(xy):
        lat = lat0 + xy[:, 1] / 110540.0
        lon = lon0 + xy[:, 0] / (math.cos(math.radians(lat0)) * 111320.0)
        return lat, lon
    zlat, zlon = xy2ll(z); tlat, tlon = xy2ll(truth[::10])
    with open(f"{TRACE_DIR}/demo_SYNTHETIC.csv", "w") as f:
        f.write("timestamp,latitude,longitude,accuracy\n")
        for i in range(T): f.write(f"{i},{zlat[i]:.7f},{zlon[i]:.7f},{sigma[i]:.1f}\n")
    with open(f"{TRACE_DIR}/demo_SYNTHETIC.truth.csv", "w") as f:
        f.write("latitude,longitude\n")
        for a, b in zip(tlat, tlon): f.write(f"{a:.7f},{b:.7f}\n")
    with open(f"{TRACE_DIR}/demo_SYNTHETIC.markers.csv", "w") as f:
        f.write("role,latitude,longitude\n")
        s = xy2ll(truth[[0]]); d = xy2ll(truth[[-1]])
        f.write(f"start,{s[0][0]:.7f},{s[1][0]:.7f}\ndest,{d[0][0]:.7f},{d[1][0]:.7f}\n")
    print(f"wrote synthetic demo trace into {TRACE_DIR}/  (label: SYNTHETIC - not for paper)")

# ---------------- main ----------------
def main():
    if '--make-demo' in sys.argv:
        make_demo(); return
    _skip = ('demo', 'sl_test', 'synthetic')
    files = sorted(f for f in glob.glob(f"{TRACE_DIR}/*.csv")
                   if not f.endswith(('.truth.csv', '_truth.csv', '.markers.csv', '_markers.csv'))
                   and not any(s in os.path.basename(f).lower() for s in _skip))
    if not files:
        print(f"No traces found in ./{TRACE_DIR}/. Add <name>.csv (+ <name>.truth.csv) or run --make-demo.")
        return
    methods = ['raw', 'dec', 'accdec', 'kalman', 'chikf', 'cwtr']
    lab = {'raw': 'Raw fixes', 'dec': 'Decimation (5 m)', 'accdec': 'Accuracy+decim.',
           'kalman': 'Kalman/RTS (fixed R)', 'chikf': 'Chi2-gated KF (adapt. R)',
           'cwtr': 'CWTR (proposed)'}
    agg = {m: {'rmse': [], 'jitter': [], 'origin': [], 'dest': []} for m in methods}
    print(f"{'Trace':<26}{'N':>5}  per-trace RMSE (m): raw / dec / accdec / kalman / chikf / cwtr")
    for fp in files:
        try:
            res, n = process(fp)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fp)}: {e}"); continue
        for m in methods:
            for key in ('rmse', 'jitter', 'origin', 'dest'):
                if key in res[m]: agg[m][key].append(res[m][key])
        r = res
        print(f"{os.path.basename(fp):<26}{n:>5}  "
              f"{r['raw']['rmse']:.2f} / {r['dec']['rmse']:.2f} / {r['accdec']['rmse']:.2f} / "
              f"{r['kalman']['rmse']:.2f} / {r['chikf']['rmse']:.2f} / {r['cwtr']['rmse']:.2f}")
    N = len(agg['cwtr']['rmse'])
    if N == 0:
        print("No analysable traces."); return
    print(f"\n=== AGGREGATE over N={N} traces (mean +/- SD) ===")
    print(f"{'Method':<22}{'RMSE (m)':<18}{'Origin (m)':<16}{'Jitter (deg)':<12}")
    for m in methods:
        rm = np.array(agg[m]['rmse']); jt = np.array(agg[m]['jitter']); og = np.array(agg[m]['origin'])
        ostr = f"{og.mean():.2f}+/-{og.std():.2f}" if len(og) else "   -   "
        print(f"{lab[m]:<22}{rm.mean():.2f} +/- {rm.std():<8.2f}{ostr:<16}{jt.mean():<12.1f}")
    cw = np.array(agg['cwtr']['rmse']); ka = np.array(agg['kalman']['rmse']); ra = np.array(agg['raw']['rmse'])
    ck = np.array(agg['chikf']['rmse'])
    W, p = wilcoxon_signedrank(cw, ka)
    Wc, pc = wilcoxon_signedrank(cw, ck)
    print(f"\nCWTR vs Kalman RMSE: {100*(1-cw.mean()/ka.mean()):+.1f}%   "
          f"vs raw: {100*(1-cw.mean()/ra.mean()):+.1f}%   Wilcoxon p={p:.2e} (N={N})")
    print(f"CWTR vs Chi2-KF RMSE: {100*(1-cw.mean()/ck.mean()):+.1f}%   Wilcoxon p={pc:.2e} (N={N})")
    # write results csv
    with open("real_trace_results.csv", "w") as f:
        f.write("method,rmse_mean,rmse_sd,origin_mean,jitter_mean,N\n")
        for m in methods:
            rm = np.array(agg[m]['rmse']); og = np.array(agg[m]['origin']); jt = np.array(agg[m]['jitter'])
            f.write(f"{m},{rm.mean():.3f},{rm.std():.3f},"
                    f"{(og.mean() if len(og) else float('nan')):.3f},{jt.mean():.3f},{N}\n")
    print("wrote real_trace_results.csv")
    # optional figure
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight'})
        means = [np.array(agg[m]['rmse']).mean() for m in methods]
        sds = [np.array(agg[m]['rmse']).std() for m in methods]
        cols = ['0.72', '0.72', '0.82', '#9DC3E6', '#B7D7A8', '#E69B96']  # light palette (matches make_figures.py)
        hatches = ['//', '\\\\', 'xx', '..', 'oo', None]
        plt.figure(figsize=(5.4, 3.6))
        rbars = plt.bar(range(len(methods)), means, yerr=sds, capsize=4, color=cols, edgecolor='0.30', linewidth=0.8)
        for b, h in zip(rbars, hatches):
            if h: b.set_hatch(h)
        plt.xticks(range(len(methods)), [lab[m].split(' (')[0].replace('Chi2', r'$\chi^2$') for m in methods],
                   rotation=20, ha='right', fontsize=8)
        plt.ylabel('RMSE to ground-truth path (m)'); plt.title(f'Real GPS traces (N={N})')
        plt.tight_layout()
        plt.savefig('real_trace_rmse.pdf')           # vector, primary
        plt.savefig('real_trace_rmse.png', dpi=300)   # high-res raster fallback
        print("wrote real_trace_rmse.pdf + .png")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
