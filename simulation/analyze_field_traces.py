"""
Truth-free field-trace evaluation for CWTR  (Campaign A, high-multipath extension)
==================================================================================
Processes the Sensor Logger recordings in  ../data/GPS_Data/extracted/<batch>/<rec>/Location.csv
through the IDENTICAL CWTR pipeline and baselines used in analyze_real_traces.py.

These field traces have no surveyed ground-truth polyline, so only
ground-truth-free metrics are reported:
  - trajectory jitter (mean absolute turn angle, deg)   [same def. as the paper]
  - kinematic-violation rate (% of steps with implied speed > VMAX = 2.5 m/s)
  - path-length inflation (estimate path length / CWTR path length)
plus trace-level reception characterization (median / P90 reported horizontal
accuracy, % fixes > 30 m, raw burst fraction) and CWTR's own gating rate.

Traces are stratified by median reported horizontal accuracy:
  benign   <= 10 m       moderate  10-30 m        severe   > 30 m

EXCLUSIONS (stated in the paper): recordings that are not single walked
street-block segments (one long mixed-mode segment and one long pause-dominated
segment); pre-recording cached fixes (seconds_elapsed < 0) are dropped.

USAGE:  cd code && python3 analyze_field_traces.py
Outputs: field_trace_results.csv, field_trace_characterization.csv,
         ../figures/fieldtrace_jitter.{pdf,png},
         ../figures/fieldtrace_acc_cdf.{pdf,png},
         ../figures/fieldtrace_example.{pdf,png}
"""
import os, csv, glob, math
import numpy as np

from analyze_real_traces import (A0, VMAX, SIGMA0, TAU, GATE, Q_ACC,
                                 kalman_rts_vardt, kalman_rts_chi2_vardt,
                                 confidence, decim, jitter,
                                 to_xy, wilcoxon_signedrank)

ROOT = os.path.join('..', 'data', 'GPS_Data', 'extracted')
EXCLUDE = (
    'b01_haram_r01',   # 25-min pause-dominated, not a single walked block
    'b02_haram_r08', # 1.9-km mixed-mode segment
)
METHODS = ['raw', 'dec', 'accdec', 'kalman', 'chikf', 'cwtr']
LAB = {'raw': 'Raw fixes', 'dec': 'Decimation (5 m)', 'accdec': 'Accuracy+decim.',
       'kalman': 'Kalman/RTS (fixed R)', 'chikf': 'Chi2-gated KF (adapt. R)',
       'cwtr': 'CWTR (proposed)'}


def load_location_csv(path):
    rows = list(csv.DictReader(open(path, newline='')))
    t = np.array([float(r['seconds_elapsed']) for r in rows])
    lat = np.array([float(r['latitude']) for r in rows])
    lon = np.array([float(r['longitude']) for r in rows])
    acc = np.array([float(r['horizontalAccuracy']) for r in rows])
    keep = t >= 0                      # drop pre-recording cached fixes
    t, lat, lon, acc = t[keep], lat[keep], lon[keep], acc[keep]
    return t, lat, lon, np.clip(acc, 1.0, None)


def implied_speeds(est, dts):
    d = np.linalg.norm(np.diff(est, axis=0), axis=1)
    dt = np.clip(dts[1:], 1e-3, None)
    return d / dt


def path_len(est):
    return float(np.sum(np.linalg.norm(np.diff(est, axis=0), axis=1)))


def stratum(med_acc):
    if med_acc <= 10.0:  return 'benign'
    if med_acc <= 30.0:  return 'moderate'
    return 'severe'


def process(path):
    t, lat, lon, acc = load_location_csv(path)
    if len(t) < 10:
        raise ValueError('too few fixes')
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    z = to_xy(lat, lon, lat0, lon0)
    dts = np.diff(t, prepend=t[0]); dts[dts <= 0] = np.median(dts[dts > 0]) if np.any(dts > 0) else 1.0

    ests = {'raw': z, 'dec': decim(z, 5.0)}
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

    per = {}
    for m, e in ests.items():
        s = implied_speeds(e, dts)
        per[m] = {'jitter': jitter(e),
                  'viol': 100.0 * float(np.mean(s > VMAX)) if len(s) else 0.0,
                  'plen': path_len(e)}
    raw_s = implied_speeds(z, dts)
    char = {'n': len(z),
            'dur': float(t[-1] - t[0]),
            'med_acc': float(np.median(acc)),
            'p90_acc': float(np.percentile(acc, 90)),
            'pct_acc30': 100.0 * float(np.mean(acc > 30.0)),
            'burst': 100.0 * float(np.mean(raw_s > VMAX)) if len(raw_s) else 0.0,
            'max_acc': float(acc.max()),
            'gated': 100.0 * float(np.mean(c < TAU))}
    return per, char, (z, ests['cwtr'], ests['kalman'], acc, c)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, '*', '*', 'Location.csv')))
    files = [f for f in files if os.path.basename(os.path.dirname(f)) not in EXCLUDE]
    recs = []
    for fp in files:
        batch = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        name = os.path.basename(os.path.dirname(fp))
        site = 'UPM campus' if 'upm' in batch else 'Haram district'
        try:
            per, char, raw = process(fp)
        except Exception as e:
            print(f"  SKIP {name}: {e}"); continue
        recs.append({'name': name, 'site': site, 'stratum': stratum(char['med_acc']),
                     'per': per, 'char': char, 'raw': raw})
    print(f"analysed {len(recs)} traces "
          f"({sum(r['site'] == 'Haram district' for r in recs)} Haram, "
          f"{sum(r['site'] == 'UPM campus' for r in recs)} UPM); "
          f"{len(EXCLUDE)} recordings excluded")

    strata = ['benign', 'moderate', 'severe']
    # ---- characterization table ----
    with open('field_trace_characterization.csv', 'w') as f:
        f.write('stratum,n_traces,fixes_total,fixes_mean,dur_mean_s,med_acc_mean,'
                'p90_acc_mean,pct_acc30_mean,burst_mean,gated_mean\n')
        for s in strata + ['ALL']:
            rs = [r for r in recs if s == 'ALL' or r['stratum'] == s]
            if not rs: continue
            ch = [r['char'] for r in rs]
            f.write(f"{s},{len(rs)},{sum(c['n'] for c in ch)},"
                    f"{np.mean([c['n'] for c in ch]):.1f},{np.mean([c['dur'] for c in ch]):.1f},"
                    f"{np.mean([c['med_acc'] for c in ch]):.2f},{np.mean([c['p90_acc'] for c in ch]):.2f},"
                    f"{np.mean([c['pct_acc30'] for c in ch]):.2f},{np.mean([c['burst'] for c in ch]):.2f},"
                    f"{np.mean([c['gated'] for c in ch]):.2f}\n")
    print('wrote field_trace_characterization.csv')

    # ---- per-method results, overall + per stratum ----
    def agg(rs, m, key):
        return np.array([r['per'][m][key] for r in rs])
    with open('field_trace_results.csv', 'w') as f:
        f.write('stratum,method,jitter_mean,jitter_sd,viol_mean,viol_sd,pleninfl_mean,N\n')
        for s in ['ALL'] + strata:
            rs = [r for r in recs if s == 'ALL' or r['stratum'] == s]
            if not rs: continue
            for m in METHODS:
                jt = agg(rs, m, 'jitter'); vi = agg(rs, m, 'viol')
                infl = np.array([r['per'][m]['plen'] / r['per']['cwtr']['plen'] for r in rs])
                f.write(f"{s},{m},{jt.mean():.3f},{jt.std():.3f},{vi.mean():.3f},{vi.std():.3f},"
                        f"{infl.mean():.3f},{len(rs)}\n")
    print('wrote field_trace_results.csv')

    # console summary + Wilcoxon
    for s in ['ALL'] + strata:
        rs = [r for r in recs if s == 'ALL' or r['stratum'] == s]
        if not rs: continue
        print(f"\n--- {s} (N={len(rs)}) ---")
        print(f"{'Method':<22}{'Jitter(deg)':<16}{'Viol(%)':<14}{'PathInfl':<10}")
        for m in METHODS:
            jt = agg(rs, m, 'jitter'); vi = agg(rs, m, 'viol')
            infl = np.array([r['per'][m]['plen'] / r['per']['cwtr']['plen'] for r in rs])
            print(f"{LAB[m]:<22}{jt.mean():6.2f}+/-{jt.std():<7.2f}{vi.mean():6.2f}+/-{vi.std():<6.2f}{infl.mean():<10.2f}")
        W, p = wilcoxon_signedrank(agg(rs, 'cwtr', 'jitter'), agg(rs, 'kalman', 'jitter'))
        Wc, pcj = wilcoxon_signedrank(agg(rs, 'cwtr', 'jitter'), agg(rs, 'chikf', 'jitter'))
        Wv, pv = wilcoxon_signedrank(agg(rs, 'cwtr', 'viol'), agg(rs, 'raw', 'viol'))
        print(f"CWTR vs Kalman jitter Wilcoxon p={p:.2e};  vs Chi2-KF p={pcj:.2e};  CWTR vs raw viol p={pv:.2e}")

    # ---- figures ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight'})
    FIGD = os.path.join('..', 'figures')
    cols = ['0.72', '0.72', '0.82', '#9DC3E6', '#E69B96']
    hatches = ['//', '\\\\', 'xx', '..', None]

    # (a) CDF of per-fix reported accuracy by stratum
    plt.figure(figsize=(5, 3.4))
    sc = {'benign': '#4C7DB2', 'moderate': '#C99A3C', 'severe': '#B5524C'}
    for s in strata:
        acc_arrays = [r['raw'][3] for r in recs if r['stratum'] == s]
        if not acc_arrays: continue
        accs = np.concatenate(acc_arrays)
        xs = np.sort(accs); ys = np.arange(1, len(xs) + 1) / len(xs)
        n = sum(r['stratum'] == s for r in recs)
        plt.semilogx(xs, ys, color=sc[s], lw=1.8, label=f"{s} (N={n})")
    plt.axvline(30, color='0.5', ls=':', lw=1)
    plt.text(31, 0.05, '30 m', fontsize=8, color='0.4')
    plt.xlabel('Reported horizontal accuracy (m)'); plt.ylabel('CDF of fixes')
    plt.legend(fontsize=8); plt.grid(alpha=0.3, which='both'); plt.tight_layout()
    plt.savefig(os.path.join(FIGD, 'fieldtrace_acc_cdf.pdf'))
    plt.savefig(os.path.join(FIGD, 'fieldtrace_acc_cdf.png'), dpi=300)

    # (b) jitter by method, grouped by stratum
    plt.figure(figsize=(5.4, 3.6))
    w = 0.25
    xb = np.arange(len(METHODS))
    for i, s in enumerate(strata):
        rs = [r for r in recs if r['stratum'] == s]
        if not rs: continue
        means = [agg(rs, m, 'jitter').mean() for m in METHODS]
        sds = [agg(rs, m, 'jitter').std() for m in METHODS]
        plt.bar(xb + (i - 1) * w, means, w, yerr=sds, capsize=2,
                color=list(sc.values())[i], edgecolor='0.30', linewidth=0.7,
                label=f"{s} (N={len(rs)})", alpha=0.9)
    plt.xticks(xb, [LAB[m].split(' (')[0] for m in METHODS], rotation=20, ha='right', fontsize=8)
    plt.ylabel('Trajectory jitter (deg)'); plt.legend(fontsize=8)
    plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGD, 'fieldtrace_jitter.pdf'))
    plt.savefig(os.path.join(FIGD, 'fieldtrace_jitter.png'), dpi=300)

    # (c) example burst-multipath trace: raw vs Kalman vs CWTR (partial gating)
    cand = [r for r in recs if 1.0 <= r['char']['gated'] <= 30.0 and r['char']['n'] >= 80]
    if cand:
        ex = max(cand, key=lambda r: r['char']['max_acc'] / max(r['char']['med_acc'], 1.0))
        z, cw, ka, acc, c = ex['raw']
        g = c < TAU
        plt.figure(figsize=(5, 4.2))
        plt.plot(z[:, 0], z[:, 1], '.', color='0.65', ms=3.5, label='Raw fixes')
        plt.plot(z[g, 0], z[g, 1], 'x', color='#B5524C', ms=6, mew=1.3,
                 label=f'Gated by CWTR ({100*g.mean():.0f} %)')
        plt.plot(ka[:, 0], ka[:, 1], '-', color='#9DC3E6', lw=1.6, label='Kalman/RTS (fixed R)')
        plt.plot(cw[:, 0], cw[:, 1], '-', color='#E69B96', lw=2.0, label='CWTR (proposed)')
        # frame on the walked route (inliers), so distant outliers do not flatten the view
        xi, yi = z[~g, 0], z[~g, 1]
        if len(xi) > 5:
            mx, my = 0.25 * (xi.max() - xi.min() + 10), 0.25 * (yi.max() - yi.min() + 10)
            plt.xlim(xi.min() - mx, xi.max() + mx); plt.ylim(yi.min() - my, yi.max() + my)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('East (m)'); plt.ylabel('North (m)')
        plt.legend(fontsize=8, loc='best'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIGD, 'fieldtrace_example.pdf'))
        plt.savefig(os.path.join(FIGD, 'fieldtrace_example.png'), dpi=300)
        print(f"\nexample trace: {ex['name']} (stratum {ex['stratum']}, "
              f"median acc {ex['char']['med_acc']:.1f} m, P90 {ex['char']['p90_acc']:.0f} m, "
              f"max {acc.max():.0f} m, gated {ex['char']['gated']:.1f}%)")
    print('wrote figures to ../figures/fieldtrace_*.{pdf,png}')


if __name__ == '__main__':
    main()
