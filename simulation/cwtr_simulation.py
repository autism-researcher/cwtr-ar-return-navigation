"""
CWTR Monte-Carlo simulation  (reproducible, numpy-only)
=======================================================
Reconstruction-accuracy study for Confidence-Weighted Trajectory Reconstruction
(CWTR): a constant-velocity Kalman/RTS smoother whose per-fix measurement noise
is driven by a multi-factor confidence score, with hard outlier gating.

The script generates synthetic pedestrian ground truth, heteroscedastic GNSS
observations with HDOP-driven multipath outlier bursts, evaluates four baselines
plus CWTR, and runs a Wilcoxon signed-rank test. Everything is seeded.

It reports:
  (1) a main results table at a documented operating point (dense-crowd,
      heavy-multipath regime representative of the target environment), and
  (2) a multipath-severity SENSITIVITY SWEEP that shows where CWTR helps and
      where its advantage disappears (the high-noise crossover).

Reproduce:
    python3 cwtr_simulation.py
All printed values are the direct output of this script (no hand-editing).
"""
import numpy as np
from math import erf, sqrt

# ----------------------------------------------------------------------
# Fixed model parameters (documented; change here to explore sensitivity)
# ----------------------------------------------------------------------
SEED      = 42
T         = 200      # samples
DT        = 1.0      # s  (1 Hz)
SPEED     = 1.3      # m/s walking
SIGMA_BASE= 4.0      # base GNSS noise std (m) at nominal HDOP
A0        = 8.0      # confidence: accuracy scale (m)
VMAX      = 2.5      # confidence: pedestrian speed ceiling (m/s)
SIGMA0    = 4.0      # CWTR nominal measurement std (m)
TAU       = 0.30     # CWTR confidence gate threshold
GATE      = 1e4      # CWTR variance inflation when c_t < TAU
Q_ACC     = 0.05     # Kalman process (acceleration) noise
M         = 120      # Monte-Carlo trials

# Operating point: per-step base outlier probability (modulated by HDOP).
# PRIMARY reflects a heavy-multipath, dense-crowd / urban-canyon environment
# (the Makkah/Madinah target). The sweep below covers light -> severe.
PRIMARY_RATE = 0.12


def wilcoxon_signedrank(x, y):
    """Wilcoxon signed-rank test, normal approximation (paired)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    W = float(np.sum(ranks[d > 0]))
    mu = n * (n + 1) / 4.0
    sd = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sd
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return W, p


def gen_truth(rng):
    h = rng.uniform(0, 2 * np.pi); p = np.zeros(2); pos = np.zeros((T, 2))
    for t in range(T):
        h += rng.normal(0, 0.15)                         # smooth correlated heading
        p = p + SPEED * DT * np.array([np.cos(h), np.sin(h)])
        pos[t] = p
    return pos


def gen_obs(truth, rate, rng):
    hdop = np.zeros(T); h = 1.5
    for t in range(T):
        h = 0.9 * h + 0.1 * 1.5 + rng.normal(0, 0.3)     # AR(1) HDOP process
        hdop[t] = max(0.5, h)
    sigma = SIGMA_BASE * (hdop / 1.5)                     # heteroscedastic noise
    z = truth + rng.normal(0, 1, (T, 2)) * sigma[:, None]
    pi = np.clip(rate * (hdop - 0.5), 0, 0.6)            # outlier prob rises with HDOP
    t = 0
    while t < T:
        if rng.random() < pi[t]:
            burst = int(rng.integers(1, 4))              # 1-3 corrupted fixes
            for k in range(burst):
                if t + k < T:
                    mag = rng.uniform(15, 55); ang = rng.uniform(0, 2 * np.pi)
                    z[t + k] = truth[t + k] + mag * np.array([np.cos(ang), np.sin(ang)])
            t += burst
        else:
            t += 1
    a = np.clip(np.abs(sigma * (1 + rng.normal(0, 0.2, T))), 1.0, None)  # reported accuracy
    return z, a, sigma


def kalman_rts(z, Rseq):
    F = np.array([[1, DT, 0, 0], [0, 1, 0, 0], [0, 0, 1, DT], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    G = np.array([[DT * DT / 2, 0], [DT, 0], [0, DT * DT / 2], [0, DT]])
    Q = G @ G.T * Q_ACC; n = len(z)
    xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4)); xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    x = np.array([z[0, 0], 0, z[0, 1], 0.0]); P = np.eye(4) * 100.0
    for t in range(n):
        if t > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xp[t] = x; Pp[t] = P
        S = H @ P @ H.T + Rseq[t]; K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z[t] - H @ x); P = (np.eye(4) - K @ H) @ P
        xs[t] = x; Ps[t] = P
    xsm = xs.copy()
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xsm[t] = xs[t] + C @ (xsm[t + 1] - xp[t + 1])
    return xsm[:, [0, 2]]


CHI2_99 = 9.21   # chi-square 99% quantile, 2 dof

def kalman_rts_chi2(z, a, chi2=CHI2_99, inflate=GATE):
    """Adaptive robust-KF baseline: per-fix R from reported accuracy plus
    chi-square innovation gating (99%, 2 dof), RTS-smoothed."""
    F = np.array([[1, DT, 0, 0], [0, 1, 0, 0], [0, 0, 1, DT], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    G = np.array([[DT * DT / 2, 0], [DT, 0], [0, DT * DT / 2], [0, DT]])
    Q = G @ G.T * Q_ACC; n = len(z)
    xs = np.zeros((n, 4)); Ps = np.zeros((n, 4, 4)); xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    x = np.array([z[0, 0], 0, z[0, 1], 0.0]); P = np.eye(4) * 100.0
    for t in range(n):
        if t > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xp[t] = x; Pp[t] = P
        R = np.eye(2) * float(a[t]) ** 2
        S = H @ P @ H.T + R; v = z[t] - H @ x
        if t > 0 and float(v @ np.linalg.solve(S, v)) > chi2:
            R = R * inflate; S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ v; P = (np.eye(4) - K @ H) @ P
        xs[t] = x; Ps[t] = P
    xsm = xs.copy()
    for t in range(n - 2, -1, -1):
        C = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xsm[t] = xs[t] + C @ (xsm[t + 1] - xp[t + 1])
    return xsm[:, [0, 2]]


def conf(z, a, dts=None, use_acc=True, use_kin=True, use_tmp=True):
    """Canonical multi-factor confidence, Eqs. (1)-(3) of the paper.
    Single source of truth for ALL scripts (simulation, real traces, field
    campaign, figures, component evaluations).
    dts=None  -> uniform 1 Hz sampling: c_tmp is inactive (equals 1), exactly
                 the simulation setting described in Sec. IV-A.
    dts given -> variable-rate recorded data: c_tmp activates as in Eq. (1).
    use_*     -> ablation toggles (Sec. V-C) without re-implementing the score."""
    n = len(z); c = np.ones(n)
    nominal = np.median(dts[1:]) if (dts is not None and n > 1) else DT
    for t in range(n):
        cacc = 1 / (1 + (a[t] / A0) ** 2) if use_acc else 1.0
        if t > 0:
            dt = max(dts[t], 1e-3) if dts is not None else DT
            s = np.linalg.norm(z[t] - z[t - 1]) / dt
            ckin = 1 / (1 + (max(0, s - VMAX)) ** 2) if use_kin else 1.0
            ctmp = 1 / (1 + (max(0, dt / nominal - 1.5)) ** 2) if (use_tmp and dts is not None) else 1.0
        else:
            ckin = ctmp = 1.0
        c[t] = np.clip(cacc * ckin * ctmp, 1e-3, 1)
    return c


def metrics(est, truth):
    rmse = np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1)))
    origin = np.linalg.norm(est[0] - truth[0])
    d = np.diff(est, axis=0); ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs(np.diff(ang)); turn = np.minimum(turn, 2 * np.pi - turn)
    return rmse, origin, np.degrees(np.mean(turn))


def decim(z, mindist):
    keep = [0]; last = z[0]
    for t in range(1, len(z)):
        if np.linalg.norm(z[t] - last) >= mindist:
            keep.append(t); last = z[t]
    keep = np.array(keep); est = np.zeros_like(z)
    est[:, 0] = np.interp(np.arange(len(z)), keep, z[keep, 0])
    est[:, 1] = np.interp(np.arange(len(z)), keep, z[keep, 1])
    return est


def run(rate, seed=SEED, return_example=False):
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ['raw', 'dec', 'accdec', 'kalman', 'chikf', 'cwtr']}
    example = None
    for m in range(M):
        truth = gen_truth(rng); z, a, sigma = gen_obs(truth, rate, rng)
        res['raw'].append(metrics(z, truth))
        res['dec'].append(metrics(decim(z, 5.0), truth))
        zacc = z.copy(); mask = a <= 30
        if mask.sum() >= 2:
            idx = np.where(mask)[0]
            zacc[:, 0] = np.interp(np.arange(len(z)), idx, z[idx, 0])
            zacc[:, 1] = np.interp(np.arange(len(z)), idx, z[idx, 1])
        res['accdec'].append(metrics(decim(zacc, 5.0), truth))
        mv = np.mean(sigma ** 2)
        res['kalman'].append(metrics(kalman_rts(z, [np.eye(2) * mv] * T), truth))
        res['chikf'].append(metrics(kalman_rts_chi2(z, a), truth))
        c = conf(z, a)
        Rc = [np.eye(2) * ((SIGMA0 ** 2 / c[t]) * (GATE if c[t] < TAU else 1.0)) for t in range(T)]
        cwtr_est = kalman_rts(z, Rc)
        res['cwtr'].append(metrics(cwtr_est, truth))
        if return_example and m == 0:
            example = dict(truth=truth, z=z, cwtr=cwtr_est, c=c)
    out = {k: np.array(v) for k, v in res.items()}
    return (out, example) if return_example else out


if __name__ == "__main__":
    lab = {'raw': 'Raw fixes', 'dec': 'Decimation (5 m)', 'accdec': 'Accuracy gate + decim.',
           'kalman': 'Kalman/RTS (fixed R)', 'chikf': 'Chi2-gated KF (adapt. R)',
           'cwtr': 'CWTR (proposed)'}

    print(f"seed={SEED}, M={M} trials, primary outlier rate={PRIMARY_RATE}\n")
    out = run(PRIMARY_RATE)
    print("=== MAIN RESULTS (primary operating point) ===")
    print(f"{'Method':<26}{'RMSE (m)':<18}{'Origin (m)':<18}{'Jitter (deg)':<12}")
    for k in ['raw', 'dec', 'accdec', 'kalman', 'chikf', 'cwtr']:
        A = out[k]
        print(f"{lab[k]:<26}{A[:,0].mean():5.2f} +/- {A[:,0].std():4.2f}     "
              f"{A[:,1].mean():5.2f} +/- {A[:,1].std():4.2f}     {A[:,2].mean():4.1f}")
    cw = out['cwtr'][:, 0]; ka = out['kalman'][:, 0]; ra = out['raw'][:, 0]
    cwo = out['cwtr'][:, 1]; kao = out['kalman'][:, 1]
    cwj = out['cwtr'][:, 2]; kaj = out['kalman'][:, 2]
    W, p = wilcoxon_signedrank(cw, ka)
    ck = out['chikf'][:, 0]
    Wc, pc = wilcoxon_signedrank(cw, ck)
    print(f"\nCWTR vs Kalman RMSE: {100*(1-cw.mean()/ka.mean()):+.1f}%   "
          f"CWTR vs raw RMSE: {100*(1-cw.mean()/ra.mean()):+.1f}%")
    print(f"CWTR vs Chi2-KF RMSE: {100*(1-cw.mean()/ck.mean()):+.1f}%   "
          f"Wilcoxon p={pc:.2e}")
    print(f"CWTR vs Kalman origin-return: {100*(1-cwo.mean()/kao.mean()):+.1f}%   "
          f"jitter: {100*(1-cwj.mean()/kaj.mean()):+.1f}%")
    print(f"Wilcoxon signed-rank (CWTR vs Kalman RMSE): W={W:.0f}, p={p:.2e}")

    print("\n=== SENSITIVITY SWEEP (multipath severity) ===")
    print(f"{'rate':<8}{'raw':<10}{'kalman':<12}{'chikf':<12}{'cwtr':<12}{'vs Kalman':<12}{'vs Chi2-KF':<12}")
    for rate in [0.02, 0.04, 0.06, 0.08, 0.12, 0.18]:
        o = run(rate)
        r = o['raw'][:, 0].mean(); kk = o['kalman'][:, 0].mean(); cc = o['cwtr'][:, 0].mean()
        xk = o['chikf'][:, 0].mean()
        print(f"{rate:<8.2f}{r:<10.2f}{kk:<12.2f}{xk:<12.2f}{cc:<12.2f}{100*(1-cc/kk):+.1f}%      {100*(1-cc/xk):+.1f}%")
