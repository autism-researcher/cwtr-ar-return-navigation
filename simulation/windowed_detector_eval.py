"""
Windowed disorientation detector — evaluates the runtime rule EXACTLY as
specified in Section V-J of the manuscript (Eq. 6):

    S_W = a1*(1 - SR_W) + a2*sigma_dtheta_norm + a3*r_norm,
    (a1,a2,a3) = (0.5, 0.3, 0.2),  window Nw = 30 samples (~30 s @ 1 Hz),
    alarm iff S_W > theta_w = 0.6 sustained >= tau_d = 10 s,
    suppressed whenever mean window confidence c_bar < theta_c = 0.4,
    S_W computed on the CWTR reconstruction p_hat (not raw fixes).

Normalization choices (the manuscript states "normalized to [0,1]" without
formulas; these are the reference choices, documented here):
  * sigma_dtheta_norm = SD(|heading change|) / (pi/2), clipped to [0,1]
    (pi/2 is the maximum possible SD of a variable bounded in [0, pi]).
  * r_norm = (# 5-m grid cells entered more than once) / (# distinct cells
    visited in the window)  in [0,1]  ("fraction of visited cells revisited").

Test sets are IDENTICAL (same generators, same seeds, same order) to
extra_modules.py:
  seed 11 : 250 purposeful (hsd=0.18) then 250 wandering  (clean; detection)
  seed 24 : 200 goal-directed walks (hsd=0.08) + sigma=4 m noise + 12%
            multipath outliers of 15-55 m                  (false-trigger)

Baselines reported:
  naive    = same windowed rule on RAW fixes, no confidence gate
  proposed = windowed rule on CWTR reconstruction, with c_bar gate

Run:  python3 windowed_detector_eval.py   (from this directory)
"""
import numpy as np
from collections import Counter
import cwtr_simulation as S
from extra_modules import gen_purposeful, gen_wandering

NW, THETA_W, TAU_D, THETA_C = 30, 0.6, 10, 0.4
W1, W2, W3 = 0.5, 0.3, 0.2


def window_score(win):
    """Eq. (6) on one window of positions (Nw x 2)."""
    d = np.diff(win, axis=0)
    steps = np.linalg.norm(d, axis=1)
    L = steps.sum()
    net = float(np.linalg.norm(win[-1] - win[0]))
    sr = min(net / (L + 1e-9), 1.0)                       # straightness ratio
    ang = np.arctan2(d[:, 1], d[:, 0])
    dh = np.abs(np.diff(ang))
    dh = np.minimum(dh, 2 * np.pi - dh)                   # in [0, pi]
    sig = min(float(dh.std()) / (np.pi / 2), 1.0)         # normalized SD
    cells = Counter(tuple(x) for x in np.round(win / 5.0).astype(int))
    ncells = len(cells)
    revis = sum(1 for v in cells.values() if v > 1)
    r = revis / max(ncells, 1)                            # fraction revisited
    return W1 * (1.0 - sr) + W2 * sig + W3 * r


def confidence_series(z, a, dt=1.0):
    """Per-fix confidence, identical formula to the released pipelines."""
    T = len(z)
    c = np.ones(T)
    for t in range(T):
        s = np.linalg.norm(z[t] - z[t - 1]) / dt if t > 0 else 0.0
        cacc = 1.0 / (1.0 + (a[t] / S.A0) ** 2)
        ckin = 1.0 / (1.0 + (max(0.0, s - S.VMAX)) ** 2)
        c[t] = np.clip(cacc * ckin, 1e-3, 1.0)
    return c


def detect(pos, c, gate=True):
    """Runtime rule: alarm iff S_W>theta_w sustained tau_d s, gate on c_bar.
    Returns (fired, n_alarm_windows, n_suppressed_windows, n_windows)."""
    T = len(pos)
    consec = 0
    fired = False
    n_alarm = n_supp = n_win = 0
    for t in range(NW - 1, T):
        win = pos[t - NW + 1: t + 1]
        n_win += 1
        if gate and float(np.mean(c[t - NW + 1: t + 1])) < THETA_C:
            n_supp += 1
            consec = 0
            continue
        if window_score(win) > THETA_W:
            n_alarm += 1
            consec += 1
            if consec >= TAU_D:
                fired = True
        else:
            consec = 0
    return fired, n_alarm, n_supp, n_win


def cwtr_reconstruct(z, a):
    c = confidence_series(z, a)
    Rc = [np.eye(2) * ((S.SIGMA0 ** 2 / c[t]) * (S.GATE if c[t] < S.TAU else 1.0))
          for t in range(len(z))]
    return S.kalman_rts(z, Rc), c


def main():
    # ---- detection test: clean purposeful vs wandering (seed 11 order) ----
    r = np.random.default_rng(11)
    purp = [gen_purposeful(r) for _ in range(250)]
    wand = [gen_wandering(r) for _ in range(250)]

    def run_clean(trajs):
        hits = 0
        for p in trajs:
            a = np.full(len(p), 4.0)                      # good reception
            phat, c = cwtr_reconstruct(p, a)
            fired, *_ = detect(phat, c, gate=True)
            hits += fired
        return hits

    rec = run_clean(wand)
    fp = run_clean(purp)
    print("=== WINDOWED DETECTOR (Eq. 6, theta_w=0.6, tau_d=10 s, theta_c=0.4) ===")
    print(f"wandering recall  (250 clean wandering):   {rec/250:.3f}")
    print(f"false-positive rate (250 clean purposeful): {fp/250:.3f}")

    # ---- false-trigger test: corrupted goal-directed walks (seed 24) ----
    r = np.random.default_rng(24)
    raw_fired = gated_fired = 0
    supp_frac = []
    N = 200
    for _ in range(N):
        truth = gen_purposeful(r, 0.08)
        T = len(truth)
        z = truth + r.normal(0, 4.0, (T, 2))
        for t in range(T):
            if r.random() < 0.12:
                z[t] = truth[t] + r.uniform(15, 55) * np.array(
                    [np.cos(r.uniform(0, 2 * np.pi)), np.sin(r.uniform(0, 2 * np.pi))])
        a = np.full(T, 4.0)
        c = confidence_series(z, a)
        f_raw, *_ = detect(z, np.ones(T), gate=False)     # naive: raw fixes, no gate
        phat, c2 = cwtr_reconstruct(z, a)
        f_g, n_al, n_sp, n_w = detect(phat, c2, gate=True)
        raw_fired += f_raw
        gated_fired += f_g
        supp_frac.append(n_sp / max(n_w, 1))
    print("\n=== FALSE-TRIGGER TEST (N=200 corrupted goal-directed walks) ===")
    print(f"naive windowed detector (raw fixes):        {100*raw_fired/N:.0f}%")
    print(f"gated windowed detector (CWTR + c_bar gate): {100*gated_fired/N:.0f}%")
    print(f"mean fraction of windows gate-suppressed:    {np.mean(supp_frac):.3f}")




def threshold_sweep():
    """Sustained-S_W distributions and theta_w operating points (clean sets)."""
    r = np.random.default_rng(11)
    purp = [gen_purposeful(r) for _ in range(250)]
    wand = [gen_wandering(r) for _ in range(250)]

    def max_sustained(pos, c):
        T = len(pos); sw = []
        for t in range(NW - 1, T):
            if float(np.mean(c[t - NW + 1: t + 1])) < THETA_C: sw.append(-1.0)
            else: sw.append(window_score(pos[t - NW + 1: t + 1]))
        sw = np.array(sw); best = 0.0
        for i in range(len(sw) - TAU_D + 1):
            seg = sw[i:i + TAU_D]
            if (seg >= 0).all(): best = max(best, float(seg.min()))
        return best

    def stats(trajs):
        out = []
        for p in trajs:
            a = np.full(len(p), 4.0)
            phat, c = cwtr_reconstruct(p, a)
            out.append(max_sustained(phat, c))
        return np.array(out)

    mw, mp = stats(wand), stats(purp)
    print("\n=== THETA_W OPERATING POINTS (sustained 10-s windowed S_W, clean sets) ===")
    print("wandering  sustained-S_W: min %.3f  median %.3f  max %.3f" % (mw.min(), np.median(mw), mw.max()))
    print("purposeful sustained-S_W: min %.3f  median %.3f  max %.3f" % (mp.min(), np.median(mp), mp.max()))
    for th in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        print("  theta_w=%.2f : wandering recall %.3f   purposeful FP %.3f" % (th, (mw > th).mean(), (mp > th).mean()))
    s = np.concatenate([mp, mw]); lab = np.array([0] * len(mp) + [1] * len(mw))
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    print("AUC of sustained windowed S_W: %.4f" % ((rk[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)))


if __name__ == "__main__":
    main()
    threshold_sweep()
