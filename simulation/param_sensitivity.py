"""
Figure 6 - Parameter sensitivity (Reviewer #1, comment 2)
=========================================================
SIMULATION-BASED (does NOT use field data). Re-runs the CWTR Monte-Carlo
pipeline of cwtr_simulation.py at the heavy-multipath operating point while
sweeping each parameter one-at-a-time about its operating value, holding the
others fixed. Records reconstruction RMSE and trajectory jitter.

Parameters swept:  a0 (accuracy scale), vmax (speed ceiling),
                   tau (gate threshold), clip floor.

USAGE:   cd code && python3 param_sensitivity.py
OUTPUT:  ../figures/fig_paramsens.{pdf,png}

"""
import os
import math
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import cwtr_simulation as S


def plain_y(ax_, nbins=4, labelsize=8):
    """Plain decimal y-tick labels: no offset text (e.g. '1e-6 + 1.2579'),
    no scientific notation; decimals adapted to the tick spacing."""
    ax_.yaxis.set_major_locator(MaxNLocator(nbins=nbins))

    def _fmt(x, _pos):
        locs = ax_.yaxis.get_majorticklocs()
        steps = np.diff(np.sort(locs))
        step = float(np.min(steps[steps > 0])) if len(steps) and np.any(steps > 0) else 0.0
        dec = max(0, min(7, -int(math.floor(math.log10(step))))) if step > 0 else 2
        return f"{x:.{dec}f}"

    ax_.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax_.yaxis.get_offset_text().set_visible(False)
    ax_.tick_params(axis='y', labelsize=labelsize)
from cwtr_simulation import (gen_truth, gen_obs, kalman_rts, metrics,
                             T, M, SIGMA0, GATE, DT, SEED, PRIMARY_RATE)

# operating (default) values
A0d, VMAXd, TAUd, CLIPd = S.A0, S.VMAX, S.TAU, 1e-3
SWEEP_M = M          # trials per point (same as the main table)


def conf_p(z, a, a0, vmax, clip):
    """Confidence model with the four parameters made explicit."""
    n = len(z); c = np.ones(n)
    for t in range(n):
        cacc = 1.0 / (1.0 + (a[t] / a0) ** 2)
        s = np.linalg.norm(z[t] - z[t - 1]) / DT if t > 0 else 0.0
        ckin = 1.0 / (1.0 + (max(0.0, s - vmax)) ** 2)
        c[t] = np.clip(cacc * ckin, clip, 1.0)
    return c


def cwtr_eval(a0, vmax, tau, clip, rate=PRIMARY_RATE, seed=SEED):
    """Mean CWTR RMSE and jitter over SWEEP_M seeded trials. The rng is reset
    to the same seed each call, so only the swept parameter changes (controlled
    one-at-a-time comparison)."""
    rng = np.random.default_rng(seed)
    rmse, jit = [], []
    for _ in range(SWEEP_M):
        truth = gen_truth(rng)
        z, a, sigma = gen_obs(truth, rate, rng)
        c = conf_p(z, a, a0, vmax, clip)
        Rc = [np.eye(2) * ((SIGMA0 ** 2 / c[t]) * (GATE if c[t] < tau else 1.0))
              for t in range(T)]
        r, o, j = metrics(kalman_rts(z, Rc), truth)
        rmse.append(r); jit.append(j)
    return float(np.mean(rmse)), float(np.mean(jit))


# (label, default, sweep values, x-as-log)
SWEEPS = [
    ("$a_0$ (m)",                 A0d,   [4, 6, 8, 10, 12, 14, 16],          False),
    ("$v_{\\max}$ (m/s)",         VMAXd, [1.8, 2.1, 2.5, 2.8, 3.2, 3.5],     False),
    ("$\\tau$ (gate threshold)",  TAUd,  [0.1, 0.2, 0.3, 0.4, 0.5],          False),
    ("clip floor",               CLIPd, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],      True),
]


def main():
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for ax, (label, default, vals, logx) in zip(axes.ravel(), SWEEPS):
        rm, jt = [], []
        for v in vals:
            kw = dict(a0=A0d, vmax=VMAXd, tau=TAUd, clip=CLIPd)
            if label.startswith("$a_0"):       kw['a0'] = v
            elif label.startswith("$v_"):       kw['vmax'] = v
            elif label.startswith("$\\tau"):    kw['tau'] = v
            else:                                kw['clip'] = v
            r, j = cwtr_eval(**kw)
            rm.append(r); jt.append(j)
        rm, jt = np.array(rm), np.array(jt)
        l1, = ax.plot(vals, rm, 'o-', color='#1f3870', lw=1.8, ms=4, label='RMSE (m)')
        ax.set_xlabel(label); ax.set_ylabel('RMSE (m)', color='#1f3870')
        ax.tick_params(axis='y', labelcolor='#1f3870')
        if logx: ax.set_xscale('log')
        ax.axvline(default, color='0.55', ls='--', lw=1)
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        l2, = ax2.plot(vals, jt, 's--', color='#C0504D', lw=1.5, ms=3.5, label='Jitter (deg)')
        ax2.set_ylabel('Jitter (deg)', color='#C0504D')
        ax2.tick_params(axis='y', labelcolor='#C0504D')
        if label == "clip floor":
            # metrics vary by <0.01% across this sweep: widen the y-spans to a
            # readable scale and annotate, instead of micro-precision ticks
            ax.set_ylim(rm.mean() * 0.995, rm.mean() * 1.005)
            ax2.set_ylim(jt.mean() * 0.995, jt.mean() * 1.005)
            ax.text(0.04, 0.90, 'nearly invariant\n(variation $<0.01\\%$)',
                    transform=ax.transAxes, fontsize=8, color='0.25', va='top')
        plain_y(ax)   # no offset / scientific notation on either y-axis
        plain_y(ax2)
        # mark operating point
        di = int(np.argmin([abs(v - default) for v in vals]))
        ax.plot(vals[di], rm[di], 'o', color='#1f3870', ms=9, mfc='none', mew=1.8)
    axes[0, 0].legend(handles=[l1, l2], loc='upper left', fontsize=8, framealpha=0.9)
    fig.suptitle('CWTR parameter sensitivity (heavy-multipath operating point, '
                 f'M={SWEEP_M})', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ['../figures']:
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, 'fig_paramsens.pdf'))
            fig.savefig(os.path.join(d, 'fig_paramsens.png'), dpi=300)
            print('wrote', os.path.join(d, 'fig_paramsens.pdf'))


if __name__ == '__main__':
    main()
