"""
make_boxplots.py — generates BOTH box-plot figures of the paper:
  fig_boxsim  : Monte-Carlo per-trial RMSE distribution (simulation)
  fig_boxfield: field-campaign per-trace jitter by stratum (real data)
USAGE: cd code && python3 make_boxplots.py
OUTPUT: ../figures/fig_boxsim.{pdf,png}, ../figures/fig_boxfield.{pdf,png}
"""
def _run_boxsim():
    """
    Figure 3 - Monte-Carlo per-trial RMSE distribution (Reviewer #1, comment 3)
    ===========================================================================
    SIMULATION-BASED. Runs cwtr_simulation.run() at the heavy-multipath operating
    point and draws a BOX PLOT of the per-trial position RMSE (M trials) for each
    method, showing CWTR's tight low distribution and the chi2-gated KF's long
    upper tail (episodic burst lock-on).
    
    USAGE:   cd code && python3 make_boxplots.py
    OUTPUT:  ../figures/fig_boxsim.{pdf,png}
             ../IEEEAccess_Submission/figures/fig_boxsim.{pdf,png}   (replaces placeholder)
    """
    import os
    import numpy as np
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from cwtr_simulation import run, PRIMARY_RATE, M
    
    LAB = {'raw': 'Raw fixes', 'dec': 'Decimation (5 m)', 'accdec': 'Accuracy+decim.',
           'kalman': 'Kalman/RTS (fixed R)', 'chikf': 'Chi2-gated KF (adapt. R)',
           'cwtr': 'CWTR (proposed)'}
    METHODS = ['raw', 'dec', 'accdec', 'kalman', 'chikf', 'cwtr']
    
    
    def main():
        out = run(PRIMARY_RATE)                    # out[m] is (M x 3): RMSE, origin, jitter
        data = [out[m][:, 0] for m in METHODS]     # column 0 = RMSE
        plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', ms=3, mec='0.5'),
                        medianprops=dict(color='black', lw=1.2))
        cols = ['0.75', '0.75', '0.82', '#9DC3E6', '#E69B96', '#5B9BD5']
        for box, c in zip(bp['boxes'], cols):
            box.set(facecolor=c, alpha=0.85, edgecolor='0.3', lw=0.7)
        ax.set_xticklabels([LAB[m].split(' (')[0] for m in METHODS], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Position RMSE (m)')
        ax.set_title(f'Per-trial RMSE distribution (M={M} Monte-Carlo trials)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        for d in ['../figures']:
            if os.path.isdir(d):
                fig.savefig(os.path.join(d, 'fig_boxsim.pdf'))
                fig.savefig(os.path.join(d, 'fig_boxsim.png'), dpi=300)
                print('wrote', os.path.join(d, 'fig_boxsim.pdf'))
    
    
    if __name__ == '__main__':
        main()

def _run_boxfield():
    """
    Figure 10 - Field-campaign jitter distribution by stratum (Reviewer #1, comment 3)
    ==================================================================================
    REAL DATA. Runs the IDENTICAL CWTR pipeline + baselines (imported from
    analyze_field_traces / analyze_real_traces) over every Sensor-Logger recording
    in  ../data/GPS_Data/extracted/<batch>/<recording>/Location.csv , then draws a
    BOX PLOT of the per-trace trajectory jitter for each method, grouped by
    reception stratum (benign / moderate). Severe traces (N=3) are excluded from
    the comparison, as in the paper.
    
    USAGE:   cd code && python3 make_boxplots.py
    OUTPUT:  ../figures/fig_boxfield.{pdf,png}
             ../IEEEAccess_Submission/figures/fig_boxfield.{pdf,png}   (replaces placeholder)
    """
    import os, glob
    import numpy as np
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from analyze_field_traces import process, stratum, EXCLUDE, METHODS, LAB
    
    ROOT = '../data/GPS_Data/extracted'          # <- your real field recordings
    
    
    def collect():
        files = sorted(glob.glob(os.path.join(ROOT, '*', '*', 'Location.csv')))
        files = [f for f in files if os.path.basename(os.path.dirname(f)) not in EXCLUDE]
        recs = []
        for fp in files:
            name = os.path.basename(os.path.dirname(fp))
            try:
                per, char, _ = process(fp)
            except Exception as e:
                print(f'  SKIP {name}: {e}'); continue
            recs.append({'stratum': stratum(char['med_acc']), 'per': per})
        return recs
    
    
    def main():
        recs = collect()
        strata = ['benign', 'moderate']            # severe (N=3) excluded, as in paper
        nb = sum(r['stratum'] == 'benign' for r in recs)
        nm = sum(r['stratum'] == 'moderate' for r in recs)
        ns = sum(r['stratum'] == 'severe' for r in recs)
        print(f'collected {len(recs)} traces  (benign {nb}, moderate {nm}, severe {ns} [excluded])')
    
        plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        scol = {'benign': '#4C7DB2', 'moderate': '#C99A3C'}
        w = 0.34
        xb = np.arange(len(METHODS))
        handles = {}
        for i, s in enumerate(strata):
            rs = [r for r in recs if r['stratum'] == s]
            data = [np.array([r['per'][m]['jitter'] for r in rs]) for m in METHODS]
            pos = xb + (i - 0.5) * w
            bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True,
                            showfliers=True, flierprops=dict(marker='.', ms=3, mec='0.5'),
                            medianprops=dict(color='black', lw=1.2))
            for box in bp['boxes']:
                box.set(facecolor=scol[s], alpha=0.75, edgecolor='0.3', lw=0.7)
            handles[s] = bp['boxes'][0]
        ax.set_xticks(xb)
        ax.set_xticklabels([LAB[m].split(' (')[0] for m in METHODS], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('Trajectory jitter (deg)')
        ax.legend([handles[s] for s in strata],
                  [f'benign (N={nb})', f'moderate (N={nm})'], fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        for d in ['../figures']:
            if os.path.isdir(d):
                fig.savefig(os.path.join(d, 'fig_boxfield.pdf'))
                fig.savefig(os.path.join(d, 'fig_boxfield.png'), dpi=300)
                print('wrote', os.path.join(d, 'fig_boxfield.pdf'))
    
    
    if __name__ == '__main__':
        main()

if __name__ == "__main__":
    _run_boxsim()
    _run_boxfield()
