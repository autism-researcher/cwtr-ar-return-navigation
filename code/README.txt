CWTR paper — reproducible simulation harness
=============================================
Requirements: Python 3, numpy (matplotlib only for figures).
Run each script from THIS directory.

  python3 cwtr_simulation.py     -> Table I main results + multipath sensitivity sweep
  python3 extra_modules.py       -> ablation, wandering detection, confidence-gated
                                    false-trigger, behaviour-adaptive bandit, profile
                                    cost matrix, runtime
  python3 make_figures.py        -> regenerates all data-driven figures into ./figures/

Single fixed seed (SEED=42). Every numerical value, table entry, and data-driven
figure in the manuscript is the direct output of these scripts — no hand-editing.

Operating point: the main table is reported at a heavy-multipath regime
(PRIMARY_RATE = 0.12, per-step outlier probability modulated by HDOP) representative
of dense-crowd urban-canyon reception. Change PRIMARY_RATE in cwtr_simulation.py to
explore other regimes; the sensitivity sweep (and Fig. 5) shows the full range,
including the low-multipath crossover where the fixed-gain Kalman is as good or better.

Key reproduced values (seed 42):
  CWTR RMSE 3.76 +/- 2.10 m  |  Kalman 4.47 +/- 0.81 m  |  raw 17.46 +/- 2.36 m
  CWTR vs Kalman: -15.9% RMSE (Wilcoxon p = 9.7e-9);  vs raw: -78.5%;  jitter -65%
  Ablation: full 3.76 / no-kinematic 4.33 / no-accuracy 3.60 m
  Wandering: AUC 0.99, F1 0.99       Confidence-gated false trigger: 100% -> ~8%
  Bandit (illustrative model): J 0.43 -> 0.39 (~10%)   Profile matrix: -66% (modeled)

NOTE: the v8 draft's numbers (e.g. -39.5% vs Kalman, raw 23 m) are NOT produced by an
independent implementation of the described method and were not used. The harness above
is the authoritative source for all reported figures.

----------------------------------------------------------------------
REAL-DATA pipeline (Campaign A) — analyze_real_traces.py
----------------------------------------------------------------------
Put recorded GPS traces in ./real_traces/ :
    <name>.csv          columns: timestamp, latitude, longitude, accuracy(m)
    <name>.truth.csv    ground-truth polyline: latitude, longitude
    <name>.markers.csv  (optional) rows: start,lat,lon  and  dest,lat,lon
Run:  python3 analyze_real_traces.py
  -> prints per-trace + aggregate table, writes real_trace_results.csv
     and real_trace_rmse.png. Same CWTR + baselines as the simulation.
Test first:  python3 analyze_real_traces.py --make-demo   (SYNTHETIC; not for paper)
Sample format files (demo_SYNTHETIC.*) are in ./real_traces/ as templates.
Campaign B data-entry workbook: ../real_data/CampaignB_route_measurements.xlsx
Protocol (lock before collecting): ../real_data/PREREGISTERED_PROTOCOL.md
