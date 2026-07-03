# Reproducing the Results

Every numerical value, table, and data-driven figure in the paper is regenerated
by the scripts below from a single fixed seed (SEED=42). Nothing is hand-edited.

Setup: Python >= 3.9, then `pip install -r requirements.txt`.
All commands are run from the `simulation/` directory.

| Paper artifact | Command | Output |
|---|---|---|
| Table 1 (main simulation results) + multipath sensitivity sweep (Fig. 5 data) | `python3 cwtr_simulation.py` | console |
| Ablation, wandering AUC/F1, confidence-gated false-trigger (Fig. 11 data), bandit and profile-cost models (Table 7) | `python3 extra_modules.py` | console |
| Table 3 (ground-truthed real traces, N=8) + Fig. 7 | `python3 analyze_real_traces.py` | console + `figures/realtrace_rmse.pdf` |
| Tables 4-5 (field campaign, N=55) + Figs. 8-9 | `python3 analyze_field_traces.py` | console + CSVs + `figures/fieldtrace_*.pdf` |
| Figs. 2-5, 11 | `python3 make_figures.py` | `figures/` |
| Fig. 6 (parameter sensitivity) | `python3 param_sensitivity.py` | `figures/fig_paramsens.pdf` |
| Figs. 3 and 10 (box plots) | `python3 make_boxplots.py` | `figures/fig_boxsim.pdf`, `figures/fig_boxfield.pdf` |
| Table 6 (runtime/memory benchmark) | `python3 bench_cwtr.py` | console |
| Table 8 (Campaign B on-device accuracy) | open `data/CampaignB_route_measurements.xlsx` | auto-computed Summary sheet |

Data locations: `data/GPS_Data/extracted/` (field campaign, 57 recordings; 55 analyzed),
`simulation/real_traces/` + `data/data_groundtruth/` (Campaign A, N=8),
`data/CampaignB_route_measurements.xlsx` (Campaign B, 21 runs over 7 routes).

Note: `make_figures.py` additionally emits fig7_wandering, fig8_bandit, and
fig9_costmatrix, which correspond to the components summarized in Table 7 of the paper.
