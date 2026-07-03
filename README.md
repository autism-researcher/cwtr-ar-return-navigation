# CWTR: Self-Trained, Map-Free AR Return Navigation — Code & Field Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21029166.svg)](https://doi.org/10.5281/zenodo.21029166)

Archived at Zenodo: DOI [10.5281/zenodo.21029166](https://doi.org/10.5281/zenodo.21029166).

Reproducibility package for the IEEE Access manuscript:

> M. B. Hossain et al., "Self-Trained, Map-Free AR Return Navigation with
> Confidence-Weighted Trajectory Reconstruction."

Every numerical value, table entry, and data-driven figure in the paper is the
direct output of the scripts in `code/` run on the data in this repository,
from a single fixed seed (SEED=42). Nothing is hand-edited.

## Contents

| Path | Description |
|---|---|
| `simulation/` | Simulation harness and analysis scripts (Python 3 + NumPy; Matplotlib for figures) |
| `simulation/real_traces/` | Campaign A ground-truthed walking traces (N=8) with surveyed truth polylines |
| `data/data_groundtruth/` | Raw Sensor Logger recordings (Location/Metadata/Annotation CSV) for the ground-truthed routes |
| `data/GPS_Data/extracted/` | High-multipath field campaign, June 2026: 57 Sensor Logger recordings (b01–b06 Haram district, Madinah; b07 UPM campus). 55 analysed, 2 excluded as stated in the paper |
| `data/CampaignB_route_measurements.xlsx` | Campaign B on-device end-point accuracy measurements (21 runs over 7 routes) |
| `figures/` | Output directory for regenerated figures |

## Reproducing the paper

```bash
cd simulation
python3 cwtr_simulation.py      # Table I main results + multipath sensitivity sweep
python3 extra_modules.py        # ablation, wandering detection, confidence gating,
                                # behaviour-adaptive bandit, profile cost matrix, runtime
python3 analyze_real_traces.py  # Campaign A ground-truthed real-trace table (N=8)
python3 analyze_field_traces.py # field-campaign characterization + metrics (N=55)
python3 make_figures.py         # regenerates the core data-driven figures into ../figures/
# See reproduce_results.md for the full artifact-to-script mapping.
python3 param_sensitivity.py    # parameter-sensitivity sweep (Fig. 6, Sec. V-E)
python3 make_boxplots.py        # box-plot figures: MC RMSE distribution + field jitter by stratum
python3 bench_cwtr.py           # runtime/memory benchmark (Table 6)
```

Requirements: Python ≥ 3.9, `numpy`, `matplotlib` (figures only). No other dependencies.

## Data notes

* Recordings were made with the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) iOS app
  (iPhone 12 Pro, iOS 18.5). Each recording folder contains `Location.csv`
  (GNSS fixes: timestamp, lat/lon, reported horizontal accuracy, speed, bearing),
  `Metadata.csv`, and `Annotation.csv`.
* Recording folders are renamed to neutral identifiers (`b01_haram_r01`, …).
  The renaming preserves lexicographic processing order; analysis outputs are
  byte-identical to those produced from the originally named recordings.
* Pre-recording cached fixes (`seconds_elapsed < 0`) are dropped by the
  analysis scripts, not removed from the raw files.

## Licenses

* Code: MIT (see `LICENSE`)
* Data (`GPS_Data/`, `data_groundtruth/`, `simulation/real_traces/`, `CampaignB_route_measurements.xlsx`): CC BY 4.0 (see `DATA_LICENSE`)

## Citation

See `CITATION.cff`. If you use this code or data, please cite the paper above
and the Zenodo archive DOI shown in the repository sidebar.
