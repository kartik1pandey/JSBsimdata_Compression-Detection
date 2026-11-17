# Cruise Lab

Minimal JSBSim generator and viewer for synthetic B747 **cruise-phase** flight data. Each flight is either normal or carries an **observation anomaly** (a sensor corrupted before the autopilot reads it) or a **control anomaly** (a command corrupted after the autopilot issues it).

```
..\.venv\Scripts\python -m lab.generate          # demo: 20 flights + normal twins, ~1 min
..\.venv\Scripts\python viewer\server.py         # open http://localhost:8050
```

- Every flight logs all ~680 JSBSim properties at 4 Hz, of which ~329 change in flight. It also logs the 9 sensor readings the autopilot uses, the autopilot commands and the flight plan.
- Each anomalous flight also has a normal twin: same seed, same turbulence, no anomaly. Before onset the twin is identical to the flight, sample for sample.
- The viewer has a flight drop-down and four chart panels with any channel. It shows the anomaly window, the twin and true-value overlays, and per-channel roles (injection point, strong, moderate or weak response).

Training data for the task-aware QCD code: `..\.venv\Scripts\python -m lab.qcd_dataset`, then `-m lab.qcd_check` (see [docs/06_qcd_dataset.md](docs/06_qcd_dataset.md)).

Full documentation: [docs/README.md](docs/README.md), also served at http://localhost:8050/docs.html.

## Task-aware compression + change detection (`qcd_spec/`)

The supervisor's task-aware compression method, implemented from the spec and trained on the `qcd_pure` dataset
(built with `python -m lab.qcd_pure`; the data itself is not in this repository):

- `qcd_spec/spec_run.py`: one training run (encoder f, decoder g, likelihood models q0/q1, CUSUM calibration and test).
- `qcd_spec/models.py`: transformer / GRU / TCN / linear backbones, diagonal-Gaussian and mixture-of-Gaussians heads.
- `qcd_spec/report_full.py`: pools every result into `qcd_spec/results/FULL_TABLES.md` and `qcd_spec/docs/figs/full/`.
- `qcd_spec/plot_runs.py`: per-run dashboards (train vs validation losses, codes, CUSUM).
- `qcd_spec/docs/GUIDE.md`, `EXPLAIN.md` and `FIGURES.md`: what every term, figure and number means.
- `qcd_spec/make_more_figures.py`: architecture diagrams and the cross-variation result figures (`docs/figs/more/`).

```
set QCD_DATA=path\to\qcd_pure
python qcd_spec\spec_run.py --pipeline cae --enc tf --q tf --k 8 --lam 0.1 --head mog --enc-out linear --seed 0
python qcd_spec\report_full.py
```
