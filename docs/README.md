# Cruise Lab: overview

Cruise Lab is a small, self-contained generator and viewer for **synthetic B747 cruise-phase flight data**. The data comes from the JSBSim flight-dynamics engine, which is also the physics inside FlightGear. Each flight is either:
- **normal**,
- an **observation anomaly**: a sensor reading is corrupted *before* the autopilot uses it, or
- a **control anomaly**: an autopilot command is corrupted *after* the autopilot computes it.

Every flight logs **all ~680 JSBSim properties plus the sensor readings and autopilot commands**. About **329 of these channels change in flight**. Each anomalous flight also comes with a **normal twin**: the same flight, flown again without the anomaly. The viewer overlays the twin, so you can see exactly what the anomaly changed.

## Quick start

From `D:\IISc\flightgear\cruise_lab`, using the project venv:

```
..\.venv\Scripts\python -m lab.generate               # demo set: 4 normal + 8 observation + 8 control (~1 min, ~330 MB)
..\.venv\Scripts\python viewer\server.py              # then open http://localhost:8050
```

Other data sets:

```
..\.venv\Scripts\python -m lab.generate --preset catalog          # every anomaly type x variable (64) + 10 normal (~4 min, ~1.3 GB)
..\.venv\Scripts\python -m lab.generate --preset random --n 30    # random draws: 1/3 normal, 1/3 observation, 1/3 control
```

Re-running a preset only adds flights that are missing. To start over, delete `data\`.

Requirements: Python 3.10+ with `jsbsim` (1.3.1 tested), `numpy`, `pandas` and `pyarrow` (see `requirements.txt`). The viewer loads Plotly and the docs page loads `marked` from a CDN, so the browser needs internet access.

## The idea in one picture

```
 flight plan ──► AUTOPILOT ──command──►[ ACTUATORS ]──applied──► JSBSim B747 ──┐
 (sel/*)           ▲  (ap/*)          ▲                         (fcs/*-pos,     │ true state
                   │                  │ CONTROL anomaly          engines, ...)  │
                   │                  │ (after the autopilot)                   │
                   └──reading──[ SENSORS ]◄──────────────────────────────────────┘
                     (sensor/*)    ▲
                                   │ OBSERVATION anomaly (before the autopilot)
```

- **Observation anomaly.** The autopilot reads a wrong value and makes a *correct* decision from it. The aircraft really moves away from where it should be, while the faulty sensor often looks normal.
- **Control anomaly.** The autopilot reads correct values and makes the right decision, but the command is altered on its way to the surface or engine. The autopilot then sees the aircraft respond wrongly and compensates.

## Documents

| File | What it explains |
|---|---|
| [01_how_it_works.md](01_how_it_works.md) | How one flight is generated, step by step: scenario, closed loop, autopilot equations, logging, noise, the normal twin |
| [02_anomalies.md](02_anomalies.md) | **Every anomaly type**: where it enters, how it is injected (formula), magnitude limits, time profile, onset and duration, and what you will see |
| [03_channels.md](03_channels.md) | **Every logged channel** that changes in flight, grouped by subsystem, with unit and meaning (generated from the data) |
| [04_viewer.md](04_viewer.md) | How to use the viewer: flight drop-down, the four chart panels, roles, effect scores, overlays |
| [05_data_format.md](05_data_format.md) | Files on disk, column conventions, labels, manifest, effect scores, how to load a flight in Python |
| [06_qcd_dataset.md](06_qcd_dataset.md) | **Training dataset** for the task-aware QCD code: windows, splits, all 14 anomaly types, channels, normalisation, validation |

## Folder layout

```
cruise_lab/
  lab/                 generator (pure Python + JSBSim)
    aircraft.py        JSBSim set-up: B747 copy with payload, wind, turbulence, EGT law
    autopilot.py       altitude / speed / heading / sideslip loops
    scenario.py        per-flight random conditions (normal variability)
    anomalies.py       anomaly catalogue, sampling, injection operators
    simulate.py        closed loop at 60 Hz, logs every property at 4 Hz
    generate.py        CLI: presets, normal twins, effect scores, manifest, channel table
    qcd_dataset.py     training dataset for the task-aware QCD code (npz windows)
    qcd_check.py       validates that dataset, naive baseline
    property_info.csv  meaning/status of each JSBSim property (from the project's property inventory)
  viewer/
    server.py          local web server (standard library + pandas), JSON API
    static/            index.html, app.js, style.css, docs.html
  docs/                these documents
  data/                generated: flights/, meta/, manifest.json, channels.json/.csv
  qcd_data/            generated: QCD training dataset (see 06_qcd_dataset.md)
```
