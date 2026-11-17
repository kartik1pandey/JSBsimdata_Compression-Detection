# 6. QCD training dataset (`qcd_data/`)

> **Current version (v3, 2026-11-04).** `qcd_data/` is now rebuilt with: identical window times in every
> flight (720, 840, 960 s; change windows 720 s with onset 760-800 s), every flight a complete 1500 s,
> all 327 changing channels (D = 334 with sin/cos), **no clipping, no duplicate removal, no extra noise**.
> Full details: `qcd_data/DATASET_DETAILS.md` and `qcd_data/channels.csv`. The v2 data described below is
> kept in `qcd_data_v2_random_windows/` (and `qcd_data_noisy/`); the training results in `qcd_experiment/`
> were produced on v2.

A dataset for the supervisor's **task-aware compression for quickest change detection** code
(`task(QCD)_aware_data_compression_VAR1.zip`). It uses the same file layout as that code's simulated VAR data,
so it can replace it directly. Instead of a 100-D VAR process, the sequences are B747 cruise flights with all
14 Cruise Lab anomaly types.

```
..\.venv\Scripts\python -m lab.qcd_dataset              # build (980 flights, ~20 min; add --workers 10 if memory is short)
..\.venv\Scripts\python -m lab.qcd_check                # validate + naive baseline -> check_report.json
..\.venv\Scripts\python -m lab.qcd_dataset --scale 0.1 --out ..\qcd_smoke   # 10 % smoke test
```

## What a sample is

One sample is a **window of T = 480 time steps = 120 s at 4 Hz**, with **D = 185 features**. The features
cover every channel that changes in flight, minus the exclusions listed below.

| File | x shape (full build) | Content |
|---|---|---|
| `train_normal.npz`, `val_normal.npz`, `test_normal.npz` | see "Counts" | Normal flight throughout |
| `train_abnormal.npz`, `val_abnormal.npz`, `test_abnormal.npz` | | The anomaly acts for the whole window |
| `test_change.npz` | | Normal, then the anomaly from `change_point` (step 160–320) to the end |

Keys in every file. The supervisor's loader reads only `x` (and `change_point`); the rest is extra:

| Key | Shape | Meaning |
|---|---|---|
| `x` | (N, T, D) float32 | Standardised features, clipped to ±10 |
| `label` | (N, T) int8 | 1 while the anomaly acts: all of an abnormal window, the part from `change_point` in a change window |
| `change_point` | (N,) | `test_change` only: index of the first abnormal step |
| `flight_id` | (N,) | Flight the window was cut from (see `flights.csv`) |
| `anomaly_type`, `group`, `variable`, `profile` | (N,) | `normal` for normal windows |
| `severity` | (N,) | 0–1, the log-scale position of the magnitude between its limits |
| `window_start_s` | (N,) | Window start, in seconds since the start of cruise |

Other files:
- `normalization.npz`: `mean`, `std`, `channels`. To recover physical units, compute raw = x · std + mean.
- `metadata.json`: config, the feature list with units and meanings, counts, and every dropped channel with the reason.
- `flights.csv`: one row per flight with split, role, anomaly parameters, seed and window start times.
- `check_report.json`: the validation result.

## How it is built

1. **Flights and roles.** Every flight gets exactly **one** role, so no flight appears in two splits:
   - **Every flight is 1,500 s (25 min) long**, normal or abnormal.
   - **Normal flights:** 280 train, 70 validation, 70 test. That is as many as all abnormal flights of each split together (14 types × 20 / 5 / 5), so normal and abnormal windows are **balanced 1 : 1**.
   - **Anomalous flights, per type:** 20 train, 5 validation, 5 test, 10 change. All 14 types get the same numbers, so 560 flights.
   - **Hard conditions** (moderate turbulence, strong wind, hot day, cold day, heavy) are given to 20 % of the flights **in every role, normal and abnormal alike**, rotated over the whole split so each condition appears equally often in both classes. Otherwise a condition would itself carry the label.
2. **Anomaly draw.** Each anomalous flight draws, from its own seed:
   - the variable or engine;
   - severity ~ U(0, 1);
   - the profile: step or ramp. Intermittent bursts are not used, so the fault never switches off;
   - the sign;
   - onset ~ U(400, 700) s.

   **The anomaly starts at the onset and continues to the end of the flight** (800–1,100 s). The per-type maximum durations of Cruise Lab are not applied. Every abnormal window is therefore inside the acting fault, with label 1 throughout. After the change point the fault persists, as change detection assumes.
   **Flight-envelope check.** Some severe control faults, such as a large stuck aileron, can drive the aircraft out of controlled flight, and then JSBSim produces NaN. Each flight is therefore checked against these limits:

   | Quantity | Limit |
   |---|---|
   | Bank | ±45° |
   | Pitch | −15° to +25° |
   | Altitude error from target | 3,000 ft |
   | Mach | at most 0.92 |
   | CAS | at least 160 kt |
   | Any of these values | NaN |

   If the aircraft goes outside the limits, its data are cut **10 s before** that point. Windows are then drawn only from the part that remains. If the flight can no longer give its full 3 windows (1 for a change flight), the anomaly is redrawn from a new seed with severity × 0.7 per attempt, up to 8 attempts. Long faults that are survivable only when mild, such as a stuck elevator, therefore end up at milder severities. Cuts and redraws are recorded:
   - in `flights.csv` (`envelope_exit_s`, `attempts`);
   - in `metadata.json` (`envelope_cuts`, per type).

   This way the dataset contains departures from normal flight right up to the point of upset, but never the simulator blowing up.
3. **Windows.**
   - **Normal and pure abnormal: exactly 3 non-overlapping windows per flight**, drawn the same way, at random 120 s slots in [region start, end]. The region start is the onset for an abnormal flight. A normal flight gets a dummy onset from the same U(400, 700) s, so both classes come from the same part of the flight (same fuel burn, same time since start) and timing cannot give the label away.
   - **Equal per type:** every type gives 60 train, 15 validation and 15 test windows.
   - **Change:** one window per flight, placed so the onset falls at a random step between T/3 and 2T/3.
4. **Channels.**
   - All channels that change inside the training windows are kept.
   - **Dropped as bookkeeping:** simulation clock and frame counter, latitude and longitude, ECI/ECEF coordinates, distance from the start point, orbital elements, and Earth-model constants. These encode time or place, not aircraft behaviour, so they would only identify flights.
   - **Dropped as a simulator artefact:** any channel that is exactly constant in all normal training windows. In the full build there are two:
     - `moments/n-prop-lbsft`, the yawing moment from engine thrust. It is exactly 0 because the four simulated engines are identical, so any throttle fault would flag itself perfectly; no real recorder gives that.
     - `aero/coefficient/CDmach`, the drag rise with Mach. It is 0 below drag-divergence Mach, so it lights up only when a fault pushes the aircraft faster.
   - **Dropped as a duplicate.** If two channels have |corr| > 0.9999 on all training data, normal and abnormal, only the first is kept. This removed 112 channels: the same quantity in rad and deg, ft and m, kts and fps, `pct-full` versus `contents-lbs`, and so on.
     - Why: a per-channel likelihood would count each copy again.
     - Because the test includes abnormal data, a sensor is never merged with the true value it stops following during a fault.
   - **Circular angles:** the heading-like angles (`sensor/HDG`, `sel/hdg-deg`, `attitude/psi-*`, `heading-true-rad`, `flight-path/psi-gt-rad`, wind direction `atmosphere/psiw-rad`) are each replaced by a **sin/cos pair**, so that 359° → 0° is not a jump.
5. **Noise.** Measurement noise is added to `sensor/*`, exactly as in Cruise Lab.
6. **Normalisation.** z-score using the train-normal mean and std:

   std = max(std_normal, 0.1 · std_all_train, 1e-8)

   The floor only applies to channels that barely move in normal flight.

   After that, values are **clipped to ±10**. This touches 0.02 % of normal values and about 0.5 % of abnormal ones. Without clipping, a single extreme step costs thousands of nats in a Gaussian likelihood model and dominates the CUSUM threshold. Roll and aileron channels during turns are affected most; they are listed in `check_report.json`.

## Noisy variant (`qcd_data_noisy/`)

```
..\.venv\Scripts\python -m lab.qcd_dataset --assemble-only --cache qcd_data\cache --out qcd_data_noisy --noise-z 0.1
```

This variant is built from the same flights, so nothing is flown again.

**What it adds:** white noise with std **0.1 standardised units** to every channel that has no noise yet. That is every channel except `sensor/*`.

**Why:**
- Most of the channels are exact simulator values, so a predictor can forecast them almost perfectly.
- A fault's small deviation then costs a huge likelihood penalty, which inflates the information rate K_X.
- Real recorders quantise and add noise to everything.

**Effect in the first experiment:**
- K_X (all types) fell from 4,796 to 2,447.
- Raw CUSUM detection rose from 40 % to 50 %.
- Task-aware detection on throttle rose from 20 % to 47 %.

The naive baseline is unchanged at 0.29.

## Using it with the supervisor's code

The file names match `data.py`'s simulated loader. In `run_all.py` / `alpha_detection_sweep.py` / `rho_sweep.py`:

```python
DATA_DIR = Path(r"D:\IISc\flightgear\cruise_lab\qcd_data")   # or wherever qcd_data is copied
CONFIG["generate_data"]   = False    # do not overwrite with VAR data
CONFIG["data_dim"]        = 185      # metadata.json -> n_channels
CONFIG["sequence_length"] = 480
```

`data.py` needs `SIMULATED_ROOT` pointed at the same folder.

**Things that differ from the VAR data, and that the method should expect:**
- **Many anomaly types, one class.** "Abnormal" mixes 14 types, so p1 learns a mixture. To train on one type or one family, filter by `anomaly_type` or `group`.
- **Some faults are latent.** Examples: a stuck sensor while the true value is steady, or loss of effectiveness in straight flight. Labels are 1, but the data looks almost normal. `severity` and `anomaly_type` allow these to be analysed separately.
- **Dynamics are not stationary.** Normal flights include turns, step climbs and Mach changes.
- **Size.** D = 185 against 100, and T = 480 against 300/600.

## Validation (`check_report.json`)

`lab.qcd_check` verifies:
- shapes and finite values;
- normal labels are 0, and abnormal labels are mostly 1 (intermittent gaps are 0);
- labels match `change_point`;
- no flight appears in two splits;
- train_normal has mean 0 and std 1 (std below 1 only for channels whose std was floored);
- every anomaly type is present in every abnormal split.

It also runs a **naive baseline with no training**: the largest |z| over channels, averaged over the window, thresholded at the 95th percentile of test-normal windows. The per-type detection rates show which faults are easy and which are hard; see "Counts and baseline" below.

## Counts and baseline

Full build: **980 flights** of 1,500 s each, about 400 h simulated. Flying took about 15 min on 10 workers. Each window has **D = 185** features after removing duplicates and clipping (297 before) and T = 480 steps. Size on disk is about 2.6 GB including the cache.

| File | Flights | x shape | Abnormal-label share |
|---|---|---|---|
| train_normal | 280 | (840, 480, 185) | 0 |
| val_normal | 70 | (210, 480, 185) | 0 |
| test_normal | 70 | (210, 480, 185) | 0 |
| train_abnormal | 280 (20 per type) | (840, 480, 185): **60 per type** | 1.000 |
| val_abnormal | 70 (5 per type) | (210, 480, 185): 15 per type | 1.000 |
| test_abnormal | 70 (5 per type) | (210, 480, 185): 15 per type | 1.000 |
| test_change | 140 (10 per type) | (140, 480, 185) | 0.505 (change point at step 160–320) |

Normal and abnormal windows are balanced **1 : 1** in every split. The mix of hard conditions is identical in normal and abnormal flights of each split, for example 3/3/3/3/2 per condition in test.

- **Profiles:** 455 step, 105 ramp.
- **Envelope check:** 25 abnormal flights were cut short (12 stuck surface, 8 pitot blockage, 5 stuck sensor). 19 needed a milder redraw (541 flew at the first attempt, 12 at the 2nd, 5 at the 3rd, 1 each at the 4th and 5th). Mean severity per type stays between 0.44 and 0.53.
- **Channels:** 352 never change, 36 are bookkeeping, 2 are constant in normal flight, 112 are duplicates, and none contain NaN.

**All validation checks pass.** Train-normal max |mean| is 6e-5 and std is 0.999 to 1.001.

**Naive baseline** (no training, 5 % false alarms on test_normal): it catches **0.29** of abnormal windows.

| Type | Detected |
|---|---|
| surface_stuck | 0.80 |
| surface_runaway | 0.73 |
| pitot_blockage | 0.67 |
| throttle_thrust_loss | 0.53 |
| sensor_stuck | 0.40 |
| surface_bias | 0.27 |
| sensor_bias, sensor_noise | 0.20 |
| sensor_drift, sensor_spikes | 0.13 |
| throttle_lever_stuck | 0.07 |
| loss_of_effectiveness, oscillation, throttle_lever_bias | 0.00 |

Reading this:
- A single "largest |z|" rule catches the gross faults. Small biases, oscillation, loss of effectiveness and lever faults stay inside the normal range of every channel, so they need temporal or cross-channel structure. That is what a learned model has to find.
- **Weather check:** the baseline flags 26 % of hard-condition normal windows and 0 % of calm ones. For abnormal windows it flags 38 % hard and 27 % calm. The conditions are spread equally over both classes, so weather alone does not separate them.

An earlier build rotated the conditions per type, which gave every hard abnormal test flight turbulence. That made the baseline look better than it was; it is fixed. The 710-flight version 1 (uneven counts, type-limited anomaly durations) is kept in `qcd_data_v1/`.
