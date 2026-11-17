# 5. Data files and scripts

## Files (`data/`)

| File | Contents |
|---|---|
| `flights/<id>.parquet` | One flight at 4 Hz. Columns: `t` (s since the start of cruise), `label` (0/1), then about 680 channels (float32; latitude/longitude in float64). |
| `flights/<id>_twin.parquet` | Anomalous flights only: the normal twin, same columns, from onset − 60 s to the end. |
| `meta/<id>.json` | Everything about one flight: `scenario`, `anomaly`, `injection_channels`, `changing` (channels that vary in this flight), `effects`, `twin_identical_before_onset`, `wall_s`. |
| `manifest.json` | One entry per flight: id, seed, kind (`normal` / `observation` / `control`), hard_normal, anomaly, duration, rows, injection channels. |
| `channels.json`, `channels.csv` | Every channel: name, subsystem, unit, meaning, status, `context` flag, `changing`, and the number of flights it changes in. |

Size: about 12 MB per flight and 10 MB per twin. The demo set is about 330 MB; the catalog set is about 1.3 GB.

Flight IDs:
- `N1xx`, `O1xx`, `C1xx`: demo normal, observation and control flights.
- `N2xx`, `O2xx`, `C2xx`: catalog.
- `N3xxx`, `O3xxx`, `C3xxx`: random.

## Column naming

| Prefix | Meaning |
|---|---|
| `sensor/ALT, CAS, VS, PITCH, ROLL, HDG, BETA, Q, P` | What the autopilot read (true value + observation anomaly) + measurement noise. Units: ft, kt, ft/min, deg, deg, deg, deg, deg/s, deg/s. |
| `sensor/TLA1..4` | Recorded thrust-lever angle, 0..1. |
| `ap/elevator-cmd, aileron-cmd, rudder-cmd, throttle-cmd` | Autopilot commands *before* any control anomaly. |
| `ap/target-alt-ft` | Autopilot internal (rate-limited) altitude target. |
| `sel/alt-ft, cas-kt, hdg-deg, mach` | Flight plan. |
| `derived/engine[i]/egt-degc` | EGT per engine (JSBSim's turbine law). |
| any JSBSim property | The true state, e.g. `position/h-sl-ft`, `velocities/vc-kts`, `fcs/elevator-pos-deg`, `propulsion/engine[1]/n1`, `aero/alpha-deg`. Engine `[0]` is engine 1. |

Useful pairs:

| Question | Compare |
|---|---|
| Is the sensor lying? | `sensor/ALT` vs `position/h-sl-ft` (likewise CAS / `velocities/vc-kts`, HDG / `attitude/psi-deg`, ...) |
| Did the command reach the surface? | `ap/elevator-cmd` vs `fcs/elevator-cmd-norm` (applied) vs `fcs/elevator-pos-deg` (moved) |
| Did the engine get what the lever says? | `sensor/TLA2` vs `fcs/throttle-cmd-norm[1]` vs `propulsion/engine[1]/n1` |

## Effect scores (`meta/<id>.json` → `effects`)

For each channel c of an anomalous flight x and its twin y:

```
scale_c  = max( std(y_c), 0.001·|mean(y_c)|, 0.05·std(x_c) )
score_c  = max over t ≥ onset of |x_c(t) − y_c(t)|  /  scale_c          (capped at 999)
first_s  = first t ≥ onset with |x_c − y_c| > 0.5·scale_c, minus onset
role     = injection point | strong (≥3) | moderate (≥1) | weak (≥0.25) | no visible effect
```

The floors keep nearly-constant channels from getting absurd scores. Examples are wind, which changes slightly with altitude, and the yawing moment from propulsion, which is zero when thrust is symmetric. Channels with a score below 0.05 are not stored.

The score uses the simulator twin, which a real detector would never have. It is meant for **understanding and verifying** the injected anomalies, not as a detector.

## Loading a flight in Python

```python
import json, pandas as pd
d = 'D:/IISc/flightgear/cruise_lab/data'
meta = json.load(open(f'{d}/meta/O101.json', encoding='utf-8'))
df   = pd.read_parquet(f'{d}/flights/O101.parquet')                 # all channels
tw   = pd.read_parquet(f'{d}/flights/O101_twin.parquet', columns=['t', 'sensor/CAS', 'velocities/vc-kts'])
a = meta['anomaly']; print(a['type'], a['variable'], a['onset_s'], a['end_s'])
win = df[df.label == 1]
```

## Scripts

| Command | What it does |
|---|---|
| `python -m lab.generate [--preset demo/catalog/random] [--n N] [--out DIR] [--workers K]` | Fly flights and twins, compute effects, write manifest, channels and `docs/03_channels.md` |
| `python viewer/server.py [--data DIR] [--port P]` | Start the viewer |

To add your own anomaly type:
1. Add an entry to `CATALOG` in `lab/anomalies.py`: group, vars, bounds, profiles, weight, max_dur, formula, text.
2. Add its branch in `Anomaly.sense()` (observation) or `Anomaly.actuate()` (control).

The generator, viewer and docs pick it up automatically. To add it to the demo set, add it to `DEMO` in `generate.py`.
