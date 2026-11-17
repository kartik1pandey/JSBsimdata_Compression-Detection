# 4. Using the viewer

```
..\.venv\Scripts\python viewer\server.py          # http://localhost:8050
..\.venv\Scripts\python viewer\server.py --data D:\other\data --port 8051
```

You can open a given flight directly, e.g. `http://localhost:8050/?flight=O100`. The page remembers the last flight you looked at.

## Top bar

- **Show.** Filters the flight list to all flights, normal flights, observation anomalies or control anomalies.
- **Flight.** The drop-down of all generated flights, grouped by kind. Each entry reads `id · group · type · target · profile`, e.g. `O101 · observation · sensor_drift · CAS · ramp`.
- **Overlay normal twin.** Draws the same flight flown without the anomaly, in orange, on every chart. It starts 60 s before onset; before onset it is identical to the flight.
- **List constant channels too.** By default the channel lists show only the ~329 channels that change in flight. Tick this to list all ~680.
- **Docs.** Opens these documents.

## Cards

1. **Anomaly card.** Shows:
   - group (observation or control) and type;
   - target variable or engine;
   - magnitude with unit, sign and severity;
   - profile;
   - window (onset → end, duration), plus bursts, spikes or frequency where they apply;
   - the injection formula and a plain-language description.

   *Injected at* lists the channel(s) where the anomaly enters. Click one to plot it in panel 1.

2. **Flight conditions.** The scenario of this seed: cruise altitude, Mach, heading, route turns, step climb, Mach change, wind, turbulence, ISA deviation, payload and the autopilot gain factors.

3. **Most affected channels.** All channels ranked by **effect score**: the largest departure from the normal twin after onset, divided by the channel's normal spread.
   - Each row has buttons **1–4** that put the channel in that chart panel.
   - Environment and bookkeeping channels are hidden unless you tick the box. These include wind, air density at the new altitude, latitude/longitude and fuel bookkeeping.

## The four chart panels

Each panel has:
- a **filter box** (e.g. `n1`, `elevator`, `alt`, `fuel`), which matches name and meaning;
- a **channel drop-down**, grouped by subsystem. Subsystems include sensors, autopilot commands, flight plan, flight controls, Engine 1–4, attitude, velocities, aerodynamics, forces, moments and more.
  - `◆` marks the injection point of the current flight's anomaly.
  - `●` marks channels with a strong response.
- **true value** (sensor channels only). Overlays the true simulated value behind the reading, in violet dashed. For an observation anomaly this shows the gap between what the autopilot believed and reality.
- **other engines** (engine channels only). Overlays the same quantity for the other three engines, so engine-to-engine splits are visible.
- a **role line** saying what the anomaly does to this channel:
  - **◆ Injection point**: the anomaly is applied here (type, window, formula).
  - **Strong / moderate / weak response**: score ≥ 3 / ≥ 1 / ≥ 0.25, with how many times the normal spread it departs by and how many seconds after onset it first becomes visible (departure > 0.5 × normal spread).
  - **No visible effect**: the departure is below 0.05 × the normal spread.
  - Normal flights say that there is no anomaly.
- the **chart**:
  - this flight in blue;
  - the normal twin in orange;
  - the true value or other engines when ticked;
  - the anomaly window shaded, with each burst shaded separately for intermittent anomalies;
  - dotted lines at onset and end.

  Hovering shows all values at that time. Zooming one chart (drag a box) zooms all four in time. Double-click resets the zoom.

When you select a flight, the four panels are pre-filled with channels that tell its story:

| Target | Default panels |
|---|---|
| ALT | ALT reading (+ true), VS reading, pitch, elevator position |
| CAS | CAS reading (+ true), Mach, autothrottle command, engine 1 N1 |
| HDG | HDG reading (+ true), roll, aileron command, sideslip |
| elevator | autopilot command, applied command (injection point), surface position, altitude |
| engine e | lever TLA_e, engine e N1, engine e fuel flow, rudder position |

Change any panel freely.

## Reading the charts: what to look for

- **Observation anomaly.** Tick *true value* on the injected sensor. The reading stays where the autopilot wants it while the true value moves away. Then look at the commands (`ap/*`) and surfaces, which react.
- **Control anomaly.** Plot `ap/<surface>-cmd` (what the autopilot asked for) next to `fcs/<surface>-cmd-norm` (what was applied) and `fcs/<surface>-pos-deg` (what moved). For engines, tick *other engines* on N1 or fuel flow.
- **Normal flights.** Look at the turns, step climbs and Mach changes, and at the turbulence level. All of this is normal variability that a detector must tolerate.
