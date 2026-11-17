# 2. Anomalies: what, where, how, how long

Code: `lab/anomalies.py`: `CATALOG`, the `sample()` function, and the `Anomaly.sense()` / `Anomaly.actuate()` operators.

## 2.1 Two groups, split by where they enter the loop

| Group | Enters | Hook | What is corrupted | What the autopilot does |
|---|---|---|---|---|
| **observation** | *before* the autopilot | `sense(t, variable, true_value)` | one of the 9 values the autopilot reads (ALT, CAS, VS, PITCH, ROLL, HDG, BETA, Q, P) | Acts correctly on wrong information. The **true aircraft really deviates** while the faulty sensor often looks fine. |
| **control** | *after* the autopilot | `actuate(t, commands)` | one command on its way to a surface (elevator, aileron, rudder) or to one engine's throttle | Sees the aircraft respond wrongly and compensates. The tell-tale sign is that the **command and what was applied or moved disagree**. |

Surfaces and throttle are deliberately **different families**. A surface fault changes a moment on the airframe. A throttle fault changes one engine's thrust, which also causes yaw through thrust asymmetry.

## 2.2 How one anomaly is drawn (`sample()`)

Each level can also be fixed by hand: the presets fix type, variable, severity and profile.

1. **Group, then type.** Drawn with the type weights in the tables below.
2. **Variable.** Uniform over the type's variables. Engine faults pick one engine, uniform over 1–4.
3. **Severity** s ~ U(0, 1). The demo and catalog presets use s = 0.7.
4. **Magnitude.** m = lo · (hi/lo)^s, which is log-uniform between the variable's limits (tables below). Small, hard-to-see faults and large, obvious ones are equally likely per decade.
5. **Sign.** ±1 with equal probability. It applies to bias, drift and runaway.
6. **Time profile**, chosen among the profiles the type allows, with weights step 0.5, ramp 0.3, intermittent 0.2.
7. **Onset** t₀ ~ U(300 s, T − 420 s), where T is the flight length. The anomaly never starts in the first 5 min, so the initial transient has died out.
8. **Duration.** ln d ~ N(ln 300 s, 0.6²): the median is 5 min and 90 % of draws fall between 2 and 13 min. It is clipped to [60 s, max duration of the type] and to the time left minus 2 min, so the recovery is always recorded. An intermittent anomaly lasts at least 300 s.

The time envelope g(t) is 0 outside [t₀, t₁) with t₁ = t₀ + d. Inside the window:

| Profile | g(t) | Meaning |
|---|---|---|
| step | 1 | switches on at t₀, off at t₁ |
| ramp | (t − t₀)/(t₁ − t₀) | grows linearly from 0 to 1 |
| intermittent | 1 during bursts, 0 between | bursts of U(20, 60) s separated by gaps of U(30, 120) s, starting at t₀ |

**Labels.** `label = 1` for t in [t₀, t₁). For intermittent anomalies, `label = 1` only during bursts.

## 2.3 Observation anomalies

The operator replaces the true value y(t) the autopilot would read with ỹ(t). The recorded `sensor/<VAR>` shows ỹ plus measurement noise, as a real recorder does: it stores the same sensor bus.

| Type | Injection (inside the window) | Profiles | Max duration | Weight | What it represents |
|---|---|---|---|---|---|
| `sensor_bias` | ỹ = y + sign · m · g(t) | step, intermittent | 900 s | 0.25 | calibration error, wrong reference |
| `sensor_drift` | ỹ = y + sign · 2m · (t − t₀)/(t₁ − t₀) (reaches 2m at the end) | ramp | 900 s | 0.25 | slow sensor degradation |
| `sensor_stuck` | ỹ = y(t₀), frozen | step | 180 s | 0.15 | frozen sensor / stale data |
| `sensor_noise` | ỹ = y + N(0, (m/2)²), a new draw every 1/60 s | step, intermittent | 600 s | 0.15 | loose connector, failing sensor |
| `sensor_spikes` | ỹ = y ± 3m during spikes of 0.25–1 s, about 4 per minute | step | 600 s | 0.10 | electrical interference, bus errors |
| `pitot_blockage` | CAS computed from a frozen total pressure (below) | step | 300 s | 0.10 | iced or blocked pitot tube |

**Magnitude limits (lo – hi) per variable.** Bias and drift use the first column; noise and spikes use the second.

| Variable | Unit | bias / drift | noise / spikes |
|---|---|---|---|
| ALT | ft | 50 – 400 | 20 – 150 |
| CAS | kt | 3 – 15 | 1 – 6 |
| VS | ft/min | 150 – 600 | 100 – 400 |
| PITCH, ROLL | deg | 0.5 – 3 | 0.2 – 1.5 |
| HDG | deg | 2 – 15 | 0.5 – 4 |
| BETA | deg | 0.3 – 2 | 0.1 – 0.8 |
| Q, P | deg/s | 0.11 – 0.57 | 0.06 – 0.34 |

**Pitot blockage, step by step.** At t₀ the total pressure is frozen: p_t* = p_s(h₀) + q_c(CAS₀). The ISA static pressure is p_s(h) = 2116.217 · ((518.67 − 3.566·h/1000)/518.67)^5.2559 lb/ft². Afterwards the indicated speed is

CAS_read = 661.48 · √(5 · [((p_t* − p_s(h))/2116.217 + 1)^(2/7) − 1]) kt.

The reading no longer depends on the true speed and changes only with altitude. The autothrottle holds the frozen reading while the true speed wanders. This is the mechanism of the Birgenair 301 and Aeroperú 603 accidents.

**What you will typically see:**
- **ALT, CAS or HDG bias/drift.** The recorded sensor stays on target. The true altitude, speed or heading moves by the bias, or by up to 2m for drift. Use *true value* in the viewer to see it.
- **Stuck VS, Q or P.** The rate signal that damps the loop is frozen, so the aircraft can enter a pitch or roll oscillation. Stuck VS is the most severe: ±1000 ft altitude oscillation.
- **Stuck ALT in level flight.** Latent: the frozen value equals the true one until the aircraft climbs.
- **Noise or spikes.** The autopilot reacts to them, so commands and surfaces become noisy or kicked, while the true state changes only a little.
- **BETA bias.** The rudder loop holds a false sideslip, so the true aircraft flies with a real sideslip and a rudder offset.

## 2.4 Control anomalies

The operator replaces the command u (from `ap/*`) with the applied value u_app. That value is written to JSBSim, where it is visible as `fcs/<surface>-cmd-norm` or `fcs/throttle-cmd-norm[i]`. The surface or engine then responds physically.

**Surfaces** (elevator, aileron, rudder; command units −1…1, about 20° per unit for elevator and aileron):

| Type | Injection | Profiles | Max duration | Weight | What it represents |
|---|---|---|---|---|---|
| `surface_bias` | u_app = u + sign · m · g(t) | step, ramp, intermittent | 900 s | 0.18 | actuator rigging / position-sensor offset |
| `surface_stuck` | u_app = u(t₀), frozen | step | elevator 120 s, aileron/rudder 600 s | 0.15 | jammed surface |
| `surface_loss_of_effectiveness` | u_app = u · (1 − m · g(t)) | step, ramp | 900 s | 0.12 | hydraulic loss, partial damage |
| `surface_oscillation` | u_app = u + m · g(t) · sin(2πf(t − t₀)), f ~ U(0.3, 1.5) Hz | step, intermittent | 600 s | 0.09 | oscillatory failure case (faulty servo loop) |
| `surface_runaway` | u_app = u + sign · m · min(1, (t − t₀)/30 s) | step | 300 s | 0.06 | runaway toward hard-over |

| Limits (lo – hi) | elevator | aileron | rudder |
|---|---|---|---|
| bias | 0.02 – 0.10 | 0.02 – 0.15 | 0.02 – 0.15 |
| loss of effectiveness (fraction lost) | 0.15 – 0.5 | 0.2 – 0.7 | 0.2 – 0.7 |
| oscillation amplitude | 0.01 – 0.04 | 0.01 – 0.06 | 0.01 – 0.06 |
| runaway | 0.03 – 0.12 | 0.05 – 0.25 | 0.05 – 0.25 |

**Throttle**, one engine e only. The other three engines keep receiving the autothrottle command.

| Type | Injection | Recorded lever `sensor/TLA<e>` | Limits | Profiles | Weight |
|---|---|---|---|---|---|
| `throttle_thrust_loss` | u_app,e = u · (1 − m · g(t)) | normal (the lever is fine, the engine gets less) | 0.05 – 0.40 | step, ramp, intermittent | 0.20 |
| `throttle_lever_stuck` | u_app,e = u(t₀), frozen | frozen | — | step | 0.10 |
| `throttle_lever_bias` | u_app,e = u + sign · m · g(t) | normal (rigging offset) | 0.03 – 0.15 | step, ramp | 0.10 |

All throttle faults have a maximum duration of 900 s.

**What you will typically see:**
- **Elevator bias.** The pitch integrator shifts `ap/elevator-cmd` by the opposite amount. The surface, pitch and altitude barely change, so the signature is command vs applied mismatch.
- **Stuck elevator.** Severe pitch and altitude excursion. Its duration is capped at 120 s for that reason.
- **Loss of effectiveness.** Visible only when the command is large (turns, step climbs). It is often almost invisible in straight cruise.
- **Rudder or aileron runaway.** The loops counter it. The aircraft rolls or yaws during the 30 s ramp, and with aileron a small heading offset remains.
- **Throttle faults.** The engine-to-engine N1, fuel-flow and EGT split between engine e and the others is the clearest sign. You also get a yawing moment from thrust asymmetry (`moments/n-prop-lbsft`) and a rudder correction. Use *other engines* in the viewer.

## 2.5 The demo set as generated (`--preset demo`, severity 0.7)

| Flight | Group | Type | Target | Magnitude | Profile | Onset → end [s] | Duration | Strong / moderate responses* |
|---|---|---|---|---|---|---|---|---|
| N100 | normal | ordinary | | | | | 26.3 min flight | |
| N101 | normal | moderate turbulence | | | | | 33.5 min flight | |
| N102 | normal | strong wind | | | | | 34.4 min flight | |
| N103 | normal | heavy | | | | | 25.2 min flight | |
| O100 | observation | sensor_bias | ALT | +214 ft | step | 900 → 1436 | 536 s | 81 / 60 |
| O101 | observation | sensor_drift | CAS | −9.26 kt (reaches −18.5 kt) | ramp | 717 → 1065 | 347 s | 91 / 20 |
| O102 | observation | sensor_stuck | VS | value at onset | step | 683 → 863 | 180 s | 145 / 4 |
| O103 | observation | sensor_noise | PITCH | m = 0.82° (noise sd 0.41°) | step | 636 → 790 | 153 s | 17 / 14 |
| O104 | observation | sensor_spikes | HDG | m = 2.14° (spikes ±6.4°) | step | 678 → 856 | 178 s | 33 / 40 |
| O105 | observation | pitot_blockage | CAS | frozen total pressure | step | 887 → 1187 | 300 s | 117 / 81 |
| O106 | observation | sensor_bias | BETA | −1.13° | intermittent | 965 → 1265 | 300 s | 80 / 40 |
| O107 | observation | sensor_drift | HDG | −8.2° (reaches −16.4°) | ramp | 576 → 1094 | 517 s | 82 / 59 |
| C100 | control | surface_bias | elevator | −0.062 | step | 501 → 612 | 111 s | 16 / 45 |
| C101 | control | surface_stuck | elevator | value at onset | step | 502 → 622 | 120 s | 76 / 53 |
| C102 | control | surface_loss_of_effectiveness | elevator | 35 % lost | step | 1495 → 1805 | 310 s | 0 / 10 |
| C103 | control | surface_oscillation | aileron | 0.035 amplitude | step | 432 → 846 | 414 s | 6 / 12 |
| C104 | control | surface_runaway | rudder | +0.154 | step | 405 → 705 | 300 s | 73 / 59 |
| C105 | control | throttle_thrust_loss | engine 2 | 21 % lost | intermittent | 1383 → 1892 | 508 s | 34 / 57 |
| C106 | control | throttle_lever_stuck | engine 1 | value at onset | step | 342 → 504 | 161 s | 1 / 2 |
| C107 | control | throttle_lever_bias | engine 3 | +0.093 | step | 565 → 712 | 147 s | 14 / 30 |

\* The last column counts channels with an effect score ≥ 3 (strong) or ≥ 1 (moderate), excluding environment and bookkeeping channels (see `05_data_format.md`).

Weak ones are realistic, not errors:
- C102: loss of effectiveness in straight flight, where the elevator command is small.
- C106: a stuck lever while the autothrottle hardly moves.

The exact values come from the seed. They are also shown in the viewer's anomaly card and stored in `data/meta/<id>.json`.
