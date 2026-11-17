# 1. How a flight is generated

A flight is fully defined by its **seed** and its **anomaly** (or none). Running it again gives an identical result, bit for bit.

```
seed ──► scenario (scenario.py) ──► closed-loop flight (simulate.py, 60 Hz) ──► 4 Hz log
                         anomaly (anomalies.py) ──┘                              │
                                                                                 ▼
                                          sensor noise on sensor/* ──► flights/<id>.parquet
             same seed, no anomaly ──► normal twin ──► flights/<id>_twin.parquet ──► effect scores
```

## 1.1 Scenario: why no two normal flights are alike

`scenario.draw_spec(seed)` draws the conditions below. They are all *normal* variability, present in every flight.

| Level | What varies | Distribution |
|---|---|---|
| Flight plan | cruise altitude | U(28 000, 34 000) ft |
| | cruise Mach | U(0.705, 0.735) |
| | heading | U(0, 360)° |
| | cruise length | U(1500, 2100) s = 25–35 min |
| | route turns | 0–3 turns, each ±(10–60)°, at random times |
| | step climb | +2000 ft with probability 0.3 |
| | Mach change | ±(0.005–0.015) with probability 0.3 |
| Initial state | after trim the aircraft is disturbed | Δpitch N(0, 1.5²)°, roll N(0, 4²)°, sideslip N(0, 0.7²)°, body rates p, q, r |
| | initial speed | trimmed CAS ± 5 kt |
| Controller | five autopilot gains (altitude P and I; throttle P, I, D) | each × U(0.85, 1.15) |
| Environment | ISA temperature deviation | N(0, 5²) °C |
| | wind | U(0, 50) kt, from U(0, 360)°; grows with altitude and veers by ±40° |
| | turbulence (JSBSim MIL-SPEC Dryden, seeded) | light 75 %, calm 25 % |
| Mass | payload | U(20 000, 120 000) lb |
| | fuel | each wing tank 60–90 % of 25 000 lb; centre tank 0–50 % |
| Hard normals | demo/catalog presets include some | moderate turbulence, strong wind 70–100 kt, hot or cold day (±15–22 °C), heavy weight. All labelled **normal**. |

The initial disturbance produces a transient in the first 1–3 minutes, mostly in pitch, VS and throttle. This is part of every flight, and anomalies never start before 300 s.

## 1.2 The closed loop (one 1/60 s step, `simulate.fly`)

1. **Flight plan.** The selected altitude, heading and Mach come from the scenario. The selected CAS is computed from the selected Mach at the current altitude. These are logged as `sel/*`.
2. **Sensors.** The autopilot reads 9 values: ALT, CAS, VS, PITCH, ROLL, HDG, BETA (sideslip), Q and P (pitch and roll rate). Every read goes through the anomaly's `sense()` function. **An observation anomaly changes the reading here.**
3. **Autopilot** (`autopilot.py`). It computes elevator, aileron, rudder and one throttle command for all 4 engines. These are logged as `ap/*`.
4. **Actuators.** The commands go through the anomaly's `actuate()` function. **A control anomaly changes the command here.** The result is clipped to the travel limits (surfaces −1…1, throttles 0…1) and written to JSBSim's `fcs/*-cmd-norm` properties.
5. **JSBSim** advances the physics by one step: 6-DOF rigid body, aerodynamics, 4 turbofan engines, fuel system, atmosphere with wind and turbulence.

### Autopilot equations (all inputs are the *sensed* values)

- **Altitude to elevator**
  - The selected altitude is ramp-limited to R ft/min, with R ~ U(1500, 2200).
  - VS command = clip(2.0 · (h_target − h), ±2000) ft/min.
  - Pitch command = PI on the VS error: integral gain 5×10⁻⁵, proportional gain 0.0022, limits −8…12°.
  - Elevator = clip(sign · [0.060 · (θ_cmd − θ) − 1.5 · q], ±1).
- **Speed to throttle**
  - PID on the CAS error: P 0.020, I 0.002, D 0.020.
  - Anti-windup, slew limit 0.055 per second, range 0.18…1.
  - The same command goes to all four engines.
- **Heading to aileron**
  - Bank command = clip(1.2 · heading error, ±25°).
  - Aileron = clip(0.020 · (φ_cmd − φ) − 0.55 · p, ±1).
- **Sideslip to rudder**
  - Rudder = clip(−(0.03 · β + 0.01 · ∫β dt), ±0.4).
  - This acts on top of the B747 model's own yaw damper.

The integrators explain why some anomalies are "absorbed":
- A constant elevator offset is re-trimmed away by the pitch integrator.
- A constant CAS-sensor bias is *not* removed. The loop drives the wrong reading to the target, so the true speed settles off by the bias.

## 1.3 Logging (4 Hz)

After each logged physics step:
- **Every readable JSBSim property is logged**: about 680, excluding landing-gear and ground-reaction properties, which are meaningless in cruise and partly hold uninitialised memory. The first array element is written with an explicit index, e.g. `propulsion/engine[0]/n1`.
- **The sensor readings** `sensor/ALT … sensor/P` are logged in the units shown in the viewer: VS in ft/min, P and Q in deg/s. They are what the autopilot actually used, *including* any observation anomaly.
- **The recorded thrust-lever angles** `sensor/TLA1…4` are logged. A lever shows a different value only if that lever itself is stuck.
- **The autopilot commands and targets** `ap/*` are logged, as is the flight plan `sel/*`.
- **EGT per engine** is logged as `derived/engine[i]/egt-degc`. It uses JSBSim's internal turbine law (TAT + 363.1 + 357.1 × throttle position), which JSBSim does not expose as a property.
- **`label`** is 1 inside the anomaly window. For an intermittent profile it is 1 only during the bursts.

Then **measurement noise** is added to the `sensor/*` channels only:

| Channel | Noise sd |
|---|---|
| ALT | 10 ft |
| CAS | 0.5 kt |
| VS | 30 ft/min |
| PITCH, ROLL, BETA | 0.05° |
| HDG | 0.1° |
| Q, P | 0.02 °/s |
| TLA | 0.002 |

The JSBSim properties are the **true simulated state**, without noise. The autopilot itself reads the noise-free value plus the anomaly; the noise is added only to the log.

## 1.4 The normal twin

For every anomalous flight the same scenario is flown a second time with **no anomaly**. The twin has the same seed, the same turbulence gusts and the same sensor-noise draw. Up to the onset the two logs are **identical sample for sample**: the generator checks this and records `twin_identical_before_onset: true`. All 16 demo flights pass. From the onset on, every difference between flight and twin is caused by the anomaly and the closed-loop reaction to it.

The twin is stored from 60 s before onset to the end of the flight. It is used for:
- the overlay in the viewer;
- the **effect score** of each channel. This is the largest |flight − twin| after onset, divided by the channel's normal spread, and it gives each channel its role (strong, moderate or weak response, or no effect). See `05_data_format.md`.

## 1.5 Limitations (what this is not)

- JSBSim surfaces have no actuator dynamics. The command reaches the surface through the B747 model's flight-control channel, about 20° per unit command for the elevator and aileron, plus trim.
- EGT comes from JSBSim's simple law and is not a gas-path model. N1, N2, fuel flow and thrust are JSBSim turbine-model outputs.
- The noise levels and magnitude limits are engineering choices from exploratory runs. They are not fitted to a real flight-data recorder or to fleet data.
- Wind is steady with altitude-dependent strength and direction, plus Dryden turbulence. There is no weather and there are no fronts.
