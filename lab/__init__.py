"""cruise_lab: minimal JSBSim B747 cruise-phase data generator with
observation (before-autopilot) and control (after-autopilot) anomalies.

    lab/aircraft.py   JSBSim set-up: B747 copy with payload, wind, turbulence
    lab/autopilot.py  altitude / speed / heading / sideslip loops
    lab/scenario.py   per-flight random conditions (normal variability)
    lab/anomalies.py  anomaly catalogue, sampling, the injection operators
    lab/simulate.py   the closed loop + logging of every JSBSim property
    lab/generate.py   CLI: fly flights (+ normal twins), effects, manifest
"""
