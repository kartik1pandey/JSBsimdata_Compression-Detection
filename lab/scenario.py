"""One flight = one seed. Everything that makes a NORMAL flight differ from the
next is drawn here; the anomaly (if any) is drawn separately in anomalies.py.

Cruise phase only: the aircraft starts trimmed at cruise altitude and speed and
cruises for 25-35 min with the autopilot engaged.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .aircraft import N_ENG, TANK_CAPACITY_LBS
from .autopilot import cas_for_mach

HARD_NORMALS = ('moderate_turbulence', 'strong_wind', 'hot_day', 'cold_day', 'heavy')


@dataclass
class Spec:
    flight_id: str
    seed: int
    duration_s: float = 1800.0
    # flight plan
    cruise_alt: float = 31000.0
    cruise_mach: float = 0.72
    start_cas: float = 265.0
    heading: float = 90.0
    ramp_fpm: float = 1800.0
    turns: tuple = ()                     # ((t_s, heading offset deg), ...)
    step_climb: Optional[tuple] = None    # (t_s, +ft)
    mach_change: Optional[tuple] = None   # (t_s, +dMach)
    # initial-state disturbance
    ic_theta_deg: float = 0.0
    ic_phi_deg: float = 0.0
    ic_beta_deg: float = 0.0
    ic_p: float = 0.0
    ic_q: float = 0.0
    ic_r: float = 0.0
    # controller
    gain_scale: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    # environment
    isa_dev_c: float = 0.0
    wind_kts: float = 0.0
    wind_from_deg: float = 0.0
    wind_veer_deg: float = 0.0
    turb_severity: int = 3
    # mass
    payload_lbs: float = 60000.0
    fuel_frac: tuple = (0.8, 0.8, 0.8, 0.8)
    center_fuel_lbs: float = 0.0
    hard_normal: Optional[str] = None
    anomaly: Optional[dict] = None
    extra: dict = field(default_factory=dict)

    # ---- the flight plan the autopilot is given (selected values) ----
    def sel_alt(self, t):
        if self.step_climb and t >= self.step_climb[0]:
            return self.cruise_alt + self.step_climb[1]
        return self.cruise_alt

    def sel_hdg(self, t):
        h = self.heading
        for ti, off in self.turns:
            if t >= ti:
                h = self.heading + off
        return h % 360.0

    def sel_mach(self, t):
        if self.mach_change and t >= self.mach_change[0]:
            return self.cruise_mach + self.mach_change[1]
        return self.cruise_mach

    def sel_cas(self, alt_ft, t):
        return cas_for_mach(self.sel_mach(t), alt_ft)

    def summary(self):
        d = asdict(self)
        d.pop('anomaly')
        d.pop('extra')
        d['gross_weight_note'] = 'see inertia/weight-lbs channel'
        return d


def draw_spec(flight_id, seed, hard_normal=None, duration=None):
    rng = np.random.default_rng(seed)
    s = Spec(flight_id=flight_id, seed=int(seed))
    s.duration_s = float(rng.uniform(1500.0, 2100.0))
    if duration is not None:                 # fixed length (QCD dataset); same rng stream otherwise
        s.duration_s = float(duration)
    # flight plan
    s.cruise_alt = float(rng.uniform(28000.0, 34000.0))
    s.cruise_mach = float(rng.uniform(0.705, 0.735))
    s.start_cas = cas_for_mach(s.cruise_mach, s.cruise_alt) + float(rng.uniform(-5.0, 5.0))
    s.heading = float(rng.uniform(0.0, 360.0))
    s.ramp_fpm = float(rng.uniform(1500.0, 2200.0))
    turns, off = [], 0.0
    for _ in range(int(rng.integers(0, 4))):
        off += float(rng.choice([-1, 1]) * rng.uniform(10.0, 60.0))
        turns.append((float(rng.uniform(120.0, s.duration_s - 120.0)), off))
    s.turns = tuple(sorted(turns))
    if rng.uniform() < 0.3:
        s.step_climb = (float(rng.uniform(300.0, 0.6 * s.duration_s)), 2000.0)
    if rng.uniform() < 0.3:
        s.mach_change = (float(rng.uniform(300.0, s.duration_s - 300.0)),
                         float(rng.choice([-1, 1]) * rng.uniform(0.005, 0.015)))
    # initial-state disturbance (dies out in ~5 min, so anomalies start after 5 min)
    s.ic_theta_deg = float(rng.normal(0.0, 1.5))
    s.ic_phi_deg = float(rng.normal(0.0, 4.0))
    s.ic_beta_deg = float(rng.normal(0.0, 0.7))
    s.ic_p = float(rng.normal(0.0, 0.015))
    s.ic_q = float(rng.normal(0.0, 0.01))
    s.ic_r = float(rng.normal(0.0, 0.005))
    # controller: each gain +-15 %
    s.gain_scale = tuple(float(x) for x in rng.uniform(0.85, 1.15, 5))
    # environment
    s.isa_dev_c = float(rng.normal(0.0, 5.0))
    s.wind_kts = float(rng.uniform(0.0, 50.0))
    s.wind_from_deg = float(rng.uniform(0.0, 360.0))
    s.wind_veer_deg = float(rng.uniform(-40.0, 40.0))
    s.turb_severity = int(rng.choice([0, 3], p=[0.25, 0.75]))
    # mass
    s.payload_lbs = float(rng.uniform(20000.0, 120000.0))
    s.fuel_frac = tuple(float(x) for x in rng.uniform(0.6, 0.9, N_ENG))
    s.center_fuel_lbs = float(rng.uniform(0.0, 0.5)) * TANK_CAPACITY_LBS
    # 'hard' normal conditions: still labelled normal
    if hard_normal:
        s.hard_normal = hard_normal
        if hard_normal == 'moderate_turbulence':
            s.turb_severity = 4
        elif hard_normal == 'strong_wind':
            s.wind_kts = float(rng.uniform(70.0, 100.0))
        elif hard_normal == 'hot_day':
            s.isa_dev_c = float(rng.uniform(15.0, 22.0))
        elif hard_normal == 'cold_day':
            s.isa_dev_c = float(rng.uniform(-22.0, -15.0))
        elif hard_normal == 'heavy':
            s.payload_lbs = float(rng.uniform(140000.0, 160000.0))
            s.fuel_frac = (0.95,) * N_ENG
            s.center_fuel_lbs = 0.8 * TANK_CAPACITY_LBS
        else:
            raise ValueError(hard_normal)
    return s
