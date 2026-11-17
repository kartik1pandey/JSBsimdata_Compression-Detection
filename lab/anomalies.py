"""Anomaly catalogue, sampling and injection operators.

Where an anomaly enters the closed loop:

    true state --> SENSORS --> reading --> AUTOPILOT --> command --> ACTUATORS --> aircraft
                       ^                                                ^
              OBSERVATION anomaly                               CONTROL anomaly
              sense(): reading = S(true value)                  actuate(): applied = A(command)

An anomaly is a dict (stored as-is in the manifest):
    group, type, variable, engine, severity, magnitude, unit, profile, sign,
    onset_s, duration_s, end_s, bursts (intermittent only), freq_hz (oscillation)

Time envelope g(t) in [0, 1], zero outside [onset, end):
    step         g = 1
    ramp         g = (t - onset) / (end - onset)
    intermittent g = 1 during bursts (20-60 s on, 30-120 s off), else 0
"""
import numpy as np

# variables the autopilot reads: name -> (JSBSim property, unit of the magnitude,
#                                         factor magnitude -> property units)
SENSED = {
    'ALT':   ('position/h-sl-ft', 'ft', 1.0),
    'CAS':   ('velocities/vc-kts', 'kt', 1.0),
    'VS':    ('velocities/h-dot-fps', 'ft/min', 1.0 / 60.0),
    'PITCH': ('attitude/theta-deg', 'deg', 1.0),
    'ROLL':  ('attitude/phi-deg', 'deg', 1.0),
    'HDG':   ('attitude/psi-deg', 'deg', 1.0),
    'BETA':  ('aero/beta-deg', 'deg', 1.0),
    'Q':     ('velocities/q-rad_sec', 'deg/s', np.pi / 180.0),
    'P':     ('velocities/p-rad_sec', 'deg/s', np.pi / 180.0),
}
PROP_TO_SENSED = {v[0]: k for k, v in SENSED.items()}
SURFACES = ('elevator', 'aileron', 'rudder')

# magnitude bounds (lo, hi); magnitude = lo * (hi/lo)**severity  (log-uniform)
OBS_BIAS = {'ALT': (50, 400), 'CAS': (3, 15), 'VS': (150, 600), 'PITCH': (0.5, 3.0), 'ROLL': (0.5, 3.0),
            'HDG': (2, 15), 'BETA': (0.3, 2.0), 'Q': (0.11, 0.57), 'P': (0.11, 0.57)}
OBS_NOISE = {'ALT': (20, 150), 'CAS': (1, 6), 'VS': (100, 400), 'PITCH': (0.2, 1.5), 'ROLL': (0.2, 1.5),
             'HDG': (0.5, 4), 'BETA': (0.1, 0.8), 'Q': (0.06, 0.34), 'P': (0.06, 0.34)}

CATALOG = {
    # ---------------- observation: corrupt what the autopilot READS ----------------
    'sensor_bias': dict(
        group='observation', vars=tuple(SENSED), bounds=OBS_BIAS, profiles=('step', 'intermittent'),
        weight=0.25, max_dur=900,
        formula='reading = true + sign * m * g(t)',
        text='Constant offset on one sensor (calibration error, wrong reference). The autopilot '
             'drives the WRONG reading to the target, so the true aircraft settles off-target by m.'),
    'sensor_drift': dict(
        group='observation', vars=tuple(SENSED), bounds=OBS_BIAS, profiles=('ramp',),
        weight=0.25, max_dur=900,
        formula='reading = true + sign * 2m * (t - onset)/(end - onset)',
        text='Offset that grows linearly to 2m over the window (slow sensor degradation). The '
             'true aircraft drifts away while the reading stays on target.'),
    'sensor_stuck': dict(
        group='observation', vars=tuple(SENSED), bounds=None, profiles=('step',),
        weight=0.15, max_dur=180,
        formula='reading = true(onset)  (frozen)',
        text='The reading freezes at its value at onset. Latent while the true value does not '
             'change; severe for rate signals (VS, Q, P) because the loop loses its damping.'),
    'sensor_noise': dict(
        group='observation', vars=tuple(SENSED), bounds=OBS_NOISE, profiles=('step', 'intermittent'),
        weight=0.15, max_dur=600,
        formula='reading = true + N(0, (m/2)^2), new draw every 1/60 s',
        text='Extra white noise on the reading (loose connector, failing sensor). The autopilot '
             'reacts to the noise, so commands and surfaces get noisier.'),
    'sensor_spikes': dict(
        group='observation', vars=tuple(SENSED), bounds=OBS_NOISE, profiles=('step',),
        weight=0.10, max_dur=600,
        formula='reading = true + 3m * (+-1) during spikes of 0.25-1 s, ~4 per minute',
        text='Short large spikes (electrical interference, bus errors). Each spike kicks the loop briefly.'),
    'pitot_blockage': dict(
        group='observation', vars=('CAS',), bounds=None, profiles=('step',),
        weight=0.10, max_dur=300,
        formula='p_t frozen at onset; CAS_read = CAS(p_t* - p_s(h))',
        text='Pitot (total-pressure) port blocked, static port working: indicated CAS no longer '
             'depends on true speed and changes only with altitude. The autothrottle holds the '
             'frozen reading while the true speed wanders (Birgenair 301 / Aeroperu 603 mechanism).'),
    # ---------------- control: corrupt the COMMAND on its way to the aircraft ----------------
    'surface_bias': dict(
        group='control', vars=SURFACES,
        bounds={'elevator': (0.02, 0.10), 'aileron': (0.02, 0.15), 'rudder': (0.02, 0.15)},
        profiles=('step', 'ramp', 'intermittent'), weight=0.18, max_dur=900,
        formula='applied = command + sign * m * g(t)',
        text='Offset between the autopilot command and what the surface receives (actuator '
             'rigging / sensor offset). Integrators re-trim it away, so it shows as a '
             'command-vs-surface mismatch.'),
    'surface_stuck': dict(
        group='control', vars=SURFACES, bounds=None, profiles=('step',),
        weight=0.15, max_dur={'elevator': 120, 'aileron': 600, 'rudder': 600},
        formula='applied = command(onset)  (frozen)',
        text='Surface jams at its position at onset; the autopilot keeps commanding but nothing moves. '
             'Severe for the elevator (capped at 120 s); invisible for aileron/rudder in straight flight.'),
    'surface_loss_of_effectiveness': dict(
        group='control', vars=SURFACES,
        bounds={'elevator': (0.15, 0.5), 'aileron': (0.2, 0.7), 'rudder': (0.2, 0.7)},
        profiles=('step', 'ramp'), weight=0.12, max_dur=900,
        formula='applied = command * (1 - m * g(t))',
        text='Surface moves only a fraction of what is commanded (hydraulic loss, partial damage). '
             'Shows up only when the command is large (turns, manoeuvres).'),
    'surface_oscillation': dict(
        group='control', vars=SURFACES,
        bounds={'elevator': (0.01, 0.04), 'aileron': (0.01, 0.06), 'rudder': (0.01, 0.06)},
        profiles=('step', 'intermittent'), weight=0.09, max_dur=600,
        formula='applied = command + m * g(t) * sin(2 pi f (t - onset)),  f ~ U(0.3, 1.5) Hz',
        text='Oscillatory failure case (OFC): a sinusoid added to the surface by a faulty servo loop.'),
    'surface_runaway': dict(
        group='control', vars=SURFACES,
        bounds={'elevator': (0.03, 0.12), 'aileron': (0.05, 0.25), 'rudder': (0.05, 0.25)},
        profiles=('step',), weight=0.06, max_dur=300,
        formula='applied = command + sign * m * min(1, (t - onset)/30 s)',
        text='Surface runs away toward a hard-over, reaching the full offset m in 30 s.'),
    'throttle_thrust_loss': dict(
        group='control', vars=('engine',), bounds={'engine': (0.05, 0.40)},
        profiles=('step', 'ramp', 'intermittent'), weight=0.20, max_dur=900,
        formula='applied_e = command * (1 - m * g(t))  (one engine)',
        text='One engine receives less than the lever commands (fuel control fault). The other '
             'three engines push harder to hold speed; the lever (TLA) looks normal.'),
    'throttle_lever_stuck': dict(
        group='control', vars=('engine',), bounds=None, profiles=('step',),
        weight=0.10, max_dur=900,
        formula='applied_e = command(onset)  (one engine, frozen)',
        text='One thrust lever jams; that engine stops following the autothrottle.'),
    'throttle_lever_bias': dict(
        group='control', vars=('engine',), bounds={'engine': (0.03, 0.15)},
        profiles=('step', 'ramp'), weight=0.10, max_dur=900,
        formula='applied_e = command + sign * m * g(t)  (one engine)',
        text='Rigging offset: one engine gets more/less than its lever position. Engine-to-engine '
             'N1/fuel-flow split while the levers agree.'),
}
UNITS = {'sensor': lambda v: SENSED[v][1], 'surface': lambda v: 'command units (-1..1, ~20 deg per unit)',
         'engine': lambda v: 'fraction of throttle (0..1)'}
PROFILE_W = {'step': 0.5, 'ramp': 0.3, 'intermittent': 0.2}
MIN_ONSET_S = 300.0          # initial transient gone
TAIL_S = 120.0               # always leave >= 2 min after the anomaly to see recovery
DUR_MEDIAN_S, DUR_SIGMA = 300.0, 0.6


def unit_of(typ, var):
    if var in SENSED:
        return SENSED[var][1]
    if var in SURFACES:
        return UNITS['surface'](var)
    return UNITS['engine'](var)


def _choice(rng, d):
    k = list(d)
    p = np.array([d[x] for x in k], float)
    return k[int(rng.choice(len(k), p=p / p.sum()))]


def sample(rng, duration_s, group=None, typ=None, variable=None, severity=None, profile=None,
           onset=None, duration=None, cap=True):
    """Draw one anomaly: group -> type -> variable -> severity -> magnitude ->
    profile -> onset -> duration -> sign. Any level can be fixed; a fixed
    `duration` is still capped by the type's max_dur and the time left."""
    if typ is None:
        types = {k: c['weight'] for k, c in CATALOG.items() if group is None or c['group'] == group}
        typ = _choice(rng, types)
    c = CATALOG[typ]
    var = variable or c['vars'][int(rng.integers(len(c['vars'])))]
    sev = float(rng.uniform()) if severity is None else float(severity)
    mag = None
    if c['bounds']:
        lo, hi = c['bounds'][var]
        mag = float(lo * (hi / lo) ** sev)
    prof = profile or _choice(rng, {p: PROFILE_W[p] for p in c['profiles']})
    if onset is None:
        onset = float(rng.uniform(MIN_ONSET_S, duration_s - TAIL_S - 300.0))
    mx = c['max_dur'][var] if isinstance(c['max_dur'], dict) else c['max_dur']
    if duration is None:
        dur = float(np.exp(rng.normal(np.log(DUR_MEDIAN_S), DUR_SIGMA)))
    else:
        dur = float(duration)
    if prof == 'intermittent':
        dur = max(dur, 300.0)
    left = duration_s - TAIL_S - onset
    dur = float(np.clip(dur, 60.0, min(mx if prof != 'intermittent' else max(mx, 300.0), left) if cap else left))
    a = dict(group=c['group'], type=typ, variable=None if var == 'engine' else var,
             engine=int(rng.integers(1, 5)) if var == 'engine' else None,
             severity=round(sev, 3), magnitude=None if mag is None else round(mag, 5),
             unit=unit_of(typ, var), profile=prof, sign=1 if rng.uniform() < 0.5 else -1,
             onset_s=round(onset, 2), duration_s=round(dur, 2), end_s=round(onset + dur, 2))
    if prof == 'intermittent':
        b, t = [], onset
        while t < a['end_s']:
            on = float(rng.uniform(20.0, 60.0))
            b.append([round(t, 2), round(min(t + on, a['end_s']), 2)])
            t += on + float(rng.uniform(30.0, 120.0))
        a['bursts'] = b
    if typ == 'surface_oscillation':
        a['freq_hz'] = round(float(rng.uniform(0.3, 1.5)), 3)
    if typ == 'sensor_spikes':
        n = max(1, int(rng.poisson(4.0 * dur / 60.0)))
        a['spikes'] = [[round(float(s), 2), round(float(s + rng.uniform(0.25, 1.0)), 2),
                        int(rng.choice([-1, 1]))] for s in np.sort(rng.uniform(onset, a['end_s'], n))]
    a['formula'] = c['formula']
    a['description'] = c['text']
    return a


def envelope(a, t):
    """g(t) for scalar or array t."""
    t = np.asarray(t, float)
    inside = (t >= a['onset_s']) & (t < a['end_s'])
    if a['profile'] == 'ramp':
        span = a.get('ramp_s') or (a['end_s'] - a['onset_s'])      # optional: reach full size after ramp_s
        g = np.clip((t - a['onset_s']) / max(1.0, span), 0.0, 1.0)
    elif a['profile'] == 'intermittent':
        g = np.zeros_like(t)
        for x, y in a['bursts']:
            g = np.where((t >= x) & (t < y), 1.0, g)
    else:
        g = np.ones_like(t)
    out = np.where(inside, g, 0.0)
    return float(out) if out.ndim == 0 else out


def injection_channels(a):
    """Logged channels where the anomaly is injected (shown as 'injection point')."""
    if a is None:
        return []
    if a['group'] == 'observation':
        return [f"sensor/{a['variable']}"]
    if a['variable'] in SURFACES:
        return [f"fcs/{a['variable']}-cmd-norm"]
    e = a['engine'] - 1
    return [f'fcs/throttle-cmd-norm[{e}]', f"sensor/TLA{a['engine']}"]


# =================================================================== operators

class NoAnomaly:
    def sense(self, t, name, value, fdm):
        return value

    def actuate(self, t, u):
        return u

    def lever(self, e, u_cmd, u_app):
        """What the recorded thrust-lever angle of engine e shows."""
        return u_cmd


class Anomaly(NoAnomaly):
    def __init__(self, a, seed):
        self.a, self.m, self.sg = a, a['magnitude'] or 0.0, float(a['sign'])
        self.frozen = None
        self.nrng = np.random.default_rng(seed + 99)
        self.pt = None

    def g(self, t):
        return envelope(self.a, t)

    # ---- observation ------------------------------------------------------
    def sense(self, t, name, value, fdm):
        a = self.a
        if a['group'] != 'observation' or name != a['variable']:
            return value
        if not (a['onset_s'] <= t < a['end_s']):
            self.frozen = None
            return value
        if a['type'] == 'pitot_blockage':
            return self._pitot(value, fdm['position/h-sl-ft'])
        g = self.g(t)
        if g <= 0:
            self.frozen = None
            return value
        m = self.m * SENSED[name][2]                      # into the property's units
        typ = a['type']
        if typ == 'sensor_bias':
            return value + self.sg * m * g
        if typ == 'sensor_drift':
            return value + self.sg * 2.0 * m * g
        if typ == 'sensor_stuck':
            if self.frozen is None:
                self.frozen = value
            return self.frozen
        if typ == 'sensor_noise':
            return value + self.nrng.normal(0.0, 0.5 * m)
        if typ == 'sensor_spikes':
            for x, y, s in a['spikes']:
                if x <= t < y:
                    return value + s * 3.0 * m
        return value

    @staticmethod
    def _ps(h):
        return 2116.217 * ((518.67 - 3.566 * min(h, 36089.0) / 1000.0) / 518.67) ** 5.2559

    def _pitot(self, cas, h):
        P0, A0 = 2116.217, 661.4786
        if self.pt is None:
            self.pt = self._ps(h) + P0 * ((1.0 + 0.2 * (cas / A0) ** 2) ** 3.5 - 1.0)
        qc = max(self.pt - self._ps(h), 0.0)
        return max(A0 * np.sqrt(5.0 * ((qc / P0 + 1.0) ** (2.0 / 7.0) - 1.0)), 30.0)

    # ---- control ----------------------------------------------------------
    def actuate(self, t, u):
        a = self.a
        if a['group'] != 'control':
            return u
        g = self.g(t)
        k = a['variable'] if a['variable'] else f"thr{a['engine'] - 1}"
        if g <= 0:
            self.frozen = None
            return u
        c, typ = u[k], a['type']
        if typ in ('surface_bias', 'throttle_lever_bias'):
            u[k] = c + self.sg * self.m * g
        elif typ in ('surface_stuck', 'throttle_lever_stuck'):
            if self.frozen is None:
                self.frozen = c
            u[k] = self.frozen
        elif typ in ('surface_loss_of_effectiveness', 'throttle_thrust_loss'):
            u[k] = c * (1.0 - self.m * g)
        elif typ == 'surface_oscillation':
            u[k] = c + self.m * g * np.sin(2 * np.pi * a['freq_hz'] * (t - a['onset_s']))
        elif typ == 'surface_runaway':
            u[k] = c + self.sg * self.m * min(1.0, (t - a['onset_s']) / 30.0)
        return u

    def lever(self, e, u_cmd, u_app):
        a = self.a
        if a['type'] == 'throttle_lever_stuck' and a['engine'] - 1 == e:
            return u_app                                  # the lever itself is jammed
        return u_cmd


def make(a, seed):
    return NoAnomaly() if a is None else Anomaly(a, seed)
