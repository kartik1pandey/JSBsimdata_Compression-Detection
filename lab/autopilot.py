"""The autopilot: the same control laws as synth/core.py (v2 gains) plus the
sideslip loop, written out without the notebook scaffolding.

Every reading goes through `s[...]`, the SENSED view of the aircraft -- that
is where observation anomalies enter. Every output is a COMMAND that the
actuator layer may corrupt afterwards -- that is where control anomalies enter.

    altitude  : h_sel -> ramped target -> VS command -> pitch (PI) -> elevator
    speed     : CAS_sel -> throttle PID (anti-windup, slew-limited), all 4 engines
    heading   : HDG_sel -> bank command -> aileron (P + roll-rate damping)
    sideslip  : beta -> rudder (PI), on top of the B747 model's own yaw damper
"""
import numpy as np


def clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def cas_for_mach(mach, alt_ft):
    """Calibrated airspeed (kt) for a Mach number at a pressure altitude (ISA)."""
    T0, P0, a0 = 518.67, 2116.217, 661.4786
    if alt_ft <= 36089.0:
        P = P0 * ((T0 - 3.566 * alt_ft / 1000.0) / T0) ** 5.2559
    else:
        P = P0 * 0.223360 * np.exp(-(alt_ft - 36089.0) / 20805.8)
    qc = P * ((1.0 + 0.2 * mach ** 2) ** 3.5 - 1.0)
    return float(a0 * np.sqrt(5.0 * ((qc / P0 + 1.0) ** (2.0 / 7.0) - 1.0)))


class Autopilot:
    def __init__(self, elev_sign, dt, gain_scale=(1, 1, 1, 1, 1), ramp_fpm=1800.0):
        g = gain_scale
        self.es, self.dt = elev_sign, dt
        self.kp_alt, self.ki_alt = 2.0 * g[0], 5e-5 * g[1]
        self.kp_thr, self.ki_thr, self.kd_thr = 0.020 * g[2], 0.002 * g[3], 0.020 * g[4]
        self.thr_slew, self.thr_min = 0.055, 0.18
        self.ramp_fpm = ramp_fpm
        self.vs_limit_fpm = max(2000.0, ramp_fpm + 200.0)
        self.theta_i = None
        self.thr = self.thr_i = 0.5
        self.cas_prev = None
        self.tgt_alt = None
        self.beta_i = 0.0
        self.cmd = {'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0, 'throttle': 0.5}

    def step(self, s, sel_alt, sel_cas, sel_hdg):
        h = s['position/h-sl-ft']
        cas = s['velocities/vc-kts']
        vs = s['velocities/h-dot-fps'] * 60.0
        theta, phi, psi = s['attitude/theta-deg'], s['attitude/phi-deg'], s['attitude/psi-deg']
        q, p = s['velocities/q-rad_sec'], s['velocities/p-rad_sec']
        beta = s['aero/beta-deg']
        if self.theta_i is None:
            self.theta_i, self.tgt_alt = theta, h

        # altitude -> pitch -> elevator
        ramp = self.ramp_fpm / 60.0 * self.dt
        self.tgt_alt = clip(sel_alt, self.tgt_alt - ramp, self.tgt_alt + ramp)
        vs_cmd = clip((self.tgt_alt - h) * self.kp_alt, -self.vs_limit_fpm, self.vs_limit_fpm)
        self.theta_i = clip(self.theta_i + (vs_cmd - vs) * self.ki_alt, -8.0, 12.0)
        theta_cmd = clip(self.theta_i + (vs_cmd - vs) * 0.0022, -8.0, 12.0)
        elev = clip(((theta_cmd - theta) * 0.060 - q * 1.5) * self.es, -1.0, 1.0)

        # speed -> throttle (PID, anti-windup, slew limit)
        e = cas - sel_cas
        rate = 0.0 if self.cas_prev is None else (cas - self.cas_prev) / self.dt
        self.cas_prev = cas
        raw = self.thr_i - e * self.kp_thr - rate * self.kd_thr
        if not ((raw <= self.thr_min and e > 0) or (raw >= 1.0 and e < 0)):
            self.thr_i -= e * self.ki_thr * self.dt
        self.thr_i = clip(self.thr_i, self.thr_min, 1.0)
        cmd = self.thr_i - e * self.kp_thr - rate * self.kd_thr
        lim = self.thr_slew * self.dt
        self.thr = clip(clip(cmd, self.thr - lim, self.thr + lim), self.thr_min, 1.0)

        # heading -> bank -> aileron
        phi_cmd = clip(wrap180(sel_hdg - psi) * 1.2, -25.0, 25.0)
        ail = clip((phi_cmd - phi) * 0.020 - p * 0.55, -1.0, 1.0)

        # sideslip -> rudder (PI)
        self.beta_i = clip(self.beta_i + beta * self.dt, -30.0, 30.0)
        rud = clip(-(0.03 * beta + 0.01 * self.beta_i), -0.4, 0.4)

        self.cmd = {'elevator': elev, 'aileron': ail, 'rudder': rud, 'throttle': self.thr}
        return self.cmd
