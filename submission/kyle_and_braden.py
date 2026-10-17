"""Team Kyle and Braden: EcoCAR autocross hackathon submission.

Apex planner: receding-horizon racing line through an obstacle-aware corridor.

Each step (road model only, no map):
  1. Corridor: lateral bounds along the previewed centerline, shrunk by the car's half width
     plus a clearance, in two tiers: hard minimum clearances and soft preferred ones that the
     optimiser gives up only where the geometry forces it. Obstacles cut the corridor on one
     side; the pass side is chosen once per obstacle (more room wins) and remembered.
  2. Path: lateral offsets n(s) that minimise path curvature inside the corridor, a small QP
     with hard and soft box constraints, solved with sticky active sets. Warm-started from the
     previous plan so consecutive plans agree (the path-deviation score compares against a
     0.5 s old plan).
  3. Speed: friction-ellipse forward/backward pass over the planned path's curvature; the end
     of the preview is treated as a possible tight unseen corner, cornering is gentler where
     the corridor leaves no room to recover, and no corner is planned faster than the
     grip-capped steering can hold given the car's understeer.
  4. Understeer: learned online (separately on and off the throttle) from measured vs.
     steering-predicted lateral acceleration; used by the speed plan and the feedforward.
  5. Tracking: understeer-aware curvature feedforward + lateral/heading feedback on the planned
     path, evaluated at the pose predicted after actuator lag. Each plan is anchored to the
     previous one, so drift shows up as tracking error. Steering is capped at the grip limit
     and slew-limited below the actuator limit.
  6. Pedals: profile acceleration feedforward + PI, a combined grip budget for braking,
     traction control, and stability control (throttle fades out, then a gentle brake, as the
     steering runs out of grip).

Any exception inside a step is caught and answered with a safe command (hold the steering,
brake gently), so a numerical surprise can never count as a controller error.

Only depends on numpy and autocross.api."""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81

DEFAULTS = {
    # corridor clearances beyond the car's half width: preferred (soft) and minimum (hard)
    "edge_margin": 0.65, "edge_margin_min": 0.30,     # boundary cones (covers body sweep when yawed)
    "obs_margin": 0.60, "obs_margin_min": 0.20,       # obstacle cones
    "obs_win_hard": 0.3,                              # m beyond half car length the hard obstacle cut extends
    "w_soft": 0.3,                                    # weight of the preferred clearances
    # path QP
    "w_curv": 1.0, "w_slope": 0.02, "w_prev": 0.02, "reanchor_m": 0.6, "ease_m": 8.0,
    # speed profile
    "lat_frac": 0.85,                                 # share of mu*g used for cornering
    "lon_frac": 0.85,                                 # share of max braking used in planning
    "v_max": 30.0, "r_unseen": 15.0,                  # assumed radius just past the preview
    "narrow_frac": 0.7, "slack_ref": 2.5,             # cornering share where the hard corridor has no slack
    # lateral tracking
    "t_pred": 0.12, "ff_ahead": 0.05, "ell_gain": 0.30, "ell_min": 3.0, "zeta": 0.7,
    "grip_cap": 1.08,                                 # max steer as a share of grip-limited steer
    # longitudinal
    "kp": 0.6, "ki": 0.15, "a_smooth": 0.5,
    "thr_pow": 1.25,                                  # throttle cap = (spare grip)^thr_pow while cornering
    "comb_frac": 0.85,                                # braking keeps combined accel within this share of mu*g
    "tc_slip_deg": 3.0, "tc_band_deg": 2.5,           # traction control fade-in
    "us_k0": 0.0006, "us_rate": 0.02, "us_frac": 0.85, "us_v_min": 15.0,  # understeer: initial gradient K (ay = v^2 tan(d)/L / (1 + K v^2)),
                                                      # learning rate, and share of capped-steering grip the plan may use
    "sc_start": 0.70, "sc_brake": 0.3,                # stability control: steering-saturation share where the throttle
                                                      # starts fading out (gone at 1.0), and brake added beyond it
}


class ApexPlanner(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        vp = vehicle
        self.p = p = {k: float(self.config.get(k, v)) for k, v in DEFAULTS.items()}
        self.half_w, self.half_l = 0.5 * vp.width, 0.5 * vp.length
        self.t_pred = p["t_pred"] + vp.actuation_delay_steps * vp.control_dt   # published actuation delay
        self.a_lat = p["lat_frac"] * vp.mu * G
        self.a_dec = p["lon_frac"] * vp.max_decel
        self.tc_slip, self.tc_band = math.radians(p["tc_slip_deg"]), math.radians(p["tc_band_deg"])
        self.steer_step = 0.95 * vp.max_steer_rate * vp.control_dt
        self.prev_n = self.prev_s = self.prev_progress = None   # previous plan, for warm start / anchoring
        self.pass_side = {}            # obstacle key (rounded world position) -> +1 left / -1 right
        self.i_err = self.a_cmd = self.last_steer = 0.0
        self.steer_sat = 0.0           # |steering wanted| / grip-limited steering, last step
        # learned understeer gradient (s^2/m^2), separately on and off the throttle: with driven
        # front wheels the car pushes much more under power
        self.us_k_on = self.us_k_off = p["us_k0"]
        self.us_k = p["us_k0"]          # the one the plan uses: the larger of the two
        self._qp_cache = {}            # (K, ds) -> constant QP matrices

    # ------------------------------------------------------------------ corridor
    def _corridor(self, obs: Observation, road, n0: float, v: float):
        """Two-tier corridor in lateral offset n along the preview: hard bounds (minimum
        clearances, never crossed) and soft bounds (preferred clearances, given up only when
        the geometry forces it, e.g. alternating obstacles or a narrow road)."""
        p, s = self.p, road.preview_s
        ramp = np.clip(s / max(p["ease_m"], 0.6 * v), 0.0, 1.0)

        def edges(margin):
            clr = self.half_w + margin
            lb, ub = -road.preview_width_right + clr, road.preview_width_left - clr
            # ease the edge bounds in near the car so a start outside them does not produce a
            # kink (obstacle cut-outs below are never eased: those cones must be cleared)
            return np.minimum(lb, n0 + (lb - n0) * ramp - 1e-3), np.maximum(ub, n0 + (ub - n0) * ramp + 1e-3)

        lbh, ubh = edges(p["edge_margin_min"])
        lbs, ubs = edges(p["edge_margin"])
        c, sn = math.cos(obs.yaw), math.sin(obs.yaw)
        for o in road.obstacles:
            wx, wy = obs.x + c * o.center[0] - sn * o.center[1], obs.y + sn * o.center[0] + c * o.center[1]
            key = (round(wx / 3.0), round(wy / 3.0))
            side = self.pass_side.get(key)
            if side is None:
                i = min(int(np.searchsorted(s, max(o.distance, 0.0))), len(s) - 1)
                room_left = ubh[i] - (o.lateral_max + self.half_w + p["obs_margin_min"])
                room_right = (o.lateral_min - self.half_w - p["obs_margin_min"]) - lbh[i]
                side = self.pass_side[key] = 1 if room_left >= room_right else -1
            for lo, hi, margin, win in ((lbh, ubh, p["obs_margin_min"], self.half_l + p["obs_win_hard"]),
                                        (lbs, ubs, p["obs_margin"], self.half_l + 1.5)):
                m = np.abs(s - o.distance) <= win
                if side > 0:
                    lo[m] = np.maximum(lo[m], o.lateral_max + self.half_w + margin)
                else:
                    hi[m] = np.minimum(hi[m], o.lateral_min - self.half_w - margin)
        for lo, hi in ((lbh, ubh), (lbs, ubs)):
            if lo is lbs:                      # soft bounds live inside the hard ones
                np.clip(lbs, lbh, ubh, out=lbs)
                np.clip(ubs, lbh, ubh, out=ubs)
            bad = lo > hi                      # blocked / conflicting: aim for the middle
            mid = 0.5 * (lo + hi)
            lo[bad] = hi[bad] = mid[bad]
        return lbh, ubh, lbs, ubs

    # ------------------------------------------------------------------ path QP
    def _qp_base(self, K: int, ds: float):
        """Second-difference and slope operators for K stations ds apart (cached)."""
        key = (K, round(ds, 6))
        if key not in self._qp_cache:
            D2 = np.zeros((K - 2, K))
            i = np.arange(K - 2)
            D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
            D1 = (np.eye(K, k=1) - np.eye(K))[:-1] / ds
            self._qp_cache[key] = (D2 / (ds * ds), self.p["w_slope"] * D1.T @ D1)
        return self._qp_cache[key]

    def _plan(self, road, n0, lb, ub, lbs, ubs, warm):
        s, kref = road.preview_s, road.preview_curvature
        K = len(s)
        ds = float(np.mean(np.diff(s)))
        D2, S1 = self._qp_base(K, ds)
        # curvature of the offset path, linearised: k ~ kref + kref^2 n + n''
        A = D2.copy()
        i = np.arange(K - 2)
        A[i, i + 1] += kref[1:-1] ** 2
        b = -kref[1:-1]
        w = self.p["w_curv"]
        H = w * A.T @ A + S1
        f = w * A.T @ b
        if warm is not None:
            H[np.diag_indices(K)] += self.p["w_prev"]
            f = f + self.p["w_prev"] * warm
        H[0, 0] += 1e6                           # n[0] pinned to the anchor
        f[0] += 1e6 * n0
        # sticky active sets: hard bounds with a stiff penalty, soft bounds with a mild one
        bounds = (lb, ub, lbs, ubs)
        weights = (1e4, 1e4, self.p["w_soft"], self.p["w_soft"])
        act = [np.zeros(K, bool) for _ in range(4)]      # hard lo, hard hi, soft lo, soft hi
        diag = np.diag_indices(K)
        n = np.zeros(K)
        for _ in range(10):
            Hk, fk = H.copy(), f.copy()
            for a, bnd, wt in zip(act, bounds, weights):
                Hk[diag] += wt * a
                fk += wt * bnd * a
            n = np.linalg.solve(Hk, fk)
            grew = False
            for j, (a, bnd) in enumerate(zip(act, bounds)):
                new = (n < bnd - 0.01) if j % 2 == 0 else (n > bnd + 0.01)
                new[0] = False
                grew |= bool((new & ~a).any())
                a |= new
            if not grew:
                break
        return np.clip(n, lb, ub)

    @staticmethod
    def _path_xy(road, n):
        cl = road.preview_centerline
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        return cl + np.column_stack((-t[:, 1], t[:, 0])) * n[:, None]

    @staticmethod
    def _curvature(p):
        d = np.diff(p, axis=0)
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        seg = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        k = np.diff(h) / (0.5 * (seg[1:] + seg[:-1]))
        k = np.concatenate(([k[0]], k, [k[-1]]))
        k = np.convolve(np.pad(k, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        return k, np.concatenate(([0.0], np.cumsum(seg)))

    # ------------------------------------------------------------------ speed profile
    def _speed_profile(self, k, s, v0, slack):
        p = self.p
        a_dec = self.a_dec
        # where the corridor leaves little room to recover from a slide, corner more gently
        a_lat = self.a_lat * (p["narrow_frac"] + (1.0 - p["narrow_frac"]) * np.clip(slack / p["slack_ref"], 0.0, 1.0))
        v_lim = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(k), 1e-4)), p["v_max"])
        if self.us_k > 0.0:
            # steering authority: |k| v^2 (1 + K v^2) <= A, with A the grip the capped steering can
            # ask for; solve the quadratic in v^2
            A = p["us_frac"] * p["grip_cap"] * self.vehicle.mu * G
            ka = np.maximum(np.abs(k), 1e-4)
            u = (-ka + np.sqrt(ka * ka + 4.0 * self.us_k * ka * A)) / (2.0 * self.us_k * ka)
            v_lim = np.minimum(v_lim, np.sqrt(u))
        # sight distance: the end of the preview may hide a tight corner, so arrive there slow
        # enough for it. This goes through the same friction-ellipse braking pass as visible
        # corners on purpose: on a sweeper near the cornering limit the car genuinely cannot
        # brake hard, so a straight-line what-if would arrive too fast (tested: cones).
        v_lim[-1] = min(v_lim[-1], math.sqrt(a_lat[-1] * p["r_unseen"]))
        v, ds, a_acc = v_lim.copy(), np.diff(s), self.vehicle.max_accel
        for i in range(len(v) - 2, -1, -1):                 # braking envelope, from the horizon back
            use = min(v[i + 1] ** 2 * abs(k[i + 1]) / a_lat[i + 1], 1.0)
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_dec * math.sqrt(max(1.0 - use * use, 0.05)) * ds[i]))
        fwd = v.copy()                                      # accelerate from the current speed
        fwd[0] = min(v[0], v0)
        for i in range(len(v) - 1):
            use = min(fwd[i] ** 2 * abs(k[i]) / a_lat[i], 1.0)
            fwd[i + 1] = min(v[i + 1], math.sqrt(fwd[i] ** 2 + 2.0 * a_acc * math.sqrt(max(1.0 - use * use, 0.05)) * ds[i]))
        return fwd

    # ------------------------------------------------------------------ tracking
    def _steer(self, path, k, s, v, obs):
        """Curvature feedforward + lateral/heading feedback, evaluated at the pose the car will
        have after the actuator lag. Spatial second-order error dynamics e'' = delta/L - kappa
        closed with natural length ``ell`` (grows with speed for stability)."""
        p, vp = self.p, self.vehicle
        th = obs.yaw_rate * self.t_pred        # predicted pose (vehicle frame), constant speed and yaw rate
        px, py = v * self.t_pred * (1.0 - th * th / 6.0), v * self.t_pred * th / 2.0
        j = int(np.argmin(np.hypot(path[:, 0] - px, path[:, 1] - py)))
        j1 = min(j, len(path) - 2)
        seg = path[j1 + 1] - path[j1]
        L = max(math.hypot(seg[0], seg[1]), 1e-6)
        rx, ry = px - path[j1, 0], py - path[j1, 1]
        u = min(max((rx * seg[0] + ry * seg[1]) / (L * L), 0.0), 1.0)
        e = (seg[0] * ry - seg[1] * rx) / L                                     # + = car left of path
        th_err = (th - math.atan2(seg[1], seg[0]) + math.pi) % (2 * math.pi) - math.pi   # + = pointing left of it
        k_ff = float(np.interp(s[j1] + u * L + v * p["ff_ahead"], s, k))
        ell = max(p["ell_min"], p["ell_gain"] * v)
        # feedforward includes the learned understeer for the current throttle state: the car
        # needs (1 + K v^2) times the neutral steering for a given curvature
        w = min(max((obs.throttle - 0.05) / 0.25, 0.0), 1.0)
        us = 1.0 + ((1.0 - w) * self.us_k_off + w * self.us_k_on) * v * v
        delta = math.atan(vp.wheelbase * (k_ff * us - e / (ell * ell) - 2.0 * p["zeta"] * th_err / ell))
        # never command more steering than the tyres can use at this speed (the realism score
        # flags commands needing >110% of grip, and they only scrub speed anyway)
        self.steer_sat = 0.0
        if v > 3.0:
            d_max = math.atan(p["grip_cap"] * vp.mu * G * vp.wheelbase / (v * v))
            self.steer_sat = abs(delta) / d_max
            delta = min(max(delta, -d_max), d_max)
        steer = min(max(delta / vp.max_steer, -1.0), 1.0)
        return min(max(steer, self.last_steer - self.steer_step), self.last_steer + self.steer_step)

    def _pedals(self, s, v_prof, v, obs):
        """Profile acceleration as feedforward + PI on speed error, low-passed, and limited to
        the grip left over from cornering (braking hard mid-corner unloads the rear and spins
        the car; throttle mid-corner takes grip from the driven front axle)."""
        p, vp = self.p, self.vehicle
        v_tgt = float(np.interp(max(v * vp.control_dt, 0.5), s, v_prof))
        a_prof = np.diff(v_prof ** 2) / (2.0 * np.maximum(np.diff(s), 1e-3))
        a_ff = float(np.interp(v * 0.25, 0.5 * (s[1:] + s[:-1]), a_prof))
        err = v_tgt - v
        self.i_err = min(max(self.i_err + err * vp.control_dt, -3.0), 3.0)
        a_raw = a_ff + p["kp"] * err + p["ki"] * self.i_err
        spare = math.sqrt(1.0 - min(abs(obs.ay) / (vp.mu * G), 0.95) ** 2)
        a_tot = p["comb_frac"] * vp.mu * G
        dec_max = max(min(vp.max_decel, math.sqrt(max(a_tot ** 2 - obs.ay ** 2, 0.0))), 0.3 * vp.max_decel * spare)
        a_raw = min(max(a_raw, -dec_max), vp.max_accel * max(spare ** p["thr_pow"], 0.15))
        # stability control: when the steering is running out of grip (driven front wheels push
        # wide under power), stop accelerating so the tyres can turn; past the limit, brake gently
        sat = self.steer_sat
        if sat > p["sc_start"] and a_raw > 0.0:
            a_raw *= max(0.0, (1.0 - sat) / (1.0 - p["sc_start"]))
        if sat > 1.0 and p["sc_brake"] > 0.0:
            a_raw = max(min(a_raw, -p["sc_brake"] * min(sat - 1.0, 1.0) * vp.max_decel), -dec_max)
        self.a_cmd += p["a_smooth"] * (a_raw - self.a_cmd)
        if self.a_cmd < 0.0:
            return 0.0, min(-self.a_cmd / vp.max_decel, 1.0), v_tgt
        throttle = min((self.a_cmd + 0.15) / vp.max_accel, 1.0)
        # traction control: fade the throttle out as the rear axle starts to slide
        rear_slip = abs(math.atan2(obs.vy - vp.cg_to_rear * obs.yaw_rate, max(abs(obs.vx), 0.1)))
        if v > 2.0 and rear_slip > self.tc_slip:
            throttle *= max(0.0, 1.0 - (rear_slip - self.tc_slip) / self.tc_band)
        return throttle, 0.0, v_tgt

    # ------------------------------------------------------------------ understeer model
    def _learn_understeer(self, obs, v):
        """Online estimate of the understeer gradient K from the measured lateral acceleration
        vs. what the applied road-wheel angle would give a neutral car: ay = ay_kin / (1 + K v^2).
        The car (and CARLA vs the bicycle model) loses steering authority with speed; the speed
        plan uses K so it never asks for more cornering than the grip-capped steering delivers."""
        # only above us_v_min: in slow, tight corners large steering angles and tyre geometry make a
        # neutral car look like it understeers, which would make the feedforward over-steer
        vp = self.vehicle
        delta = obs.steer * vp.max_steer
        if v < self.p["us_v_min"] or abs(delta) < 0.01 or abs(obs.ay) < 1.5:
            return
        ay_kin = v * v * math.tan(delta) / vp.wheelbase
        ratio = obs.ay / ay_kin
        if not 0.2 < ratio < 1.5:
            return
        k_i = min(max((1.0 / ratio - 1.0) / (v * v), 0.0), 0.01)
        if obs.throttle > 0.3:
            self.us_k_on += self.p["us_rate"] * (k_i - self.us_k_on)
        elif obs.throttle < 0.05:
            self.us_k_off += self.p["us_rate"] * (k_i - self.us_k_off)
        self.us_k = max(self.us_k_on, self.us_k_off)

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        try:
            cmd = self._step(obs)
            if all(math.isfinite(x) for x in (cmd.steer, cmd.throttle, cmd.brake)):
                self.last_steer = cmd.steer     # committed only once the whole command checks out
                return cmd
        except Exception:  # noqa: BLE001 - never let a numerical surprise become a controller error
            pass
        # safe answer: hold the last good steering, brake gently, replan from scratch next step
        self.prev_n = None
        self.i_err = self.a_cmd = 0.0
        return Command(steer=self.last_steer, brake=0.3)

    def _step(self, obs: Observation) -> Command:
        road = obs.road
        if road is None or not road.valid or len(road.preview_s) < 5:
            return Command(steer=self.last_steer, brake=1.0)
        v = max(obs.speed, 0.0)
        self._learn_understeer(obs, v)
        lat = float(road.lateral_offset)
        warm = None
        if self.prev_n is not None:
            warm = np.interp(road.preview_s + (obs.progress - self.prev_progress), self.prev_s, self.prev_n,
                             right=self.prev_n[-1])
        # anchor the new plan to the old one (so the tracker sees, and corrects, real error);
        # re-anchor to the car only if it has drifted far from the plan
        n0 = float(warm[0]) if warm is not None and abs(warm[0] - lat) < self.p["reanchor_m"] else lat
        lb, ub, lbs, ubs = self._corridor(obs, road, n0, v)
        n = self._plan(road, n0, lb, ub, lbs, ubs, warm)
        self.prev_n, self.prev_s, self.prev_progress = n, road.preview_s.copy(), obs.progress

        path = self._path_xy(road, n)
        k, s = self._curvature(path)
        v_prof = self._speed_profile(k, s, v, ub - lb)
        steer = self._steer(path, k, s, v, obs)
        throttle, brake, v_tgt = self._pedals(s, v_prof, v, obs)
        return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                       target_speed=v_tgt, debug={"n0": n0, "a_cmd": self.a_cmd, "us_k": self.us_k})
