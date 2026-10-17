"""Apex planner: receding-horizon racing line through an obstacle-aware corridor.

Each step (road model only, no map):
  1. Corridor: lateral bounds along the previewed centerline, shrunk by the car's half width
     plus a clearance, in two tiers: hard minimum clearances and soft preferred ones that the
     optimiser gives up only where the geometry forces it. Obstacles cut the corridor on one
     side; the pass side is chosen once per obstacle (more room wins) and remembered.
  2. Path: lateral offsets n(s) that minimise path curvature inside the corridor, a small
     QP with hard and soft box constraints, solved with sticky active sets. Warm-started from the previous plan
     so consecutive plans agree (the path-deviation score compares against a 0.5 s old plan).
  3. Speed: friction-ellipse forward/backward pass over the planned path's curvature; the end
     of the preview is treated as a possible tight unseen corner.
  4. Tracking: curvature feedforward + lateral/heading feedback on the planned path, evaluated
     at the pose predicted after actuator lag (pure pursuit kept as ``tracker: pp``). Each plan
     is anchored to the previous one, so drift shows up as tracking error. Steering is capped
     at the grip limit and slew-limited below the actuator limit. Longitudinal: acceleration feedforward + PI.

Only depends on numpy and autocross.api."""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81


class ApexPlanner(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c, vp = self.config, vehicle
        self.half_w = 0.5 * vp.width
        self.half_l = 0.5 * vp.length
        # clearances beyond the car's half width: preferred (soft) and minimum (hard)
        self.edge_margin = float(c.get("edge_margin", 0.65))      # boundary cones (covers body sweep when yawed)
        self.edge_margin_min = float(c.get("edge_margin_min", 0.30))
        self.obs_margin = float(c.get("obs_margin", 0.60))        # obstacle cones
        self.obs_margin_min = float(c.get("obs_margin_min", 0.20))
        self.obs_win_hard = float(c.get("obs_win_hard", 0.3))    # m beyond half car length the hard obstacle cut extends
        self.w_soft = float(c.get("w_soft", 0.3))                 # weight of the preferred clearances
        self.lat_frac = float(c.get("lat_frac", 0.85))            # share of mu*g used for cornering
        self.lon_frac = float(c.get("lon_frac", 0.65))            # share of braking used in planning
        self.v_max = float(c.get("v_max", 30.0))
        self.r_unseen = float(c.get("r_unseen", 15.0))             # assumed radius just past the preview
        self.w_curv = float(c.get("w_curv", 1.0))
        self.w_slope = float(c.get("w_slope", 0.02))
        self.w_prev = float(c.get("w_prev", 0.02))
        self.k_ld = float(c.get("lookahead_gain", 0.5))
        self.ld_min = float(c.get("lookahead_min", 2.0))
        self.tracker = str(c.get("tracker", "ff"))                 # pp = pure pursuit, ff = feedforward+feedback
        self.t_pred = float(c.get("t_pred", 0.12))                 # s, actuator-lag prediction
        self.ff_ahead = float(c.get("ff_ahead", 0.05))             # s, feedforward curvature preview
        self.ell_gain = float(c.get("ell_gain", 0.30))
        self.ell_min = float(c.get("ell_min", 3.0))
        self.zeta = float(c.get("zeta", 0.7))
        self.grip_cap = float(c.get("grip_cap", 1.08))             # max steer as a share of grip-limited steer
        self.reanchor = float(c.get("reanchor_m", 0.6))
        self.ease_m = float(c.get("ease_m", 8.0))
        self.kp, self.ki = float(c.get("kp", 0.6)), float(c.get("ki", 0.15))
        self.a_smooth = float(c.get("a_smooth", 0.5))            # low-pass factor on the accel command
        self.narrow_frac = float(c.get("narrow_frac", 0.7))      # cornering-grip share where the hard corridor has no slack
        self.slack_ref = float(c.get("slack_ref", 2.5))          # m of hard-corridor slack that counts as "roomy"
        self.thr_pow = float(c.get("thr_pow", 2.0))              # throttle cap = (spare grip)^thr_pow while cornering
        self.comb_frac = float(c.get("comb_frac", 0.85))          # max combined accel as a share of mu*g
        self.tc_slip = math.radians(float(c.get("tc_slip_deg", 3.0)))   # traction control starts here ...
        self.tc_band = math.radians(float(c.get("tc_band_deg", 2.5)))   # ... and cuts throttle fully this much later

        self.t_pred += vp.actuation_delay_steps * vp.control_dt   # published actuation delay
        self.a_lat = self.lat_frac * vp.mu * G
        self.a_dec = self.lon_frac * vp.max_decel
        self.a_acc = vp.max_accel
        self.prev_n = None            # previous plan (offsets) and the station it started at
        self.prev_progress = None
        self.pass_side = {}           # obstacle key (rounded world position) -> +1 left / -1 right
        self.i_err = 0.0
        self.last_steer = 0.0
        self.steer_step = 0.95 * vp.max_steer_rate * vp.control_dt

    # ------------------------------------------------------------------ corridor
    def _corridor(self, obs: Observation, road, n0: float, v: float):
        """Two-tier corridor in lateral offset n along the preview: hard bounds (minimum
        clearances, never crossed) and soft bounds (preferred clearances, given up only when
        the geometry forces it, e.g. alternating obstacles or a narrow road)."""
        s = road.preview_s
        ramp = np.clip(s / max(self.ease_m, 0.6 * v), 0.0, 1.0)

        def edges(margin):
            clr = self.half_w + margin
            lb, ub = -road.preview_width_right + clr, road.preview_width_left - clr
            # ease the edge bounds in near the car so a start outside them does not produce a
            # kink (obstacle cut-outs below are never eased: those cones must be cleared)
            return (np.minimum(lb, n0 + (lb - n0) * ramp - 1e-3), np.maximum(ub, n0 + (ub - n0) * ramp + 1e-3))

        lbh, ubh = edges(self.edge_margin_min)
        lbs, ubs = edges(self.edge_margin)
        c, sn = math.cos(obs.yaw), math.sin(obs.yaw)
        for o in road.obstacles:
            wx, wy = obs.x + c * o.center[0] - sn * o.center[1], obs.y + sn * o.center[0] + c * o.center[1]
            key = (round(wx / 3.0), round(wy / 3.0))
            side = self.pass_side.get(key)
            if side is None:
                i = min(int(np.searchsorted(s, max(o.distance, 0.0))), len(s) - 1)
                room_left = ubh[i] - (o.lateral_max + self.half_w + self.obs_margin_min)
                room_right = (o.lateral_min - self.half_w - self.obs_margin_min) - lbh[i]
                side = 1 if room_left >= room_right else -1
                self.pass_side[key] = side
            for lo, hi, margin, win in ((lbh, ubh, self.obs_margin_min, self.half_l + self.obs_win_hard),
                                        (lbs, ubs, self.obs_margin, self.half_l + 1.5)):
                m = np.abs(s - o.distance) <= win
                if side > 0:
                    lo[m] = np.maximum(lo[m], o.lateral_max + self.half_w + margin)
                else:
                    hi[m] = np.minimum(hi[m], o.lateral_min - self.half_w - margin)
        bad = lbh > ubh                       # truly blocked: aim for the middle of the gap
        mid = 0.5 * (lbh + ubh)
        lbh[bad], ubh[bad] = mid[bad], mid[bad]
        lbs, ubs = np.clip(lbs, lbh, ubh), np.clip(ubs, lbh, ubh)
        bad = lbs > ubs                       # preferred clearances conflict: split the difference
        mid = 0.5 * (lbs + ubs)
        lbs[bad], ubs[bad] = mid[bad], mid[bad]
        return lbh, ubh, lbs, ubs

    # ------------------------------------------------------------------ path QP
    def _plan(self, road, n0, lb, ub, lbs, ubs, warm):
        s, kref = road.preview_s, road.preview_curvature
        K = len(s)
        ds = float(np.mean(np.diff(s))) if K > 1 else 1.0
        # curvature of the offset path, linearised: k ~ kref + kref^2 n + n''
        D2 = np.zeros((K - 2, K))
        idx = np.arange(K - 2)
        D2[idx, idx], D2[idx, idx + 1], D2[idx, idx + 2] = 1.0, -2.0, 1.0
        D2 /= ds * ds
        A = D2 + np.diag(kref[1:-1] ** 2) @ np.eye(K)[1:-1]
        b = -kref[1:-1]
        D1 = (np.eye(K, k=1) - np.eye(K))[:-1] / ds
        H = self.w_curv * A.T @ A + self.w_slope * D1.T @ D1
        f = self.w_curv * A.T @ b
        if warm is not None:
            H = H + self.w_prev * np.eye(K)
            f = f + self.w_prev * warm
        # n[0] fixed to the car's offset
        W = 1e6
        H[0, 0] += W
        f[0] += W * n0
        # sticky active sets: hard bounds with a stiff penalty, soft bounds with a mild one
        Wb, Ws = 1e4, self.w_soft
        act = [np.zeros(K, bool) for _ in range(4)]      # hard lo, hard hi, soft lo, soft hi
        bounds = (lb, ub, lbs, ubs)
        weights = (Wb, Wb, Ws, Ws)
        n = np.zeros(K)
        for _ in range(10):
            Hk, fk = H.copy(), f.copy()
            for a, bnd, w in zip(act, bounds, weights):
                Hk[np.diag_indices(K)] += w * a
                fk += w * bnd * a
            n = np.linalg.solve(Hk, fk)
            grew = False
            for j, (a, bnd) in enumerate(zip(act, bounds)):
                new = (n < bnd - 0.01) if j % 2 == 0 else (n > bnd + 0.01)
                new[0] = False
                if (new & ~a).any():
                    grew = True
                a |= new
            if not grew:
                break
        return np.clip(n, lb, ub) if K > 1 else n

    def _path_xy(self, road, n):
        cl = road.preview_centerline
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nrm = np.column_stack((-t[:, 1], t[:, 0]))
        return cl + nrm * n[:, None]

    @staticmethod
    def _curvature(p):
        d = np.diff(p, axis=0)
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        seg = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        k = np.diff(h) / (0.5 * (seg[1:] + seg[:-1]))
        k = np.concatenate(([k[0]], k, [k[-1]])) if len(k) else np.zeros(len(p))
        k = np.convolve(np.pad(k, 2, mode="edge"), np.ones(5) / 5, mode="valid")
        s = np.concatenate(([0.0], np.cumsum(seg)))
        return k, s

    # ------------------------------------------------------------------ speed profile
    def _speed_profile(self, k, s, v0, slack=None):
        # where the corridor leaves little room to recover from a slide, corner more gently
        a_lat = np.full(len(k), self.a_lat)
        if slack is not None and self.narrow_frac < 1.0:
            room = np.clip(slack / self.slack_ref, 0.0, 1.0)
            a_lat *= self.narrow_frac + (1.0 - self.narrow_frac) * room
        v_lim = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(k), 1e-4)), self.v_max)
        v_lim[-1] = min(v_lim[-1], math.sqrt(a_lat[-1] * self.r_unseen))
        v = v_lim.copy()
        ds = np.diff(s)
        for i in range(len(v) - 2, -1, -1):
            use = min((v[i + 1] ** 2 * abs(k[i + 1])) / a_lat[i + 1], 1.0)
            a = self.a_dec * math.sqrt(max(1.0 - use * use, 0.05))
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a * ds[i]))
        fwd = v.copy()
        fwd[0] = min(v[0], max(v0, 0.0))
        for i in range(len(v) - 1):
            use = min((fwd[i] ** 2 * abs(k[i])) / a_lat[i], 1.0)
            a = self.a_acc * math.sqrt(max(1.0 - use * use, 0.05))
            fwd[i + 1] = min(v[i + 1], math.sqrt(fwd[i] ** 2 + 2.0 * a * ds[i]))
        return v, fwd

    # ------------------------------------------------------------------ tracking
    def _track_ff(self, path, k, s, v, obs):
        """Curvature feedforward + lateral/heading feedback, evaluated at the pose the car will
        have after the actuator lag. Spatial second-order error dynamics e'' = delta/L - kappa
        closed with natural length ``ell`` (grows with speed for stability)."""
        vp = self.vehicle
        tp = self.t_pred
        # predicted pose (vehicle frame) after tp seconds, constant speed and yaw rate
        yr = obs.yaw_rate
        th = yr * tp
        px = v * tp * (1.0 - th * th / 6.0)
        py = v * tp * th / 2.0
        d = np.hypot(path[:, 0] - px, path[:, 1] - py)
        j = int(np.argmin(d))
        j1 = min(max(j - (1 if j == len(path) - 1 else 0), 0), len(path) - 2)
        a, b = path[j1], path[j1 + 1]
        seg = b - a
        L = max(math.hypot(*seg), 1e-6)
        rel = np.array([px, py]) - a
        u = float(np.clip((rel @ seg) / (L * L), 0.0, 1.0))
        e = (seg[0] * rel[1] - seg[1] * rel[0]) / L                   # + = car left of path
        th_err = th - math.atan2(seg[1], seg[0])                       # + = car points left of path
        th_err = (th_err + math.pi) % (2 * math.pi) - math.pi
        s_here = s[j1] + u * L
        k_ff = float(np.interp(s_here + v * self.ff_ahead, s, k))
        ell = max(self.ell_min, self.ell_gain * v)
        k_e, k_th = 1.0 / (ell * ell), 2.0 * self.zeta / ell
        kappa_cmd = k_ff - k_e * e - k_th * th_err
        return math.atan(vp.wheelbase * kappa_cmd)

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid or len(road.preview_s) < 5:
            return Command(brake=1.0)
        v = max(obs.speed, 0.0)

        lat = float(road.lateral_offset)
        warm = None
        if self.prev_n is not None and self.prev_progress is not None:
            shift = obs.progress - self.prev_progress
            warm = np.interp(road.preview_s + shift, self.prev_s, self.prev_n, right=self.prev_n[-1])
        # anchor the new plan to the old one (so the tracker sees, and corrects, real error);
        # re-anchor to the car only if it has drifted far from the plan
        n0 = float(warm[0]) if warm is not None and abs(warm[0] - lat) < self.reanchor else lat
        lb, ub, lbs, ubs = self._corridor(obs, road, n0, v)
        n = self._plan(road, n0, lb, ub, lbs, ubs, warm)
        self.prev_n, self.prev_s, self.prev_progress = n, road.preview_s.copy(), obs.progress

        path = self._path_xy(road, n)
        k, s = self._curvature(path)
        v_back, v_prof = self._speed_profile(k, s, v, ub - lb)

        e = float(np.interp(0.0, path[:, 0], path[:, 1])) if path[0, 0] <= 0.0 <= path[-1, 0] else float(path[0, 1])
        kff = float(np.interp(v * 0.2, s, k))
        if self.tracker == "ff":
            delta = self._track_ff(path, k, s, v, obs)
        else:
            # pure pursuit on the planned path
            ld = float(np.clip(self.k_ld * v + self.ld_min, self.ld_min, 14.0))
            r = np.hypot(path[:, 0], path[:, 1])
            ahead = (path[:, 0] > 0.5) & (r >= ld)
            tgt = path[int(np.argmax(ahead))] if ahead.any() else path[-1]
            alpha = math.atan2(tgt[1], max(tgt[0], 0.1))
            delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(*tgt), 1.0))
        # never command more steering than the tyres can use at this speed (the realism score
        # flags commands needing >110% of grip, and they only scrub speed anyway)
        if v > 3.0:
            d_max = math.atan(self.grip_cap * vp.mu * G * vp.wheelbase / (v * v))
            delta = float(np.clip(delta, -d_max, d_max))
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))
        steer = float(np.clip(steer, self.last_steer - self.steer_step, self.last_steer + self.steer_step))
        self.last_steer = steer

        # ---- longitudinal: the profile's own acceleration as feedforward + PI on speed error,
        # low-passed, and limited to the grip left over from cornering (braking hard mid-corner
        # unloads the rear and spins the car)
        # v_prof = accelerate from the current speed, capped by the braking envelope v_back
        v_tgt = float(np.interp(max(v * vp.control_dt, 0.5), s, v_prof))
        a_prof = np.diff(v_prof ** 2) / (2.0 * np.maximum(np.diff(s), 1e-3))
        a_ff = float(np.interp(v * 0.25, 0.5 * (s[1:] + s[:-1]), a_prof))
        err = v_tgt - v
        self.i_err = float(np.clip(self.i_err + err * vp.control_dt, -3.0, 3.0))
        a_raw = a_ff + self.kp * err + self.ki * self.i_err
        lat_use = min(abs(obs.ay) / (vp.mu * G), 0.95)
        spare = math.sqrt(1.0 - lat_use ** 2)
        # braking: keep the combined (cornering + braking) demand within comb_frac of grip
        a_tot = self.comb_frac * vp.mu * G
        dec_max = min(vp.max_decel, math.sqrt(max(a_tot ** 2 - obs.ay ** 2, 0.0)))
        a_raw = float(np.clip(a_raw, -max(dec_max, 0.3 * vp.max_decel * spare), vp.max_accel * max(spare ** self.thr_pow, 0.15)))
        self.a_cmd = getattr(self, "a_cmd", 0.0) + self.a_smooth * (a_raw - getattr(self, "a_cmd", 0.0))
        a_cmd = self.a_cmd
        if a_cmd >= 0.0:
            throttle, brake = float(np.clip((a_cmd + 0.15) / vp.max_accel, 0.0, 1.0)), 0.0
            # traction control: fade the throttle out as the rear axle starts to slide
            rear_slip = abs(math.atan2(obs.vy - vp.cg_to_rear * obs.yaw_rate, max(abs(obs.vx), 0.1)))
            if v > 2.0 and rear_slip > self.tc_slip:
                throttle *= max(0.0, 1.0 - (rear_slip - self.tc_slip) / self.tc_band)
        else:
            throttle, brake = 0.0, float(np.clip(-a_cmd / vp.max_decel, 0.0, 1.0))

        return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                       target_speed=v_tgt,
                       debug={"e_path": e, "kff": kff, "n0": n0, "a_cmd": a_cmd, "lb0": float(lb[0]),
                              "ub0": float(ub[0])})
