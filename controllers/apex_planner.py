"""Apex planner: receding-horizon racing line through an obstacle-aware corridor.

Each step (road model only, no map):
  1. Corridor: lateral bounds along the previewed centerline, shrunk by the car's half width
     plus a safety margin. Obstacles cut the corridor on one side; the pass side is chosen
     once per obstacle (more room wins) and remembered, so the plan does not flip.
  2. Path: lateral offsets n(s) that minimise path curvature inside the corridor, a small
     box-constrained QP solved with a sticky active set. Warm-started from the previous plan
     so consecutive plans agree (the path-deviation score compares against a 0.5 s old plan).
  3. Speed: friction-ellipse forward/backward pass over the planned path's curvature; the end
     of the preview is treated as a possible tight unseen corner.
  4. Tracking: pure pursuit on the planned path (each plan is anchored to the previous one,
     so drift shows up as tracking error), slew-limited below the actuator limit. Longitudinal: acceleration feedforward + PI.

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
        self.edge_margin = float(c.get("edge_margin", 0.65))      # clearance to boundary cones (covers body sweep when yawed)
        self.obs_margin = float(c.get("obs_margin", 0.40))        # extra clearance to obstacle cones
        self.lat_frac = float(c.get("lat_frac", 0.75))            # share of mu*g used for cornering
        self.lon_frac = float(c.get("lon_frac", 0.50))            # share of braking used in planning
        self.v_max = float(c.get("v_max", 30.0))
        self.r_unseen = float(c.get("r_unseen", 9.0))             # assumed radius just past the preview
        self.w_curv = float(c.get("w_curv", 1.0))
        self.w_slope = float(c.get("w_slope", 0.02))
        self.w_prev = float(c.get("w_prev", 0.02))
        self.k_ld = float(c.get("lookahead_gain", 0.5))
        self.ld_min = float(c.get("lookahead_min", 2.0))
        self.reanchor = float(c.get("reanchor_m", 0.6))
        self.ease_m = float(c.get("ease_m", 8.0))
        self.kp, self.ki = float(c.get("kp", 0.6)), float(c.get("ki", 0.15))
        self.a_smooth = float(c.get("a_smooth", 0.5))            # low-pass factor on the accel command

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
    def _corridor(self, obs: Observation, road):
        s = road.preview_s
        clr = self.half_w + self.edge_margin
        lb = -road.preview_width_right + clr
        ub = road.preview_width_left - clr
        oclr = self.half_w + self.obs_margin
        c, sn = math.cos(obs.yaw), math.sin(obs.yaw)
        for o in road.obstacles:
            wx, wy = obs.x + c * o.center[0] - sn * o.center[1], obs.y + sn * o.center[0] + c * o.center[1]
            key = (round(wx / 3.0), round(wy / 3.0))
            side = self.pass_side.get(key)
            i = min(int(np.searchsorted(s, max(o.distance, 0.0))), len(s) - 1)
            if side is None:
                room_left = ub[i] - (o.lateral_max + oclr)
                room_right = (o.lateral_min - oclr) - lb[i]
                side = 1 if room_left >= room_right else -1
                self.pass_side[key] = side
            m = np.abs(s - o.distance) <= self.half_l + 1.5
            if side > 0:
                lb[m] = np.maximum(lb[m], o.lateral_max + oclr)
            else:
                ub[m] = np.minimum(ub[m], o.lateral_min - oclr)
        bad = lb > ub
        mid = 0.5 * (lb + ub)
        lb[bad], ub[bad] = mid[bad], mid[bad]
        return lb, ub

    # ------------------------------------------------------------------ path QP
    def _plan(self, road, n0, lb, ub, warm):
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
        active_lo, active_hi = np.zeros(K, bool), np.zeros(K, bool)
        Wb = 1e4
        n = np.zeros(K)
        for _ in range(8):
            Hk, fk = H.copy(), f.copy()
            Hk[np.diag_indices(K)] += Wb * (active_lo | active_hi)
            fk += Wb * (lb * active_lo + ub * active_hi)
            n = np.linalg.solve(Hk, fk)
            new_lo, new_hi = (n < lb - 0.01), (n > ub + 0.01)
            new_lo[0] = new_hi[0] = False
            if not (new_lo & ~active_lo).any() and not (new_hi & ~active_hi).any():
                break
            active_lo |= new_lo
            active_hi |= new_hi
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
    def _speed_profile(self, k, s, v0):
        a_lat = self.a_lat
        v_lim = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(k), 1e-4)), self.v_max)
        v_lim[-1] = min(v_lim[-1], math.sqrt(a_lat * self.r_unseen))
        v = v_lim.copy()
        ds = np.diff(s)
        for i in range(len(v) - 2, -1, -1):
            use = min((v[i + 1] ** 2 * abs(k[i + 1])) / a_lat, 1.0)
            a = self.a_dec * math.sqrt(max(1.0 - use * use, 0.05))
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a * ds[i]))
        fwd = v.copy()
        fwd[0] = min(v[0], max(v0, 0.0))
        for i in range(len(v) - 1):
            use = min((fwd[i] ** 2 * abs(k[i])) / a_lat, 1.0)
            a = self.a_acc * math.sqrt(max(1.0 - use * use, 0.05))
            fwd[i + 1] = min(v[i + 1], math.sqrt(fwd[i] ** 2 + 2.0 * a * ds[i]))
        return v, fwd

    # ------------------------------------------------------------------ step
    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid or len(road.preview_s) < 5:
            return Command(brake=1.0)
        v = max(obs.speed, 0.0)

        lb, ub = self._corridor(obs, road)
        lat = float(road.lateral_offset)
        warm = None
        if self.prev_n is not None and self.prev_progress is not None:
            shift = obs.progress - self.prev_progress
            warm = np.interp(road.preview_s + shift, self.prev_s, self.prev_n, right=self.prev_n[-1])
        # anchor the new plan to the old one (so the tracker sees, and corrects, real error);
        # re-anchor to the car only if it has drifted far from the plan
        n0 = float(warm[0]) if warm is not None and abs(warm[0] - lat) < self.reanchor else lat
        # ease the corridor in near the car so a start outside it does not produce a kink
        ramp = np.clip(road.preview_s / max(self.ease_m, 0.6 * v), 0.0, 1.0)
        lb = np.minimum(lb, n0 + (lb - n0) * ramp - 1e-3)
        ub = np.maximum(ub, n0 + (ub - n0) * ramp + 1e-3)
        n = self._plan(road, n0, lb, ub, warm)
        self.prev_n, self.prev_s, self.prev_progress = n, road.preview_s.copy(), obs.progress

        path = self._path_xy(road, n)
        k, s = self._curvature(path)
        v_back, v_prof = self._speed_profile(k, s, v)

        # ---- lateral: pure pursuit on the planned path
        ld = float(np.clip(self.k_ld * v + self.ld_min, self.ld_min, 14.0))
        r = np.hypot(path[:, 0], path[:, 1])
        ahead = (path[:, 0] > 0.5) & (r >= ld)
        tgt = path[int(np.argmax(ahead))] if ahead.any() else path[-1]
        alpha = math.atan2(tgt[1], max(tgt[0], 0.1))
        delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(*tgt), 1.0))
        e = float(np.interp(0.0, path[:, 0], path[:, 1])) if path[0, 0] <= 0.0 <= path[-1, 0] else float(path[0, 1])
        kff = float(np.interp(v * 0.2, s, k))
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
        a_raw = float(np.clip(a_raw, -vp.max_decel * max(spare, 0.3), vp.max_accel * max(spare, 0.15)))
        self.a_cmd = getattr(self, "a_cmd", 0.0) + self.a_smooth * (a_raw - getattr(self, "a_cmd", 0.0))
        a_cmd = self.a_cmd
        if a_cmd >= 0.0:
            throttle, brake = float(np.clip((a_cmd + 0.15) / vp.max_accel, 0.0, 1.0)), 0.0
        else:
            throttle, brake = 0.0, float(np.clip(-a_cmd / vp.max_decel, 0.0, 1.0))

        return Command(steer=steer, throttle=throttle, brake=brake, path=path, path_frame="vehicle",
                       target_speed=v_tgt,
                       debug={"e_path": e, "kff": kff, "n0": n0, "a_cmd": a_cmd, "lb0": float(lb[0]),
                              "ub0": float(ub[0])})
