"""Builds the telemetry dashboard's data file from sim runs.

    python dashboard/build.py                     # latest run per (team, course) under runs/
    python dashboard/build.py --teams apex baseline
    python dashboard/build.py runs/<stamp>_<team>_<course> ...
    python dashboard/build.py --fresh              # re-run apex + baseline on the public courses first
    python dashboard/build.py --fresh -t "tracks/mock/*.yaml" -c controllers/apex_planner.py:apex

Writes dashboard/data.js (loaded by dashboard/index.html, which works straight from disk).

Besides the logged telemetry it derives channels the harness does not log:
  * grip use      - combined acceleration as a share of the published mu * g
  * energy        - road-load estimate (inertia + aero drag + rolling resistance) from ground
                    truth: traction energy drawn through the motor efficiency, minus what the
                    brakes could recover through regen. A modelled estimate, not a measurement.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autocross import track as T  # noqa: E402
from autocross.config import load_yaml  # noqa: E402

G = 9.81
RHO = 1.2
ETA_MOTOR = 0.90      # battery -> wheel
ETA_REGEN = 0.65      # wheel -> battery, share of braking energy recoverable

# telemetry channels read from the run (inputs to the derived channels below)
CHANNELS = ["t", "x", "y", "yaw", "vx", "vy", "speed", "yaw_rate", "ax", "ay", "steer_angle", "progress", "lateral",
            "boundary_margin", "path_err", "off_course", "cmd_steer", "cmd_throttle", "cmd_brake", "app_steer",
            "app_throttle", "app_brake", "compute_ms", "cpu_ms", "target_speed"]


# channels shipped to the dashboard: what it displays, plus a few raw ones for "show all"
SHIP = ["t", "x", "y", "yaw", "speed_kph", "target_speed", "app_throttle", "app_brake", "steer_deg", "lat_g", "lon_g",
        "grip_use", "rear_slip_deg", "path_err", "boundary_margin", "progress", "e_net_wh", "compute_ms",
        "cmd_steer", "yaw_rate", "p_batt_kw"]


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _round(a, nd=3):
    a = np.asarray(a, float)
    return [None if not math.isfinite(x) else round(float(x), nd) for x in a]


def find_track(summary: dict) -> Path:
    p = Path(summary["track"].get("path", ""))
    if p.is_file():
        return p
    name = summary["track"]["name"]
    for cand in glob.glob(str(ROOT / "tracks" / "**" / f"{name}.yaml"), recursive=True):
        return Path(cand)
    raise FileNotFoundError(f"course file for {name}")


def track_geometry(tr) -> dict:
    cones, colors, _ = tr.all_cones()
    step = max(1, int(round(1.0 / T.DS)))  # ~1 m spacing is plenty for drawing
    return {
        "closed": bool(tr.closed),
        "center": _round(tr.center[::step].ravel(), 2),
        "left": _round(tr.left[::step].ravel(), 2),
        "right": _round(tr.right[::step].ravel(), 2),
        "cones": _round(cones.ravel(), 2), "cone_colors": [int(c) for c in colors],
        "obstacles": _round(tr.obstacle_cones.ravel(), 2) if len(tr.obstacle_cones) else [],
        "start": _round(tr.cross_line(tr.s_of_progress(tr.start_progress)).ravel(), 2),
        "finish": _round(tr.cross_line(tr.s_of_progress(tr.finish_progress)).ravel(), 2) if not tr.closed else None,
        "length_m": round(float(tr.length), 1),
    }


def energy_channels(tel: dict, veh: dict, bike: dict) -> dict:
    """Road-load power and cumulative energy (Wh) from ground-truth motion."""
    m = float(veh["mass"])
    v = np.maximum(tel["vx"], 0.0)
    f_res = 0.5 * RHO * float(bike.get("drag_area", 0.55)) * v * v + float(bike.get("rolling_resistance", 0.012)) * m * G * (v > 0.05)
    f_wheel = m * tel["ax"] + f_res                 # force the tyres push with (+ drive, - braking)
    p_wheel = f_wheel * v                            # W at the wheels
    p_drive = np.maximum(p_wheel, 0.0)
    p_brake = np.maximum(-p_wheel, 0.0) * (tel["cmd_brake"] > 0.02)   # friction-brake power (coasting drag is not recoverable)
    p_batt = p_drive / ETA_MOTOR - ETA_REGEN * p_brake
    dt = np.diff(tel["t"], prepend=tel["t"][0])
    wh = lambda p: np.cumsum(p * dt) / 3600.0  # noqa: E731
    return {"p_batt_kw": p_batt / 1e3, "e_drive_wh": wh(p_drive / ETA_MOTOR), "e_regen_wh": wh(ETA_REGEN * p_brake),
            "e_net_wh": wh(p_batt), "e_brake_wh": wh(p_brake)}


def load_run(d: Path, bike: dict) -> dict:
    summary = json.loads((d / "summary.json").read_text())
    events = json.loads((d / "events.json").read_text())
    rows = list(csv.DictReader(open(d / "telemetry.csv")))[::2]      # 40 Hz physics log -> 20 Hz for display
    cols = [c for c in CHANNELS if c in rows[0]]
    tel = {c: np.array([_f(r[c]) for r in rows]) for c in cols}
    veh = summary["vehicle"]
    tel["grip_use"] = np.hypot(tel["ax"], tel["ay"]) / (float(veh["mu"]) * G)
    tel["lat_g"], tel["lon_g"] = tel["ay"] / G, tel["ax"] / G
    tel["speed_kph"] = tel["speed"] * 3.6
    tel["steer_deg"] = np.degrees(tel["steer_angle"]) if "steer_angle" in tel else tel["app_steer"] * math.degrees(veh["max_steer"])
    tel["rear_slip_deg"] = np.degrees(np.arctan2(tel["vy"] - float(veh["cg_to_rear"]) * tel["yaw_rate"], np.maximum(np.abs(tel["vx"]), 0.1)))
    tel.update(energy_channels(tel, veh, bike))

    m = summary["metrics"]
    start = next((e["t"] for e in events if e["kind"] == "start"), None)
    dist_km = max(float(tel["progress"][-1] - tel["progress"][0]), 1.0) / 1000.0
    return {
        "id": d.name, "team": summary["team"], "controller": Path(summary["controller"]["path"]).stem,
        "course": summary["track"]["name"], "backend": summary["backend"], "created": summary.get("created"),
        "vehicle": veh, "score": summary["score"], "metrics": m, "events": events, "start_t": start,
        "energy": {"net_wh": round(float(tel["e_net_wh"][-1]), 1), "drive_wh": round(float(tel["e_drive_wh"][-1]), 1),
                   "regen_wh": round(float(tel["e_regen_wh"][-1]), 1), "wh_per_km": round(float(tel["e_net_wh"][-1]) / dist_km, 1)},
        "track": track_geometry(T.load(find_track(summary))),
        "tel": {k: _round(tel[k], 4 if k in ("x", "y", "yaw") else 3) for k in SHIP if k in tel},
    }


def fresh_runs(ctrls: list[str], track_globs: list[str], jobs: int) -> list[Path]:
    """Run each controller (path:team) on each course, a few at a time: running many in
    parallel makes them compete for CPU and inflates the measured step times."""
    tracks = sorted({p for g in track_globs for p in glob.glob(str(ROOT / g))})
    todo = [(c.split(":")[0], c.split(":")[1] if ":" in c else Path(c).stem, t) for c in ctrls for t in tracks]

    def one(job):
        ctrl, team, tr = job
        subprocess.run([sys.executable, "-m", "autocross", "run", "-c", ctrl, "-t", tr, "--backend", "bicycle",
                        "--no-render", "--team", team], cwd=ROOT, capture_output=True, text=True)
        hits = sorted(glob.glob(str(ROOT / "runs" / f"*_{team}_{Path(tr).stem}")))
        print(f"  ran {team:12s} {Path(tr).stem}")
        return Path(hits[-1]) if hits else None

    with ThreadPoolExecutor(max(1, jobs)) as ex:
        return [d for d in ex.map(one, todo) if d]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="run folders (default: latest per team and course)")
    ap.add_argument("--teams", nargs="*", help="only these teams (default: all)")
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "data.js"))
    ap.add_argument("--fresh", action="store_true", help="run the controllers first (bicycle backend)")
    ap.add_argument("-c", "--controller", action="append", help="with --fresh: CTRL.py[:TEAM] (repeatable)")
    ap.add_argument("-t", "--tracks", action="append", help="with --fresh: course glob (repeatable)")
    ap.add_argument("-j", "--jobs", type=int, default=2, help="with --fresh: parallel runs (keep low)")
    a = ap.parse_args()

    if a.fresh:
        ctrls = a.controller or ["controllers/apex_planner.py:apex", "controllers/baseline_pure_pursuit.py:baseline"]
        dirs = fresh_runs(ctrls, a.tracks or ["tracks/*.yaml"], a.jobs)
    else:
        dirs = [Path(p) for p in a.runs] or [Path(p).parent for p in glob.glob(str(ROOT / "runs" / "*" / "summary.json"))]
    latest = {}
    for d in sorted(dirs):
        try:
            s = json.loads((d / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if s.get("kind") != "run" or (a.teams and s["team"] not in a.teams):
            continue
        latest[(s["team"], s["track"]["name"])] = d          # sorted by timestamp: last one wins
    bike = load_yaml(ROOT / "config" / "sim.yaml")["vehicle"].get("bicycle", {})
    runs = []
    for (team, course), d in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        try:
            runs.append(load_run(d, bike))
            print(f"  {course:22s} {team:16s} {d.name}")
        except Exception as e:  # noqa: BLE001 - one bad run should not stop the build
            print(f"  skip {d.name}: {e}")
    data = {"generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"), "runs": runs,
            "model": {"eta_motor": ETA_MOTOR, "eta_regen": ETA_REGEN, "rho": RHO, **{k: bike.get(k) for k in ("drag_area", "rolling_resistance")}}}
    Path(a.out).write_text("window.DASH = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {a.out}: {len(runs)} runs, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
