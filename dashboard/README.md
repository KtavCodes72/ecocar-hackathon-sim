# Telemetry dashboard

A Nominal-style control panel that replays sim runs: course map with the car and a speed-coloured
trail, a ghost car (e.g. the baseline) on the same map, live value tiles, synchronised time-series
lanes, a channel table, events, score breakdown, estimated energy, and a leaderboard.

```bash
# run apex_planner + the baseline on the public courses and build the data (about 20 s)
.venv/bin/python dashboard/build.py --fresh

# more courses / other controllers
.venv/bin/python dashboard/build.py --fresh -t "tracks/*.yaml" -t "tracks/mock/*.yaml" \
    -c controllers/apex_planner.py:apex -c controllers/baseline_pure_pursuit.py:baseline

# or build from runs you already have (latest per team and course)
.venv/bin/python dashboard/build.py --teams apex baseline

# serve it (or just open dashboard/index.html in a browser)
.venv/bin/python -m http.server 8765 --directory dashboard
```

Controls: Space play/pause, ←/→ step (Shift = 5 s), Home restart, `f` follow car. Drag on the
charts to zoom, double-click to reset; scroll/drag the map. Click events or leaderboard rows to jump.

Energy is a road-load estimate from ground truth (m·a + aero drag + rolling resistance, 90% motor
efficiency, 65% of friction-brake energy recovered by regen), not a measured value.

Logo: `dashboard/logo.svg` is a hand-traced script "A". To use an official logo file instead, save it
as `dashboard/logo.png` (white on black works best); it takes precedence over the SVG.

Keep `-j` (parallel runs) low: runs that compete for CPU report inflated step times.
