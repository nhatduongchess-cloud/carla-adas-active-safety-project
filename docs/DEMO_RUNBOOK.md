# Demo runbook — recording the portfolio demo

Purpose: a one-shot, reproducible recipe for the portfolio demo video and the honest results table. All commands are verified against the current CLI (`modules/runtime_config.py`, `run_scenarios.py`). Runs in the **stable envelope** (see `docs/B01_FAILURE_ANALYSIS.md`): Low quality, 640×360, `low-memory` profile, `async-stable`, bounded duration.

> Run everything from the project root with the project interpreter:
> `C:\Users\Admin\OneDrive\Desktop\Carla Simulator\Self-Driving-Perception`, `\.venvCarLa\Scripts\python.exe`.
> On Windows `async-stable`, `--inference-device auto` resolves to **CPU** by design (avoids the CUDA/D3D11 contention that triggers B01). Keep `auto` for the demo; `cuda` is opt-in only after a driver stress test.
> Before recording, confirm flags with `--help` — do not add flags not listed here.

## 0. Pre-flight (once per session)

```powershell
Set-Location 'C:\Users\Admin\OneDrive\Desktop\Carla Simulator\Self-Driving-Perception'

# Sanity: safety/geometry logic green, no simulator needed
.\.venvCarLa\Scripts\python.exe selftest.py            # expect 190 PASS / 0 FAIL

# Start the CARLA server (Low / 640x360 visible window)
.\launch_carla.bat

# Read-only readiness check against the running server
.\.venvCarLa\Scripts\python.exe wait_for_carla.py --timeout 30
```

Start your screen recorder (OBS or Xbox Game Bar). Capture both the CARLA window and the ADAS dashboard/BEV HUD. Keep each clip ~30 s.

## 1. Three demo clips

Each writes a machine-readable acceptance report you can cite in the README.

### Clip A — clear-road driving (custom controller)
```powershell
.\.venvCarLa\Scripts\python.exe -u chinh.py --town Town02 --seed 42 `
  --performance-profile low-memory --runtime-mode async-stable `
  --duration 30 --run-report logs\demo_clear.json
```
Show: lane tracking, route following, HUD (lane source/confidence), smooth speed under the safety cap.

### Clip B — AEB / hazard stop
```powershell
.\.venvCarLa\Scripts\python.exe -u chinh.py --hazard --town Town02 --seed 42 `
  --performance-profile low-memory --runtime-mode async-stable `
  --duration 30 --run-report logs\demo_hazard.json
```
Show: a stationary obstacle ahead, the committed brake-to-stop, the HUD naming the winning threat source (`LIDAR`/`RADAR`/`TRACKER`), and no creep into the obstacle.

### Clip C — L3 takeover → minimal-risk maneuver
```powershell
.\.venvCarLa\Scripts\python.exe -u chinh.py --driver-takeover --town Town02 --seed 42 `
  --performance-profile low-memory --runtime-mode async-stable `
  --duration 30 --run-report logs\demo_mrm.json
```
Show: the ODD monitor state, the `TOR → MRM → SAFE_STOP` transition, and a controlled stop.

**Alternative Clip C — sensor-loss reaction** (dual range-sensor loss → ODD violation → MRM), via the scenario harness:
```powershell
.\.venvCarLa\Scripts\python.exe run_scenarios.py --scenarios StationaryObjectCrossing `
  --town Town02 --seed 42 --fault lidar-radar-loss --fault-start 5 --fault-duration 4 `
  --report logs\demo_sensorloss.json
```

## 2. Honest results table (curated core suite)

The curated subset is the built-in **`core`** set = the 6 core scenarios
(`FollowLeadingVehicle`, `HardBrake`, `StationaryObjectCrossing`,
`DynamicObjectCrossing`, `ConstructionObstacle`, `CutInFrom_left_Lane`).

```powershell
# Core suite, 3 seeds (18 runs) -> portfolio results table
.\.venvCarLa\Scripts\python.exe run_scenarios.py --scenarios core `
  --town Town02 --seeds 42,1337,2026 --report logs\portfolio_core_3seed.json

# Core suite across 5 weather profiles (seed 42)
.\.venvCarLa\Scripts\python.exe run_scenarios.py --scenarios core `
  --town Town02 --weather clear --weathers clear,light_rain,heavy_rain,fog,storm `
  --report logs\portfolio_core_weather.json
```

Fault-injection choices for `--fault`: `camera-loss | radar-loss | lidar-loss | lidar-radar-loss`.
Read each report's `status`, `criteria`, collisions, reaction delay and GT clearance, and transcribe them into the README results table **with their exact config and date**. Report whatever the runs actually show — do not hand-edit a report to PASS.

## 3. Recording checklist

- [ ] `selftest.py` green captured (a quick terminal shot is good portfolio evidence).
- [ ] Clip A (clear), Clip B (AEB), Clip C (MRM or sensor-loss) recorded, ~30 s each.
- [ ] Trim + concatenate to a 90–120 s reel; caption each clip with map / weather / profile / device.
- [ ] Export `docs/demo.gif` (or an mp4 linked from the README).
- [ ] Core 3-seed + weather reports saved under `logs/`; results table updated in `README.md`.
- [ ] Note any run that fell outside the stable envelope, and label it as such.

## 4. If the server crashes mid-capture (B01)

Stop; do not retry the same heavier configuration or raise timeouts. Preserve the crash artifact (`%LOCALAPPDATA%\CarlaUE4\Saved\Crashes\...`). Re-record inside the stable envelope (shorter duration, `low-memory`, one map). This is the documented limitation in `docs/B01_FAILURE_ANALYSIS.md`, not a new bug to chase during recording.
