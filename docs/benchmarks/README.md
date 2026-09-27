# Benchmark artifacts

The JSON reports behind the numbers in the [main README](../../README.md#results).
They were previously only in `logs/`, which is git-ignored, so the figures could
not be checked by anyone reading the repository. They are published here
unmodified — same bytes the harness wrote.

## Evidence v2 run, 2026-09-27 — the current results

The first live run of the 2026-09-25 remediation code, commit
[`46bbaa4`](https://github.com/nhatduongchess-cloud/carla-adas-active-safety-project/commit/46bbaa48991ae607453444f30337fa247338c87c)
with `git_dirty: false`, recorded inside each file (`run.provenance`). CARLA client
and server `edf3e9f5c`, Town02, Windows 11, RTX 4070 Laptop, **CPU inference**,
synchronous mode at a fixed 0.025 s step (read back from the world after it was
applied). Reaction delay is metric **v2**, and clearance is **surface-to-surface**
(`clearance_basis: oriented_boxes` in every case). One run of each matrix.

| Artifact | Suite | Cases | Result |
|---|---|---:|---|
| [`v2_catalog_3seed.json`](v2_catalog_3seed.json) | 15 scenarios × 3 seeds, clear | 45 | **45 PASS / 0 FAIL** — gate `core_plus_catalog` **PASS** (core 18/18, catalog 45/45) |
| [`v2_core_5weather.json`](v2_core_5weather.json) | 6 core × 5 weathers, seed 42 | 30 | **29 PASS / 1 FAIL** — gate `all_planned_cases` **FAIL**: `DynamicObjectCrossing` in `storm` reacted in 1.225 s |
| [`v2_core_smoke.json`](v2_core_smoke.json) | `HardBrake`, `DynamicObjectCrossing` | 2 | 2 PASS |
| [`v2_range_loss.json`](v2_range_loss.json) | `HardBrake` with LiDAR **and** radar lost for 4 s | 1 | PASS — ODD VIOLATION → MRM → SAFE_STOP with hand brake |
| [`v2_runtime_smoke_async.json`](v2_runtime_smoke_async.json) | `chinh.py` runtime, 5 s, async-stable, 0 NPCs | — | `status: FAIL` on three latency criteria; see below |

**What held across all 78 scenario cases.** No collision recorded; zero
camera–LiDAR frame errors; zero scenario execution errors; every case's cleanup
verified; no case INVALID or ERROR. The smallest surface-to-surface clearance
was **0.88 m** (`NoSignalJunctionCrossing`, seed 2026) against a 0.25 m
criterion — the first run in which that criterion could actually fail. Five
catalog cases and one weather case were flagged `preemptive_response` (the ego was
already braking when the hazard was declared).

**`DynamicObjectCrossing` is still marginal, and its timing varies between runs.**
Today it reacted in 0.5 / 0.7 / 0.5 s in the three catalog seeds, 0.6 s in the
smoke run, and 0.7 / 0.85 / 0.75 / 0.6 / **1.225 s** across the five weathers. On
2026-09-19 the same seeds gave 0.30 / 1.225 / 1.10 s. The 45/45 therefore does
**not** mean the late reaction is fixed: one run per case cannot separate a code
effect from run-to-run variation in CPU inference timing, and the storm case still
fails at exactly the old 1.225 s. Ten-seed repeats (next section) show when
it is late and through which sensor; why is still open (G8).

**The new brake-release rule, first seen live.** In `v2_range_loss.json` the ego
spent 9.4% of the run in `BRAKE_HOLD_NO_DATA` — the latched brake held while the
LiDAR data was missing, instead of releasing — then the ODD monitor declared a
VIOLATION (range loss event count 1), the MRM ran and ended in `SAFE_STOP` with
brake 1.0 and hand brake. The control status records the MRM's 1.0 winning over
the AEB's 0.7 hold request.

**Comfort numbers show why the magnitude filter is not an impact detector.**
Every braking case has samples above the 12 m/s² cap (raw maximum 27.1 m/s²,
282 samples in the catalog) while the collision sensor recorded no contact at
all. The old field name `impact_decel_spikes` would have called these impacts.

**Latency on CPU inference misses its targets.** Per-case perception p95 ranged
47–122 ms against a 50 ms target. The safety loop p99 stayed under 25 ms in every
catalog case (worst 19.6 ms) but reached 33.9 ms in one weather case; overall
89 of 60,000 safety-loop samples missed the 25 ms deadline. The scenario harness
does not gate on these; they are reported, not passed.

**Runtime smoke (`chinh.py`, 5 s).** The measured control window delivered
**39.7 Hz** (200 unique control updates in 5.03 s wall, simulated 5.03 s,
real-time factor 0.999), control-interval p50 24.8 ms, p99 51.5 ms, max 83.1 ms,
3 deadline misses (interval > 50 ms). This is the first measured control rate
in the project, and it is five seconds on an empty road with 0 NPCs, headless,
`low-memory` profile — a smoke test, not a qualification. The run is still
`FAIL`: safety-control p99 33.8 ms > 25 ms, neural inference p95 69 ms > 50 ms,
inference age p95 157 ms > 150 ms. Cleanup verified; no collision; AEB never
triggered (`decision_trace`: 200 × DRIVE).

**Not in these files:** GPU inference, other towns, a curved-road MRM, the
near-field blind-zone case (G9), or anything on a real vehicle. Repeats of
`DynamicObjectCrossing`, single-sensor faults and a soak attempt follow.

## Follow-up runs, 2026-09-27 evening (commit `b3bba41`)

Same machine and CARLA build as the v2 run. Commit `b3bba41` only added the v2
reports, docs and a test, so the runtime code is the one measured above; each
scenario report records `git_commit: b3bba41…`, `git_dirty: false`. The runtime
(`chinh.py`) report does not yet carry a provenance block, so the soak's commit is
not recorded in its file.

| Artifact | Suite | Cases | Result |
|---|---|---:|---|
| [`v2_doc_clear_10seed.json`](v2_doc_clear_10seed.json) | `DynamicObjectCrossing`, seeds 1–10, clear | 10 | **8 PASS / 2 FAIL** — seeds 1 and 9 reacted in 1.225 s |
| [`v2_doc_storm_10seed.json`](v2_doc_storm_10seed.json) | `DynamicObjectCrossing`, seeds 1–10, storm | 10 | **7 PASS / 3 FAIL** — seeds 1, 3 and 4 reacted in 1.225 s |
| [`v2_fault_lidar_loss.json`](v2_fault_lidar_loss.json) | `HardBrake`, LiDAR lost 5–9 s | 1 | PASS — ODD stayed NORMAL, no MRM |
| [`v2_fault_radar_loss.json`](v2_fault_radar_loss.json) | `HardBrake`, radar lost 5–9 s | 1 | PASS — ODD stayed NORMAL, no MRM |
| [`v2_fault_camera_loss.json`](v2_fault_camera_loss.json) | `HardBrake`, camera lost 5–9 s | 1 | PASS — ODD stayed NORMAL, no MRM |
| [`v2_soak_5min_20veh.json`](v2_soak_5min_20veh.json) | `chinh.py`, 300 s target, 20 NPC vehicles, async-stable | — | **`FAIL`** — stopped at 51 s on a LiDAR read timeout; cleanup not verified |
| [`v2_soak_decision_trace.json.gz`](v2_soak_decision_trace.json.gz) | the soak's decision trace (gzip) | — | last 512 of 1,213 brake frames |
| [`v2_soak_telemetry.csv`](v2_soak_telemetry.csv) | the soak's 5 Hz telemetry (time, position, speed, tracked objects) | — | — |

No collision, frame error, INVALID or ERROR case in the 23 scenario runs; cleanup
verified in each.

### `DynamicObjectCrossing`: the late reactions have one signature

Across all 29 runs of this scenario on the v2 code (3 catalog, 5 weather, 1
smoke, 20 repeats), the reaction splits cleanly by the source that won the
brake decision (`source_at_reaction`):

| Reacting source | Runs | Reaction delay | Result | Surface clearance |
|---|---:|---|---|---|
| camera tracker | 23 | 0.50–0.95 s | all PASS | 2.31–3.87 m |
| radar | 6 | **1.225 s in every run** (frame 49, 8.23 m, TTC 0.99 s) | all FAIL | 2.02–2.26 m |

So the constant 1.225 s is not a random slow reaction: it is the moment the
geometric (radar/LiDAR) path detects the pedestrian inside the driving corridor,
and it is the same in every run because the scenario geometry is. The 1.0 s budget
is met only when the camera tracker triggers the brake earlier; in 6 of 29 runs
it did not. Late runs occurred 2/10 in clear and 3/10 in storm — with ten runs
each that is not evidence of a weather effect — and not on a fixed set of seeds
(only seed 1 was late in both). **Why the tracker missed in those runs is not
diagnosed**: the scenario reports carry no per-frame trace. Even the late runs
stopped with 2.0–2.3 m surface clearance and no contact; that is a measurement of
these runs, not a safety claim. In every earlier published run, a radar-decided reaction was
also exactly 1.225 s (`head_*` files); on that older code the tracker was
sometimes late too (1.10 s twice in `head_doc_probe.json`).

### Single-sensor loss: what these three runs do and do not show

All three pass with the same numbers (reaction at frame 5 via radar, 0.0 s delay,
5.78 m clearance) because `HardBrake`'s hazard fires at 0.125 s and the fault
window starts at 5 s. They show that losing one sensor while stopped **does not
release the latched brake and does not raise an ODD violation or an MRM** (range
redundancy kept, as `fault_acceptance` requires). They do **not** show detection
with a sensor missing; the next three runs do. In the radar-loss run the
arbiter spent 0.2% of frames in `NORMAL` and 0.2% in `BRAKE_TO_STOP` (about two
frames each); not investigated.

### Single-sensor loss from the start (commit `adec9c4`)

The same `HardBrake` case with the sensor absent for the whole 20 s
(`--fault-start 0 --fault-duration 20`), so the hazard has to be detected
without it. Commit `adec9c4` (clean) differs from `b3bba41` only in docs,
reports and a test.

| Sensor lost for the whole run | Brake decided by | Reaction | Surface clearance | Min TTC | ODD | Result | Report |
|---|---|---|---:|---:|---|---|---|
| LiDAR | radar | 0.0 s (frame 5) | 5.78 m | 1.33 s | NORMAL, `range_redundancy_degraded` | PASS | [`v2_fault_lidar_loss_from_start.json`](v2_fault_lidar_loss_from_start.json) |
| radar | LiDAR | **0.15 s** (frame 11, 9.2 m) | **3.04 m** | 0.13 s | NORMAL, `range_redundancy_degraded` | PASS | [`v2_fault_radar_loss_from_start.json`](v2_fault_radar_loss_from_start.json) |
| camera | radar | 0.0 s (frame 5) | 5.77 m | 0.52 s | NORMAL | PASS | [`v2_fault_camera_loss_from_start.json`](v2_fault_camera_loss_from_start.json) |

Either range sensor on its own detected the braking lead vehicle in this
scenario. Without the radar, the LiDAR path reacted six frames later and the car
stopped 2.7 m closer (3.04 m against 5.78 m with the radar), with a minimum TTC of
0.13 s. Without the camera nothing changed, because the radar decides this case
anyway. In all three the ODD stayed `NORMAL` for the entire run: one healthy
range sensor counts as within the ODD by design (R6a), and the health record only
flags `range_redundancy_degraded`. So a sustained single-sensor loss raises no
takeover request; that is a design choice to review, not something these runs
validate. One scenario, one seed, clear weather.

### Soak attempt: stopped at 51 s

- **Stop.** After 2,051 control updates (51.3 s simulated) of the planned 300 s,
  `chinh.py` stopped on `TimeoutError: async-stable timeout chờ LiDAR geometry`.
  This is the first live exercise of the timeout path (R9c): the loss was recorded
  and the safe-stop command was sent (`sensor_loss_fallback.command_sent: true`).
  Whether the vehicle then stopped is not recorded.
- **Cleanup failed.** Ten `actor.destroy` calls returned false, the rest reported
  "server unavailable or budget exhausted", and the 20 s cleanup budget ran out.
  That is consistent with the server hanging or crashing (see B01); the report
  cannot say which. The run correctly reports `FAIL` with
  `cleanup_verified: false`.
- **Until the stop.** Delivered control rate **38.5 Hz** over 53.3 s of active
  window with 20 NPC vehicles (real-time factor 0.962; the ≥ 38 Hz criterion
  passed), control interval p50 24.8 / p95 35.1 / p99 45.2 / max 69.5 ms, 10
  intervals over 50 ms. No collision, no frame error, sensor availability ≥ 99.9%.
- **Latency.** Safety-control p99 44.3 ms, with 734 of 2,051 samples (36%) over
  25 ms; neural inference p95 99 ms against 50 ms; inference age p95 5.1 s — not
  explained.
- **Driving.** The telemetry shows the ego moved 3.4 m, stopped at 5.6 s, and
  stayed stopped until 31.4 s with a radar target about 22 m ahead in its lane at
  zero closing speed (`BRAKE_TO_STOP`); it then drove about 165 m, up to
  34.5 km/h. The retained trace covers the last 12.8 s of that stop. The camera
  tracker reported a tracked object in 9 of 342 telemetry samples, and the learned
  lane was selected in 0 of 2,051 frames (map fallback throughout).

## Run on HEAD, 2026-09-19 (superseded by the v2 run above)

Re-run against a live CARLA server (build `edf3e9f5c`, client `edf3e9f5c`,
Town02, RTX 4070 Laptop) on the current code, after the F01–F12 review fixes.

| Artifact | Suite | Cases | Result |
|---|---|---:|---|
| [`head_catalog_3seed.json`](head_catalog_3seed.json) | Full catalog × 3 seeds, clear | 45 | **43 pass / 2 fail** — catalog sub-target ≥43/45 met, but the **acceptance gate recorded in the file is FAIL**: core is 16/18 against 18/18 required |
| [`head_core_5weather.json`](head_core_5weather.json) | Core × 5 weather profiles | 30 | **28 pass / 2 fail** |
| [`head_core_seed42.json`](head_core_seed42.json) | Core, seed 42, clear | 6 | **6 pass / 0 fail** |
| [`head_doc_probe.json`](head_doc_probe.json) | `DynamicObjectCrossing` × 3 seeds × 3 weathers | 9 | 3 pass / 6 fail — see below |
| [`pre_f02_doc_heavyrain.json`](pre_f02_doc_heavyrain.json) | Same scenario on the **pre-fix** commit `47d9274` | 3 | 0 pass / 3 fail — attribution evidence |

**Zero collisions recorded and zero camera–LiDAR frame errors in all 90 runs**
(45 + 30 + 6 + 9, the first four files; the pre-fix file is extra). The files
overlap — the same recipes, seeds and weathers recur — so these are not 90
independent situations. Every failure above is one criterion — reaction delay —
on one scenario.

**Definitions these files use (all superseded 2026-09-25, files unchanged).**
Reaction delay is metric **v1** (first non-NORMAL state after trigger, clamped at
zero relative to the hazard); `scenario_min_clearance_m` is **centre-to-centre**;
cases have no PASS/FAIL/INVALID/ERROR status and no cleanup verdict; the gate
counts rows against a fixed 18/45. See
[`../METRIC_DEFINITIONS.md`](../METRIC_DEFINITIONS.md). A report produced by the
current harness is **not directly comparable** on reaction delay and clearance and
must be published as a new file, not written over these. The worst per-case
perception p95 in the catalog file is **132 ms** against a 50 ms target (CPU
inference); it is recorded in the file and was not surfaced before.

### The one failing scenario, and why it is not a regression

`DynamicObjectCrossing` misses the `reaction_delay_s <= 1.0` budget about half
the time, at **1.225 s** (49 frames at 0.025 s). It brakes and it never collides;
it is late against the stated budget. (This paragraph previously added "clearance
stays above 3.1 m". That number matched nothing in the data - the smallest recorded
value is 4.26 m - and every recorded value is centre-to-centre anyway; see the
evidence review below.)

It is not caused by the F01–F12 changes, and that was tested rather than assumed.
Running the same scenario on the pre-fix commit `47d9274` gives **identical**
numbers — 1.225 s, frame 49, all three seeds:

| seed | pre-fix `47d9274` | HEAD `feed6cf` |
|---|---|---|
| 42 | 1.225 s, frame 49, FAIL | 1.225 s, frame 49, FAIL |
| 1337 | 1.225 s, frame 49, FAIL | 1.225 s, frame 49, FAIL |
| 2026 | 1.225 s, frame 49, FAIL | 1.225 s, frame 49, FAIL |

Nor is it weather-specific, which the first weather matrix made it look like.
Across seeds it straddles the threshold in **clear** weather too:

| weather | reaction delay by seed 42 / 1337 / 2026 |
|---|---|
| clear | 0.30 s ✅ · 1.225 s ❌ · 1.10 s ❌ |
| light_rain | 1.10 s ❌ · 0.60 s ✅ · 0.55 s ✅ |
| heavy_rain | 1.225 s ❌ · 1.225 s ❌ · 1.225 s ❌ |

So this is a genuinely marginal scenario on this build, and a single-seed run can
pass or fail it by luck. That is a real open finding about the system, recorded
in [`../ARCHITECTURE.md` §12](../ARCHITECTURE.md#12-known-architectural-gaps),
not a number to tune away.

### Why the older reports show 30/30 and this one shows 28/30

The 2026-09-01 reports record `hazard_frame: null` and `reaction_delay_s: 0.0`
for **every** case, including this scenario. The harness at that time had no
hazard origin to measure from, so the `reaction_delay_s <= 1.0` criterion was
satisfied trivially, always — it could not fail. The current harness measures
from `hazard_frame` (`modules/scenario_library.py:32-38`), which is why a
criterion that was previously inert now bites.

Read plainly: **part of the older 30/30 and 45/45 was measured with one of the
five acceptance criteria effectively switched off.** The newer 43/45 and 28/30
are the stricter numbers, and they are the ones to quote.

## Evidence review, 2026-09-25

An external review arrived as a script
([`scripts/claude_evidence_review_20260925.py`](../../scripts/claude_evidence_review_20260925.py)):
fourteen offline probes, each reproducing one suspected defect against the code,
plus a re-read of the published benchmark files. None of it connects to CARLA. It
was run before any change was made, and again after; both outputs are kept here
unedited as [`evidence_review_20260925_before.json`](evidence_review_20260925_before.json)
and [`evidence_review_20260925_after.json`](evidence_review_20260925_after.json).
Everything it found is listed here, including what was not fixed.

The `reported_clearance_is_center_distance` probe still prints 3.0 after the fix,
because it calls `_min_dist`, which is the centre distance used to *trigger*
scenarios and was deliberately left alone. Acceptance now reads
`min_clearance_m`; for the same geometry that is 1.0 m, basis `radius`.

### What the probes showed, and what changed

| Probe | Before | After | |
|---|---|---|---|
| ODD with all inputs NaN | `NORMAL` | `VIOLATION`, reason `odd_unmeasurable:…` | fixed |
| Takeover during ODD violation | `TOR → L3_ACTIVE → TOR` every tick; a one-shot takeover started an MRM 10.03 s later with the driver driving | `TOR → DRIVER_CONTROL`, stays until ODD is NORMAL | fixed |
| AEB requests 1.0 during an MRM | vehicle receives **0.167** | vehicle receives **1.0** | fixed |
| Scenario verdict, NaN delay + NaN clearance | pass | fail, two reasons | fixed |
| Scenario verdict, negative delay | pass | fail, timestamp fault | fixed |
| Scenario verdict, no collision count | pass | fail, "not observed" | fixed |
| L3 profile with 5 late-braking events | pass | fail — late braking is now a gate | fixed, **stricter criterion** |
| 45 copies of one case | gate `PASS`, "45/45" | gate `INVALID`, 44 duplicate rows | fixed |
| Empty suite | `NOT_EVALUATED`, but "no collision: true" | `NOT_EVALUATED`, "no collision: null" | fixed |
| KPI summary of zero frames | `success: true` | `success: false` | fixed |
| LiDAR lost with radar disabled | range loss **not** detected; disabled radar reported 100% available | range loss detected → ODD VIOLATION; disabled radar reports `null` | fixed |
| Scenario clearance | centre-to-centre: 3.0 m where the surface gap is 1.0 m | surface-to-surface, basis recorded | fixed for acceptance; trigger distance deliberately unchanged |
| Brake latch after critical stop, then empty corridor | releases to `DRIVE` after 5 frames, brake drops 1.0 → 0.7 while holding | unchanged | **not fixed** - see below |
| Friction at weather extremes | μ 0.9 → 0.4 | unchanged | not a defect - see below |
| Published benchmark gates | catalog 43/45 described as "met" | gate recorded as FAIL now stated | documentation corrected |

The new acceptance gate gives **identical verdicts** on all four published HEAD
reports; it only changes the manufactured cases above. That is checked by
`tests/test_evidence_review.py::AcceptanceGateTests::test_published_reports_keep_their_verdicts`.

### What this means for the published numbers

- **Clearance was never tested.** Every `scenario_min_clearance_m` in every file
  here is a centre-to-centre distance. For a vehicle target the 0.25 m criterion
  could not fail. The collision sensor, which is independent of this, recorded zero
  contacts in all 90 HEAD runs. The next live run will be the first to test clearance.
- **All HEAD runs used CPU inference**, recorded in each file as
  `inference_device: cpu` with the reason "safe Windows CARLA mode (avoid CUDA/D3D11
  contention)". No published scenario result was produced with GPU inference.
- **The demo run has two frame rates that disagree.** `fps_ema` is 44.9; frames
  divided by the reported wall duration is 20.0. The wall duration is sampled when
  the report is written and includes start-up and teardown, so it is not an
  isolated control-loop window either. Neither number qualifies real-time
  operation, which remains future work (L4).

### Not fixed, and why

**Brake latch release — partly fixed in the follow-up.** After a critical brake,
five empty-corridor frames released the latch, and the brake dropped from 1.0 to
0.7 while holding. The follow-up remediation (below) makes release depend on a
clear corridor *observed on valid LiDAR frames* for 0.1 s — the same cadence at
40 Hz — holds the latched level, and never releases on missing data. What remains
is the near-field blind zone: an object there produces a valid empty corridor.
Not run on the simulator. Gap **G9**.

**Friction range.** The weather model gives μ between 0.9 (dry) and 0.4 (soaked),
which is right for asphalt; CARLA models no ice or standing water. The consequence
is that the ODD monitor's μ < 0.3 violation branch cannot be reached from weather in
this simulation. Lowering the friction floor to reach it would be tuning the data to
hit a threshold. Recorded as open gap **G11** instead.

**Found by reading, not by a probe.** The MRM command sets brake and hand brake
only; steering is released to zero. On a curve the vehicle would brake in a straight
line. Not reproduced, recorded as open gap **G10**.

## Historical artifacts

| Artifact | What it is | Generated | Cases | Result |
|---|---|---|---:|---|
| [`catalog_clear_3seed_final_report.json`](catalog_clear_3seed_final_report.json) | Full catalog, clear, 3 seeds | 2026-09-01T20:57 | 45 | 45/45 — reaction criterion inert, see above |
| [`core_5weather_report.json`](core_5weather_report.json) | Core × 5 weather profiles | 2026-09-01T21:07 | 30 | 30/30 — same caveat |
| [`core_clear_3seed_report.json`](core_clear_3seed_report.json) | Core, clear, 3 seeds | 2026-09-01T20:19 | 18 | 18/18 — same caveat |
| [`demo_clear.json`](demo_clear.json) | Single live runtime report | 2026-09-12 | — | **`status: FAIL`** — see below |
| [`../scenario_baseline_report.json`](../scenario_baseline_report.json) | The before-picture, kept deliberately | 2026-08-04T20:27 | 3 | 2 pass / 1 fail, 18 collision events |

## Acceptance criteria, embedded in each scenario report

```
scenario_triggered   == true
collisions           == 0
reacted              == true after trigger
reaction_delay_s     <= 1.0
min_clearance_m      >= 0.25 when measurable
sensor_frame_errors  == 0
cut-in               must command BRAKE or EVADE
```

A verdict is the AND of three independent assessors — scenario, fault behaviour
and weather behaviour — so a case cannot pass by satisfying only the easy one.

## What these artifacts do NOT record

Stated because the alternative is letting a reader assume it was captured.

- **No file here embeds its commit SHA, CARLA server build, client version or
  hardware.** The 2026-09-01 reports predate the `run` block entirely. The
  2026-09-19 `head_*` reports have one, with town, seeds, weathers, fault,
  inference device and thresholds — but not the SHA, the build or the machine.
  (This section previously said the current harness embeds all of it; it does
  not.) The build `edf3e9f5c` and the RTX 4070 Laptop come from the project log,
  not from inside the files, and no SHA is inferred from the `head_` name.
- From 2026-09-25 the scenario runner writes a `run.provenance` block: git commit,
  dirty flag and a hash of the uncommitted diff, CARLA client and server versions,
  platform, Python, torch/CUDA/GPU name, model file hashes, and the world settings
  read back after they were applied. Anything it cannot read is `null` with a
  reason. CPU model and RAM are not captured (only the platform string). No file
  in this folder was produced with it yet.

## `demo_clear.json` — read the status field

The main README describes a 121.5 m clear-road drive. That is accurate as far as
it goes, and incomplete, so here is the rest of the file:

| Field | Value |
|---|---|
| `status` | **FAIL** |
| `stop_reason` | `error` |
| `error` | `TimeoutError: async-stable timeout chờ LiDAR geometry` |
| `cleanup.verified` | **false** — ten `actor.destroy` steps FAIL, the rest UNKNOWN after the 20 s budget |
| `driving.collisions` | 0 |
| `driving.distance_travelled_m` | 121.541 |
| `driving.max_speed_kmh` | 35.788 |
| `run.simulated_duration_s` | 22.175 (target 30.0) — this is frames / 40, **not** simulated time in async mode; the field is now called `frames_x_configured_step_s` |
| `safety.aeb_triggered` | **true** — 310 of 887 frames commanded BRAKE |
| `criteria.safety_p99_le_25ms` | false |
| `criteria.perception_p95_le_50ms` | false |
| `criteria.inference_age_p95_le_150ms` | false |

The `TimeoutError` above is the async LiDAR wait. At the time it fell into the
generic error handler: sensor health never recorded the loss and no stop was
attempted before teardown. Since 2026-09-25 that path records the loss, attempts a
safe stop and records whether the command was sent (`safety.sensor_loss_fallback`
in the report). Not yet exercised live.

So: zero collisions and clean path-following, on a run that the runtime itself
marks as failed, that ended early on the B01 native fault, whose teardown could
not be verified, where AEB was active for roughly a third of the frames, and
where three latency criteria were not met. The fail-closed design working as
intended — an unverified teardown fails the run — is the reason this file says
FAIL rather than a reason to quote the distance on its own.

## Reproducing

Needs a CARLA server. Check `--help` on HEAD first; the harness has gained
options since these files were written.

```bash
python run_scenarios.py --town Town02 --scenarios all  --seeds 42,1337,2026 --weather clear --seconds 20 --inference-device cpu --report logs/evidence_v2/catalog_3seed.json
python run_scenarios.py --town Town02 --scenarios core --seed 42 --weathers clear,light_rain,heavy_rain,fog,storm --seconds 20 --inference-device cpu --report logs/evidence_v2/core_5weather.json
python run_scenarios.py --town Town02 --scenarios HardBrake --seed 42 --weather clear --seconds 20 --fault lidar-radar-loss --fault-start 5 --fault-duration 4 --inference-device cpu --report logs/evidence_v2/range_loss.json
python chinh.py --town Town02 --vehicles 0 --seed 42 --duration 5 --no-display --performance-profile low-memory --runtime-mode async-stable --inference-device cpu --run-report logs/evidence_v2/smoke_async.json
```

Since 2026-09-25 the exit code follows the gate (0 PASS, 1 FAIL, 2 INVALID, 3
NOT_EVALUATED). Still read the JSON: `suite.not_run` and `run.partial` show a run
that never completed because the process crashed.
