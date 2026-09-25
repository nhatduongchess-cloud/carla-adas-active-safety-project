# Metric definitions

Every number this project publishes, with its source, clock, window,
denominator, validity rule and known biases. Written 2026-09-25 as part of the
evidence remediation. If a report field is not defined here, it is not an
acceptance metric.

**Clocks.** Three clocks exist and are never mixed:

- **sim** — CARLA simulation time, from sensor `timestamp` fields (seconds).
- **wall** — `time.perf_counter()` in the client process (seconds), for rates
  and durations of the Python loop.
- **mono** — `time.monotonic()`, for ages of asynchronous results.

A simulator timestamp is never subtracted from a wall timestamp.

**Missing data.** A metric that was not measured is `null` with a status of
`NOT_EVALUATED`, never `0`. Reports are written with `allow_nan=False`; NaN and
Infinity are normalised to `null` before writing (`modules/l3_report.py`,
`modules/runtime_report.py`).

**Historical reports.** Files in `docs/benchmarks/` written before 2026-09-25 use
the older definitions noted per metric below. They are kept unedited; they are not
re-rendered with new definitions.

---

## Safety metrics

### Collisions — `collisions`
- **Source:** `modules/collision_sensor.py`, a CARLA collision sensor on the ego.
- **Counting:** events from the sensor, debounced per other actor (a contact within 1 s
  of the previous one with the same actor is the same episode) by the sensor
  wrapper; the KPI keeps the maximum running count seen.
- **Validity:** the count must be a non-negative whole number and the sensor must
  have existed for the run. A missing or invalid count makes the case **INVALID**
  (`scenario_acceptance.assess_scenario`), never zero. An exception row reports
  `null`, not `-1` (older rows used `-1`).
- **Biases:** zero collisions over *N* runs of a scenario is an observation about
  those runs; it is not evidence that collisions are impossible. Runs across seeds
  of the same recipe are correlated.

### Reaction latency — `reaction_delay_s` (metric v2 from 2026-09-25)
- **Source:** `run_scenarios.py` + `scenario_acceptance.reaction_timing`.
- **Origin:** `hazard_frame`, set by the scenario oracle when its reaction
  condition first holds after trigger (default: the trigger frame itself).
- **Response:** the first frame at or after `hazard_frame` whose safety **decision**
  state is not `NORMAL`.
- **Value:** `(response frame − hazard frame) × Δt_sim`, Δt = 0.025 s (the runner
  is synchronous). Never clamped.
- **Preemptive response:** a reaction that began before the hazard and was still
  active on the hazard frame scores 0.0 s with `preemptive_response: true`. A
  reaction that began and ended before the hazard is **not** a response.
- **Validity:** finite and ≥ 0; otherwise the case is INVALID.
- **What it is not:** it is decision latency. It is not the time the command was
  applied, nor when the vehicle measurably decelerated. It does not tie the
  reaction to the hazard actor: an unrelated `FOLLOW` state that is active counts.
- **v1 (all reports up to 2026-09-19):** measured from the first non-NORMAL state
  after *trigger* and clamped with `max(0, …)`, so a pre-hazard reaction that had
  ended was scored as an instant response. Reports before 2026-09-19 also had no
  hazard origin at all (`hazard_frame: null`, delay always 0.0).

### Clearance — `scenario_min_clearance_m` + `clearance_basis`
- **Source:** `modules/clearance.py`, via `RunningScenario._min_clearance`.
- **Definition (from 2026-09-25):** minimum 2D surface-to-surface gap between the
  ego's and each scenario actor's oriented bounding-box footprints, over all
  ticks. Overlap scores 0.0 (not negative).
- **Basis:** `oriented_boxes` when both boxes are readable; `radius` or `center`
  fallbacks otherwise. The weakest basis seen in a run is recorded.
- **Validity:** actor states that could not be read are counted
  (`measurement_errors`) and make the case INVALID.
- **Limits:** 2D footprint in the road plane — not exact 3D clearance; no elevation
  gating (bridges, slopes).
- **Historical:** every published report before 2026-09-25 recorded the
  **centre-to-centre** distance in this field, now reported as
  `scenario_min_center_distance_m`. For a vehicle target that value cannot fall
  below 0.25 m, so the criterion was inert.

### Late braking — `late_braking_events` (L3 profiles)
- **Source:** `evaluate_l3.py`.
- **Definition:** count of brake **onsets** (decision `BRAKE` rising edges) whose
  TTC at onset is below 0.8 s. One per onset, not per frame.
- **Gate:** any late braking event fails the profile (from 2026-09-25; before, it
  was recorded but not gated).
- **Denominator:** not normalised; the report gives the count per profile run.

### False disengagement — `false_disengagements` (L3 profiles)
- **Definition:** transitions out of `L3_ACTIVE` while the weather ODD is `NORMAL`.
  Counted on the transition, not per frame.
- **Known bias:** a legitimate fallback caused by sensor loss in clear weather would
  be counted here; `evaluate_l3.py` does not model sensor health, so it cannot
  happen there, but the definition is not health-aware.

### TOR / takeover / MRM events
- **Source:** `L3StateMachine.tor_events`, `takeover_events`, `mrm_events`.
- **Counting:** state transitions into `TAKEOVER_REQUEST`, `DRIVER_CONTROL`,
  `MRM_EXECUTING`.
- **Clock:** the TOR window advances by the Δt passed in, which is simulation time
  in both `chinh.py` (validated sim step) and the harnesses.
- **`human_takeover_verified`:** always `false`. The takeover input is a simulated
  flag.

### False AEB
- **Not measured.** No independent clear-path oracle and no negative-exposure
  duration exist yet; "frames commanding BRAKE / total frames" is **not** a false
  AEB rate and is not reported as one.

---

## Control and timing metrics (`modules/control_timing.py`)

### `configured_step_s`
The CARLA fixed delta the run requested (`config.FIXED_DELTA`, 0.025 s → a 40 Hz
target). A configuration, not a measurement.

### `active_control_wall_s`
Wall time from the start of the control loop to loop exit. Excludes model
loading, warm-up and teardown. The window is closed before cleanup runs, and the
first close wins.

### `unique_control_updates`, `delivered_control_hz`
Control commands sent for distinct world frames inside the window, and that
count divided by `active_control_wall_s`. A repeated world frame is not a new
update. `NOT_EVALUATED` below two updates.

### `simulated_elapsed_s`, `real_time_factor`
Last minus first LiDAR simulation timestamp over the window; divided by
`active_control_wall_s`. Timestamps that go backwards are counted
(`sim_time_regressions`).

### `control_interval_ms` (count, p50, p95, p99, max) and `deadline_misses`
Wall intervals between consecutive control updates. A deadline miss is an
interval longer than **2 target periods** (50 ms at 40 Hz), i.e. at least one
whole update missed. Percentiles are reported with their sample count; a p99 over
a few dozen samples is not a strong statistic.

### Acceptance: `delivered_control_hz_ge_95pct_target`
Passes only with a `MEASURED` window and `delivered_control_hz ≥ 0.95 × 40`.
Missing measurement fails.

### `hud_fps_ema`
Exponential moving average (α = 0.1) of instantaneous loop rate, for the HUD.
**Never an acceptance number.** Until 2026-09-25 it was, as
`real_time_loop_ge_38hz`. The published demo shows why: `fps_ema` 44.9 while
frames / wall time was 20.0.

### `frames_x_configured_step_s`, `wall_since_loop_start_incl_teardown_s`
Renamed from `simulated_duration_s` and `wall_duration_s`. The first is frames ×
0.025 s — in async mode **not** simulated time. The second includes teardown and
is sampled when the report is written.

### Simulation step for the algorithms (`SimClock`)
In async mode a slow loop skips world frames. The step passed to the tracker,
the safety layer and the TOR window is the validated difference of LiDAR
simulation timestamps: `ok` for 0 ≤ Δt ≤ 0.25 s; a longer `gap` is not integrated
as one step (the configured step is used and the closing-speed history is reset);
`invalid` (backwards/non-finite) is passed to the L3 machine as NaN, which fails
toward the MRM.

### Stage latencies — `safety_p99_ms`, `perception_p95_ms`, `inference_age_p95_ms`
- **Source:** `modules/pipeline_metrics.py`; boundaries are stated in the report's
  `measurement_semantics` block.
- **Not end-to-end:** stage percentiles are not summed into an end-to-end latency.
  `sensor_to_control_wall_ms` (callback receipt → command RPC return, same trace
  ID) is **not implemented yet**; callback-receipt timestamps are not propagated
  for the camera.
- **Deadlines:** safety 25 ms, perception 50 ms, result age 150 ms
  (`config.py`). The catalog run of 2026-09-19 had a worst-case perception p95 of
  132 ms on CPU inference.

---

## Comfort metrics (`modules/kpi.py`)

- **Derivation:** decel = −Δv / Δt_configured between consecutive samples; jerk =
  Δdecel / Δt. Δt is the configured step (the scenario runner is synchronous).
- **Filtered (comfort) values:** `max_decel_ms2`, `max_jerk_ms3` exclude samples
  above 12 m/s² and |60| m/s³. The cap is a **magnitude** filter, not matched to
  collision timestamps.
- **Raw values, reported alongside:** `raw_max_decel_ms2`, `raw_max_abs_jerk_ms3`,
  `decel_samples_above_cap`, `jerk_samples_above_cap`. `impact_decel_spikes` is the
  historical name for `decel_samples_above_cap`; a sample above the cap is not
  evidence of an impact.
- **Mode:** emergency braking legitimately exceeds comfort limits; comfort
  numbers are not separated by mode yet.
- **Zero frames:** `status: NOT_EVALUATED`, `success: false`, `mean_speed_kmh:
  null`.

---

## Case and suite status

| Status | Meaning |
|---|---|
| `PASS` | All mandatory observations present and valid; every gate met; cleanup verified. |
| `FAIL` | Evidence that a requirement was not met. Kept even when data is also missing. |
| `INVALID` | Setup, oracle or measurement broken; the case cannot be judged. |
| `ERROR` | Exception / RPC / server failure, or cleanup failure after an otherwise passing case. |
| `NOT_RUN` | Planned but not attempted. |
| `NOT_EVALUATED` | Suite scope incomplete or metric unmeasured. |

Suites report `planned, attempted, completed, valid, passed, failed, invalid,
error, not_run` and two explicit pass-rate denominators (passed/planned,
passed/attempted). The 45-case catalog uses the project's `core_plus_catalog`
policy (core 18/18 and catalog ≥ 43/45 and no collision); every other matrix
uses `all_planned_cases` (every planned case PASS). `run_scenarios.py` exits 0
only on an overall PASS (1 FAIL, 2 INVALID/broken matrix, 3 NOT_EVALUATED,
4 report not written).

## Weather-derived conditions

`visibility_m`, `mu`, `snr` come from `modules/weather_model.py`: heuristic
proxies computed from CARLA `WeatherParameters`. They are not measured
visibility, not tyre–road friction (CARLA's `tire_friction` is a separate physics
attribute this project does not set), and `snr` is a 0–1 reliability proxy, not a
signal-to-noise ratio with units. The model yields μ ∈ [0.4, 0.9]; the μ < 0.3
branches are exercised only by injected-condition unit tests.

The stopping distance `d = v·t_r + v² / (2 μ g)` is a planning approximation
(flat road, ideal constant deceleration). `target_decel_ms2 / 6` is a heuristic
pedal mapping, not closed-loop deceleration tracking: an MRM "at X m/s²" means
X was requested, not achieved.
