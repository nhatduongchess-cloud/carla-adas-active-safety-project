# Evidence remediation — results, 2026-09-25 (live runs added 2026-09-27)

What was done against the remediation brief of 2026-09-25 (written for commit
`07717f7`) and its accompanying probe script
([`scripts/claude_evidence_review_20260925.py`](../scripts/claude_evidence_review_20260925.py)),
what was verified and how, and what was not done.

**Headline.** The code, tests and documentation parts were done and verified
offline on 2026-09-25, when no CARLA server was running. **The planned
simulation checks were then run on 2026-09-27** on commit `46bbaa4` (clean tree),
one run per matrix — see [Simulation verification](#simulation-verification-2026-09-27).
Only the curved-road MRM remains `NOT_RUN`.

## Commits

| Commit | Content |
|---|---|
| `1721f49` | The fourteen probes: MRM/AEB arbitration, ODD NaN, takeover oscillation, range loss with radar disabled, fail-open acceptance, duplicate/empty suites, surface clearance. `tests/test_evidence_review.py`. |
| `9ae93a2` | WP01–WP06 code: command validation, input contracts, evidence-based brake release, LiDAR-timeout fallback, takeover ownership, case/suite statuses, reaction metric v2, cleanup-final verdicts, planned matrix + exit codes, report N/A, measured control rate, simulation step. `tests/test_evidence_remediation.py`. |
| `6090545` | Test-fixture fix found by the local run (a `dict()` duplicate-key error in a new test). |
| `cf760a7` | WP07–WP10: provenance block, metric dictionary, claim register, requirement map, README / architecture / benchmark / B01 corrections. |

The brief itself (`docs/CLAUDE_CARLA_EVIDENCE_REMEDIATION_20260925.md`) is the
owner's file and was left untracked; it is not part of these commits.

## Commands run, and their output

Cloud workspace (Linux, Python 3, CARLA module stubbed only for the probe script):

| Command | Result | Exit |
|---|---|---|
| `python selftest.py` | 210 PASS / 0 FAIL | 0 |
| CI offline group (18 modules, list in `.github/workflows/selftest.yml`) | Ran 306 tests, OK | 0 |
| `ruff check .` | All checks passed | 0 |
| `mypy` | no issues in 4 source files | 0 |
| probe script, before and after | [`benchmarks/evidence_review_20260925_before.json`](benchmarks/evidence_review_20260925_before.json), [`…_after.json`](benchmarks/evidence_review_20260925_after.json) | 0 |

Development machine (Windows 11, `.venvCarLa`, CARLA client installed, **no
server**), at `cf760a7`:

| Command | Result | Exit |
|---|---|---|
| `.\.venvCarLa\Scripts\python.exe -X utf8 selftest.py` | 210 PASS / 0 FAIL | 0 |
| `… -m unittest discover -s tests -p "test_*.py" -q` | Ran 409 tests, OK | 0 |
| `… -m unittest tests.test_ego_control tests.test_sensor_runtime tests.test_runtime_report tests.test_runtime_cleanup -v` | Ran 48 tests, OK | 0 |

The 409 are unit tests (pure logic and mocked CARLA client); the 210 are
self-test checks. They are different kinds of count and are not added. Neither is
a coverage figure, and neither is a vehicle run. Before this work the numbers were
204 / 284 (brief §2.1).

## Work packages

Status words: **done** (implemented and tested offline), **partial**, **not done**.

### WP01 — AEB/MRM arbitration and valid commands — done (lateral: explicit fallback only)
- Final brake under an MRM is `max(MRM, AEB)`; the AEB request is validated;
  `throttle = 0` on every braking branch and whenever a custom command has
  `brake > 0`.
- Custom-stack commands are validated (NaN, ±inf, out of range, bool) **before**
  the RPC; an invalid command is a control fault → latched safe stop, never
  clamped and sent.
- Status records requested AEB/MRM brake, the applied command and
  `command_sent` (fire-and-forget RPC — "sent", never "applied").
- Fault latch and SAFE_STOP behaviour unchanged; autopilot is never engaged.
- **Lateral:** steer is held at 0 during AEB/MRM, now written explicitly and
  documented as a degraded fallback. Using the lateral controller while braking
  (G10) was **not done**: it changes live behaviour and needs a curved-road run.
- Tests: `tests/test_ego_control.py::EgoControllerEvidenceRemediationTests` (8),
  `tests/test_evidence_review.py::MrmAebArbitrationTests`,
  `tests/test_evidence_remediation.py::CommandValidationTests`.

### WP02 — invalid data, sensor loss, brake latch — partial
- ODD: missing key, NaN/±inf, bool and string inputs → `VIOLATION`
  (`odd_unmeasurable:<field>`), not critical. Missing SNR no longer defaults to
  a perfect 1.0. Published boundaries unchanged (tested at 19.9/20/50/50.1,
  0.299/0.3/0.599/0.6, 0.249/0.25).
- Range health: `all_enabled_range_unavailable` (= the kept key
  `range_redundancy_lost`) and `range_redundancy_degraded`; radar-disabled +
  LiDAR healthy stays normal, + LiDAR lost escalates.
- LiDAR read timeout in `chinh.py` now records the loss in health and attempts a
  safe stop, recording `command_sent` or the RPC error
  (`safety.sensor_loss_fallback` in the run report). The published demo ended on
  exactly this timeout.
- Brake latch (G9): missing/invalid LiDAR never releases it; holding never
  weakens it; release needs 0.1 s of clear corridor observed on valid frames,
  timed in seconds. Same cadence at 40 Hz as before.
- **Not done:** a single typed readout (`VALID_OBSERVATION / VALID_EMPTY /
  UNAVAILABLE / STALE / INVALID / DISABLED`) through the whole pipeline — only
  `lidar_valid` is threaded into the safety layer, and only the scenario runner
  sets it to False (the runtime's read either succeeds or raises). Duplicate /
  out-of-order frames do not yet leave health counters untouched (the camera is
  observed with a repeated frame id at 40 Hz by design, so a blanket rule would
  mark it missing). Miss thresholds are still frame counts, now reported
  alongside. The `chinh.py` timeout handler itself is not unit-tested.
- Tests: `OddInputContractTests`, `RangeHealthTests`, `BrakeReleaseEvidenceTests`.

### WP03 — takeover and ownership — done
- `TOR → DRIVER_CONTROL` on acknowledgement; repeated ACKs ignored; back to
  `L3_ACTIVE` only with an explicit engage request **and** ODD NORMAL.
- `control_owner`, `autonomy_enabled`, event counters (TOR, takeover, MRM) on
  transitions; `human_takeover_verified: false` always; the runtime report
  carries an `l3` summary.
- Invalid `dt` (negative, NaN, inf, non-number) fails toward the MRM; `dt = 0`
  is no progress; invalid speed never counts as stopped; invalid μ gives the
  gentlest MRM.
- `critical` documented as an absolute threshold, not a rate of degradation.
- After the ACK the custom controller keeps driving as the declared stand-in, and
  AEB stays active. A real manual input path does not exist.
- Tests: `TakeoverOwnershipTests` (12), selftest §8.

### WP04 — oracle, reaction latency, clearance — partial
- Clearance is 2D surface-to-surface between oriented boxes, basis recorded;
  centre distance kept under its own name and still used for triggering.
- Reaction metric v2: first reaction at or after the hazard, never clamped;
  preemptive responses flagged; a reaction that ended before the hazard does not
  count.
- Scenario tick and actor-command failures are recorded, not swallowed
  (`except: pass` removed from the tick and the four actor callbacks); unreadable
  actor states are counted. Either makes the case INVALID.
- **Not done:** the five separate timing marks (decision / applied command /
  observed vehicle response) — only the decision is timed; tying the reaction to
  the hazard actor; walker/elevation-specific geometry tests; the
  `DynamicObjectCrossing` diagnosis trace (§7.3). The latter needs live runs.
- Tests: `ReactionTimingTests`, `CaseStatusTests`,
  `tests/test_evidence_review.py::ClearanceTests`.

### WP05 — verdicts and suite reports — done
- Case status `PASS / FAIL / INVALID / ERROR`, with `failure_reasons` and
  `invalid_reasons` kept apart (an observed failure is never hidden behind
  INVALID). Invalid thresholds fail fast.
- Case verdict final only after its cleanup; destroy batches are checked
  (`apply_batch_sync`); a cleanup failure turns PASS into ERROR.
- Suites: planned matrix fixed before running; `planned / attempted / completed /
  valid / passed / failed / invalid / error / not_run`, unexpected and duplicate
  cases, two explicit denominators; `core_plus_catalog` policy for the 45-case
  catalog, `all_planned_cases` for every other matrix (the 30-case weather suite
  is judged against its 30).
- Checkpoint after every case (atomic replace); a matrix exception keeps finished
  cases and marks the rest NOT_RUN; `run_scenarios.py` exits 0 only on PASS.
- JSON written with `allow_nan=False` after normalisation.
- `evaluate_l3.py`: undeclared expected ODD is INVALID (was graded against
  itself); error rows are ERROR; tested/excluded components declared.
- `generate_validation_report.py`: N/A with case coverage, never 0.
- Found and fixed along the way: after the first commit, every L3 profile row
  collapsed into one "duplicate" case (profile rows carry `profile`, not `name`).
- **Not done:** a subprocess test of `run_scenarios.py` exit codes (the mapping is
  unit-tested; the process needs CARLA to start).
- Tests: `SuiteAccountingTests`, `CaseStatusTests`, `L3HarnessReportTests`,
  `ValidationReportGeneratorTests`, `KpiEvidenceTests`.

### WP06 — clocks and rates — partial
- `modules/control_timing.py`: measured control updates over the active window
  (closed before teardown), simulated elapsed from sensor timestamps, real-time
  factor, interval percentiles with counts, deadline misses. The runtime report's
  rate gate uses it; `fps_ema` is renamed `hud_fps_ema` and no longer gates.
- `SimClock`: the tracker, safety layer and TOR window receive the validated
  simulation step instead of the configured 0.025 s; gaps are not integrated as
  one step.
- **Not done:** callback-receipt timestamps for the camera; a same-trace-ID
  `sensor_to_control_wall_ms`; per-sensor delivered rates; telemetry on/off A/B;
  `--duration` stopping on simulated time (it still counts frames).
- Tests: `ControlTimingTests`, `SimClockTests`,
  `tests/test_runtime_report.py` (rate gate).

### WP07 — weather, friction, fusion — documentation only
- Proxies documented as proxies (`METRIC_DEFINITIONS.md`), the μ < 0.3 branches
  tested with injected values only (G11), the stopping-distance formula and the
  `/6` pedal mapping described as approximations.
- The weather-weighted fusion helper is marked **not wired into the runtime** in
  README, requirement map and claim register.
- **Not done:** renaming the `mu`/`snr` API fields; `physics_friction_config`
  fields; a metamorphic detector-independence test in the live loop; a sensing
  envelope table beyond the one line in the claim register.

### WP08 — provenance and metric dictionary — partial
- Scenario reports: `run.provenance` with git commit / dirty flag / diff hash,
  CARLA client and server versions, platform, Python, torch/GPU, model hashes,
  world settings read back after apply; unknowns null with a reason.
- `docs/METRIC_DEFINITIONS.md` created.
- KPI: raw maxima and excluded-sample counts reported next to comfort values;
  zero frames is `NOT_EVALUATED`.
- **Not done:** a manifest for `chinh.py` runtime reports; CPU/RAM capture;
  comfort metrics split by mode; collision-window-based (rather than magnitude)
  filtering; false-AEB measurement (no independent oracle).

### WP09 — harness fidelity, requirement map — done for docs, partial for code
- `docs/REQUIREMENT_MAP.md` rewritten with per-row status and untested remainder.
- `evaluate_l3.py` declares what it does and does not exercise.
- **Not done:** a shared adapter so `evaluate_l3.py` runs the same health /
  freshness path as `chinh.py`.

### WP10 — claims — done
- `docs/CLAIM_EVIDENCE_REGISTER.md` created; README intro, highlights, results
  table (90 = 45 + 30 + 6 + 9, overlapping), demo rate, limitations, engineering
  notes corrected; B01 §3b now says "consistent with", not "directly confirms".

## Simulation verification, 2026-09-27

Run by the owner on the development machine: CARLA `edf3e9f5c` (client and
server), Town02, Windows 11, RTX 4070 Laptop, CPU inference, commit `46bbaa4` with
`git_dirty: false` — all recorded in each file's `run.provenance`. Reports in
[`benchmarks/`](benchmarks/README.md#evidence-v2-run-2026-09-27--the-current-results).

| Check | Report | Result |
|---|---|---|
| Runtime smoke, async, 5 s | `v2_runtime_smoke_async.json` | `FAIL` (safety p99 33.8 ms, perception p95 69 ms, inference age p95 157 ms); control rate **39.7 Hz measured**; cleanup verified; 0 collisions |
| Core smoke | `v2_core_smoke.json` | 2/2 PASS |
| Catalog × 3 seeds | `v2_catalog_3seed.json` | **45/45 PASS, gate PASS** |
| Core × 5 weathers | `v2_core_5weather.json` | **29/30, gate FAIL** — `DynamicObjectCrossing` storm 1.225 s |
| Range loss (LiDAR + radar, 4 s) | `v2_range_loss.json` | PASS — brake held (`BRAKE_HOLD_NO_DATA` 9.4%), ODD VIOLATION, MRM → SAFE_STOP |
| Brake release with the new rule | range-loss run | observed: no release on missing data |
| Surface clearance criterion | all scenario runs | first live use; minimum 0.88 m ≥ 0.25 m |
| Curved-road MRM | — | **NOT_RUN** (no command exists) |

Across the 78 scenario runs: no collision recorded, zero frame errors, zero
INVALID/ERROR cases, zero scenario execution errors, cleanup verified in every
case. One run per case: run-to-run variation is not measured, which matters for
`DynamicObjectCrossing` (0.5–1.225 s today).

### Follow-up runs, 2026-09-27 evening (`b3bba41`)

| Check | Report | Result |
|---|---|---|
| `DynamicObjectCrossing` × 10 seeds, clear | `v2_doc_clear_10seed.json` | 8/10 — late runs 1.225 s, via radar |
| `DynamicObjectCrossing` × 10 seeds, storm | `v2_doc_storm_10seed.json` | 7/10 — late runs 1.225 s, via radar |
| One sensor lost (LiDAR / radar / camera), `HardBrake` | `v2_fault_*_loss.json` | 3 PASS; hold kept, ODD NORMAL, no MRM; fault starts after the hazard |
| One sensor lost for the whole run, `HardBrake` (`adec9c4`) | `v2_fault_*_loss_from_start.json` | 3 PASS; LiDAR-only reacted 0.15 s later with 3.04 m clearance (5.78 m with radar); ODD NORMAL throughout |
| LiDAR read timeout in `chinh.py` (R9c) | `v2_soak_5min_20veh.json` | observed live: loss recorded, safe-stop command sent |
| Soak, 300 s, 20 NPC vehicles | `v2_soak_5min_20veh.json` | `FAIL` — stopped at 51 s; cleanup not verified (server unresponsive); 38.5 Hz until then |

The repeats narrow the `DynamicObjectCrossing` question (§7.3): all 6 late runs
of 29 were decided by the radar at the same geometric point, all 23 on-time runs
by the camera tracker. The per-frame diagnosis trace is still not done.

## Measured FPS / latency

**Measured once, as a smoke test.** `v2_runtime_smoke_async.json`: 200 unique
control updates in 5.03 s of active control window → **39.7 Hz delivered**,
simulated 5.03 s, real-time factor 0.999, control interval p50 24.8 / p95 30.8 /
p99 51.5 / max 83.1 ms, 3 deadline misses (> 50 ms). Hardware: RTX 4070 Laptop
machine, CPU inference, `low-memory`, headless, 0 NPCs, empty road. Five seconds
is not a sustained-rate qualification. Neural inference p95 was 69 ms (object
detection) and 120 ms (learned lane) against a 50 ms target.

## Claims that can be used now

Exact wording is in [`CLAIM_EVIDENCE_REGISTER.md`](CLAIM_EVIDENCE_REGISTER.md).
Usable: C1 (with its conditions), C4 and C5 (now the 2026-09-27 v2 results, dated,
with the weather-matrix FAIL), C7, C10 (as a 5 s smoke measurement only), C12,
C13, C14, C16, C17. Withdrawn: "no learned component can
suppress it", "any obstacle", "never creeps", "late, not unsafe", runtime
rain-aware fusion, and any 40 Hz / real-time / hard-deadline statement.

## Remaining limitations

Native crash B01 (open); human takeover (simulated flag only); physical friction
(heuristic proxy, μ < 0.3 unreachable in CARLA weather); MRM steering (G10);
near-field blind zone after a brake (G9 remainder); detector accuracy; no
validation outside the simulator. The 2026-09-25 changes have one live run per
matrix; no repeat runs, no GPU inference, no other town, no soak.

## Definition of done (brief §17)

| Item | Status |
|---|---|
| AEB not weakened by MRM comfort; fault latch kept | done (offline); live range-loss run shows the arbiter selecting the stronger request (MRM 1.0 over AEB hold 0.7) |
| Valid brake/throttle/steer, one command per cycle, ownership | done (offline); lateral hold at 0 is an explicit fallback |
| NaN / missing / stale input never becomes ODD NORMAL or scenario PASS | done for ODD and scenario verdicts; stale handling only via existing freshness gates |
| Sole range loss detected; missing geometry does not release the latch | done; the latch hold seen live under dual LiDAR+radar loss; the radar-disabled sole-LiDAR case is offline only |
| Takeover ACK does not re-engage in violation; honest manual-source status | done |
| Reaction metric versioned; decision vs applied vs motion separated | partial — versioned, decision only |
| Actor-origin distance not called surface clearance | done |
| Unique/complete matrix; empty/duplicate/partial suites never PASS | done |
| Verdict/exit code after cleanup; partial reports kept | done; cleanup verified live in all 104 scenario cases; the runtime's failure path seen live in the soak (cleanup not verified → FAIL); the scenario runner's failure path is exercised offline only |
| Configured Hz, measured Hz, sim time, real-time factor separated | done; measured once (5 s smoke, 39.7 Hz) |
| Provenance, counts/windows, missing-data status, no NaN in JSON | partial — scenario reports only |
| Weather/SNR/μ proxies not described as physics | done |
| Unwired fusion helper not described as runtime | done |
| Test counts and coverage from real runs | done (210 / 409 / 306, 2026-09-25) |
| README free of "any", "never creeps", "late not unsafe", hard deadline / zero overhead | done |
| No Guardian / voice / firmware achievements invented | done — not addressed in this repository |
| **Simulation verification** | **done 2026-09-27, one run per matrix**; curved-road MRM NOT_RUN |
