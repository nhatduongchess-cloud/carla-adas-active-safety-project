# Evidence remediation — results, 2026-09-25

What was done against the remediation brief of 2026-09-25 (written for commit
`07717f7`) and its accompanying probe script
([`scripts/claude_evidence_review_20260925.py`](../scripts/claude_evidence_review_20260925.py)),
what was verified and how, and what was not done.

**Headline.** The code, tests and documentation parts are done and verified
offline. **No simulator run was made** — no CARLA server was running on the
development machine (no `CarlaUE4` process, port 2000 closed) — so every
simulation check below is `NOT_RUN`, and no published result has changed.

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

## Simulation verification — NOT_RUN

Every item below needs a CARLA server and was not run. Nothing here may be
reported as observed.

| Check | Planned command (from the brief, §14.3) | Status |
|---|---|---|
| Runtime smoke, async | `chinh.py --town Town02 --vehicles 0 --seed 42 --duration 5 --no-display --performance-profile low-memory --runtime-mode async-stable --inference-device cpu --run-report logs/evidence_v2_smoke_async.json` | NOT_RUN |
| Core smoke | `run_scenarios.py --scenarios HardBrake,DynamicObjectCrossing --town Town02 --seed 42 --weather clear --seconds 20 --inference-device cpu --report logs/evidence_v2_core_smoke.json` | NOT_RUN |
| Catalog × 3 seeds | `run_scenarios.py --scenarios all --town Town02 --seeds 42,1337,2026 --weather clear --seconds 20 --inference-device cpu --report logs/evidence_v2_catalog_3seed.json` | NOT_RUN |
| Core × 5 weathers | `run_scenarios.py --scenarios core --town Town02 --seed 42 --weathers clear,light_rain,heavy_rain,fog,storm --seconds 20 --inference-device cpu --report logs/evidence_v2_core_5weather.json` | NOT_RUN |
| Range loss | `run_scenarios.py --scenarios HardBrake --town Town02 --seed 42 --weather clear --seconds 20 --fault lidar-radar-loss --fault-start 5 --fault-duration 4 --inference-device cpu --report logs/evidence_v2_range_loss.json` | NOT_RUN |
| Brake release with the new rule | the scenario runs above | NOT_RUN |
| Curved-road MRM | no command exists yet | NOT_RUN |

Write new reports to new file names; do not overwrite the
`docs/benchmarks/head_*` files. A non-PASS suite now exits nonzero, so a script
chaining these commands must not stop at the first failure if the whole matrix is
wanted.

## Measured FPS / latency

**Not measured.** The new control-window measurement has never run. The only
published runtime report (2026-09-12 demo) gives 20.0 frames per wall-second
**including teardown**, with `fps_ema` 44.9 as a HUD value; neither is a
control-rate measurement.

## Claims that can be used now

Exact wording is in [`CLAIM_EVIDENCE_REGISTER.md`](CLAIM_EVIDENCE_REGISTER.md).
Usable: C1 (with its conditions), C4 and C5 (historical, dated, with the gate
FAIL), C7, C12, C13, C14, C16, C17. Withdrawn: "no learned component can
suppress it", "any obstacle", "never creeps", "late, not unsafe", runtime
rain-aware fusion, and any 40 Hz / real-time / hard-deadline statement.

## Remaining limitations

Native crash B01 (open); human takeover (simulated flag only); physical friction
(heuristic proxy, μ < 0.3 unreachable in CARLA weather); MRM steering (G10);
near-field blind zone after a brake (G9 remainder); detector accuracy; no
validation outside the simulator. The 2026-09-25 changes are verified offline
only.

## Definition of done (brief §17)

| Item | Status |
|---|---|
| AEB not weakened by MRM comfort; fault latch kept | done (offline) |
| Valid brake/throttle/steer, one command per cycle, ownership | done (offline); lateral hold at 0 is an explicit fallback |
| NaN / missing / stale input never becomes ODD NORMAL or scenario PASS | done for ODD and scenario verdicts; stale handling only via existing freshness gates |
| Sole range loss detected; missing geometry does not release the latch | done (offline) |
| Takeover ACK does not re-engage in violation; honest manual-source status | done |
| Reaction metric versioned; decision vs applied vs motion separated | partial — versioned, decision only |
| Actor-origin distance not called surface clearance | done |
| Unique/complete matrix; empty/duplicate/partial suites never PASS | done |
| Verdict/exit code after cleanup; partial reports kept | done (offline) |
| Configured Hz, measured Hz, sim time, real-time factor separated | done in code; never measured |
| Provenance, counts/windows, missing-data status, no NaN in JSON | partial — scenario reports only |
| Weather/SNR/μ proxies not described as physics | done |
| Unwired fusion helper not described as runtime | done |
| Test counts and coverage from real runs | done (210 / 409 / 306, 2026-09-25) |
| README free of "any", "never creeps", "late not unsafe", hard deadline / zero overhead | done |
| No Guardian / voice / firmware achievements invented | done — not addressed in this repository |
| **Simulation verification** | **NOT_RUN** |
