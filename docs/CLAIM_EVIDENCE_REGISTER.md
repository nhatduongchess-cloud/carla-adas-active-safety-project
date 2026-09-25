# Claim → evidence register

Every claim this project makes in its README, docs or a CV, with the wording it
may use and the evidence behind it. Created 2026-09-25 by the evidence
remediation. Requirement IDs refer to [`REQUIREMENT_MAP.md`](REQUIREMENT_MAP.md);
metric definitions are in [`METRIC_DEFINITIONS.md`](METRIC_DEFINITIONS.md).

**Status:** `SUPPORTED` (evidence of matching scope exists) ·
`SUPPORTED-HISTORICAL` (true of a published run on earlier code, must be quoted
with its date) · `UNIT-ONLY` (tests, no simulator run on current code) ·
`UNVERIFIED` (no evidence; do not use) · `WITHDRAWN` (was claimed, is wrong).

Live simulator verification of the 2026-09-25 code changes is **NOT_RUN**.
Every claim about current behaviour in a simulator therefore rests on runs of
earlier code, and says so.

| ID | Current / previous wording | Allowed wording | Scope | Evidence | Status | Gap |
|---|---|---|---|---|---|---|
| C1 | "the emergency brake is … never a detector output — so no learned component can suppress it" (README intro) | "Implemented geometry-based emergency-braking checks using range measurements and a selected driving corridor; unit tests verify that removing detector labels does not suppress a critical brake request when geometry, path and sensor validity are held fixed." | LiDAR/radar corridor check, fixed inputs | R1; `selftest.py` AEB checks; `tests/test_safety_geometry.py` | UNIT-ONLY | The learned lane can select the corridor; arbitration sits in the chain. The broad "no learned component" form is **WITHDRAWN**. |
| C2 | "brakes for *any* in-path obstacle" | "brakes for in-path obstacles within the LiDAR safety envelope (≈30 m ahead, ±10 m lateral, ground-height filtered, ≥2 returns per voxel), independent of detector class" | Safety extraction envelope | `LidarProcessor.extract_safety_obstacles` | UNIT-ONLY | "Any" **WITHDRAWN**: nothing outside the envelope or below the return threshold is seen. |
| C3 | "never creeps into a stationary obstacle" | "classifies blockers by obstacle speed and brakes to a stop rather than creeping; in the published runs no collision was recorded" | Committed FSM | R2; selftest creep checks; `head_catalog_3seed.json` | SUPPORTED-HISTORICAL (2026-09-19) + UNIT-ONLY | "Never" **WITHDRAWN**. |
| C4 | "Full catalog 43/45 ✅" | "Evaluated 15 scenario recipes across three seeds in Town02: 43/45 cases met the recorded criteria, with no collisions recorded in that matrix; two pedestrian-crossing cases exceeded the 1.0 s reaction budget. The overall core-plus-catalog gate failed." | 45 runs, 2026-09-19, CPU inference, reaction metric v1, centre-to-centre clearance | `docs/benchmarks/head_catalog_3seed.json` | SUPPORTED-HISTORICAL | Clearance criterion was inert; perception p95 up to 132 ms vs 50 ms target. |
| C5 | "Collisions, all 90 runs: 0" | "No collision was recorded in the 90 published HEAD runs of 2026-09-19 (45 catalog + 30 weather + 6 core + 9 DynamicObjectCrossing probes); these overlap and are not 90 independent situations." | 4 report files | `head_catalog_3seed`, `head_core_5weather`, `head_core_seed42`, `head_doc_probe` | SUPPORTED-HISTORICAL | The table in the README listed only three files (81 runs); corrected. |
| C6 | "late, not unsafe" / "clearance > 3.1 m" | — | — | — | WITHDRAWN | Both came from centre-to-centre distances. |
| C7 | "L3 … TOR → MRM → SAFE_STOP" | "Built a simulation prototype of ODD monitoring and takeover-request / minimum-risk-maneuver state transitions, with a project-configured 10 s takeover window." | Pure state machine + harness | R6a, R6b; selftest §L3; `TakeoverOwnershipTests` | UNIT-ONLY | Takeover is a simulated flag (`human_takeover_verified: false`); MRM steers straight (G10); no curved-lane stop tested. Not "full L3 responsibility", not DRIVE PILOT-equivalent. |
| C8 | "rain-aware fusion weighting" | "a unit-tested helper for weather-weighted range fusion (not wired into the runtime)" | `modules/sensor_fusion_eval.py` | selftest §9 | UNIT-ONLY | Not used by `chinh.py` or the harnesses. As a runtime feature: **WITHDRAWN**. |
| C9 | "friction model" | "a heuristic friction/visibility model from CARLA weather parameters, used for stopping-distance margins and ODD classification" | `weather_model`, `friction` | selftest | UNIT-ONLY | Not calibrated physics; CARLA tyre friction is not set from it; μ < 0.3 unreachable (G11). |
| C10 | "40 Hz control loop" / "real-time" | "Configured a 40 Hz simulation/control target and decoupled neural inference; the delivered control rate is measured separately from the simulation step." | Configuration + instrumentation | `modules/control_timing.py`; `tests/test_evidence_remediation.py::ControlTimingTests` | UNIT-ONLY | **No measured control rate exists yet.** The only published demo gives 20.0 frames/wall-second including teardown; its fps_ema 44.9 is a HUD value. |
| C11 | "hard deadline" / "zero overhead" | "Instrumented control and perception latency, freshness and deadline misses." | Telemetry | `modules/pipeline_metrics.py`, `modules/control_timing.py` | UNIT-ONLY | No telemetry on/off A/B has been run; no hard-real-time claim. |
| C12 | "MRM never weakens AEB" | "When an MRM and the AEB both request braking, the command sent is the stronger of the two." | Command invariant | R6c; `MrmAebArbitrationTests`; `test_ego_control.py` | UNIT-ONLY | Command only — not a guaranteed deceleration. Live: NOT_RUN. |
| C13 | "Fail-closed runtime" | "The main runtime and each scenario case report success only after teardown is verified; a failed cleanup fails the run and the exit code." | `chinh.py`, `run_scenarios.py` | R10; `tests/test_runtime_cleanup.py`; `CaseStatusTests` | UNIT-ONLY (+ the 2026-09-12 demo correctly reported FAIL on unverified cleanup) | Scenario-runner cleanup semantics new; NOT_RUN live. |
| C14 | "179 checks / 106 tests" (old CV) · "204 checks / 284 tests" | "Maintained 210 offline self-test checks and 409 passing unit tests on the development environment (2026-09-25), covering selected geometry, control, sensor and reporting behaviour." | Test inventory | runner output, 2026-09-25 | SUPPORTED | Two different kinds of count — do not add them. Not a coverage figure. 103 of the 409 are in the six modules that import the CARLA client (no server needed). |
| C15 | "Detection via pretrained COCO YOLO" | same | demo | demo clips | SUPPORTED-HISTORICAL | Accuracy not qualified. |
| C16 | "Found and fixed fail-open paths" | "An evidence review found fail-open paths (NaN ODD read as normal, MRM weakening an AEB brake, duplicate or empty suites reading as a pass, a disabled radar reported 100 % available, clearance measured centre-to-centre); each is fixed with a regression test." | Code + tests | `docs/EVIDENCE_REMEDIATION_RESULTS.md`; `tests/test_evidence_review.py`, `tests/test_evidence_remediation.py` | SUPPORTED | Fixes verified offline only. |
| C17 | B01 "directly confirms object-lifetime race" | "Native crash observed as a pure-virtual call in skeletal-mesh scene-proxy dispatch during camera scene capture; object-lifetime race is the leading, unconfirmed hypothesis. Open, contained by bounded runs." | `docs/B01_FAILURE_ANALYSIS.md` | symbolicated stack 2026-09-12 | SUPPORTED (corrected wording) | Python changes do not fix the native root cause. |

## Other projects

This register covers the CARLA project only. Numbers for other projects (a
driver-monitoring model, a voice interface, firmware) need their own evidence
records in their own repositories; nothing in this repository supports them.
