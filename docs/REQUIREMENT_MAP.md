# Requirement → evidence map

Traces each README capability to the code that implements it, the tests that
exercise it, and any simulator run that observed it. Rewritten 2026-09-25 for the
evidence remediation ([`docs/EVIDENCE_REMEDIATION_RESULTS.md`](EVIDENCE_REMEDIATION_RESULTS.md)):
the previous version marked almost every row "✅ demonstrated", which did not
separate "the code exists" from "a test exercised it" from "a simulator run
showed it". The scope reference now points at the README
[Scope & status](../README.md#scope--status) section; the
`docs/PORTFOLIO_SCOPE_PLAN.md` it used to link does not exist in this repository.

**Status vocabulary** (a row can hold more than one):

| Status | Meaning |
|---|---|
| implemented | A concrete code path exists. |
| unit-verified | A named deterministic test drives that path with stated inputs and criteria. Pure logic or mocked integration — not a vehicle run. |
| simulation-observed | A CARLA run with a published artifact (map, seed, config, date) showed the behaviour. |
| known-failing | A published run or test shows the requirement not met. |
| unverified | Neither a test nor a run covers the claim, or the run predates the code. |

**Test types:** *pure* (pure-Python logic), *mocked* (CARLA client mocked),
*live* (CARLA server), *inspection* (reading code/docs). Test counts on
2026-09-25: `selftest.py` 210 checks; `unittest discover -s tests` 409 tests on the
development machine (Windows, `.venvCarLa`); 306 of those in the CI offline group.
None of the simulation evidence below was produced after the 2026-09-25 code
changes — every live run is **NOT_RUN** on the current code.

| # | Requirement (verifiable form) | Code | Tests (type) | Simulation evidence | Status | Not covered |
|---|---|---|---|---|---|---|
| R1 | With geometry, path, speed, μ, timestamps and sensor health held fixed, removing or relabelling detector output does not remove a critical geometric brake request. | `modules/active_safety.py` (`update`, `_scan_corridors`) | selftest §AEB (pure); `tests/test_safety_geometry.py` (pure) | `head_catalog_3seed.json` (2026-09-19, earlier code): 0 collisions in 45 runs | implemented, unit-verified for fixed inputs; simulation-observed on earlier code | No metamorphic test over detector labels in the live loop. Lane selection (learned path) and arbitration are still in the chain, so "no learned component can influence braking" is **not** claimed. Extraction range ≈30 m ahead limits "any obstacle". |
| R2 | The committed FSM does not creep into a static obstacle, and a latched brake is released only after a clear corridor is observed on valid LiDAR frames for 0.1 s; missing LiDAR data never releases it. | `modules/active_safety.py` | selftest §FSM/creep/latch (pure); `tests/test_evidence_remediation.py::BrakeReleaseEvidenceTests` (pure) | none on the new release rule | implemented, unit-verified | Live AEB with the new release rule: **NOT_RUN**. A valid empty corridor from an object inside the LiDAR near-field blind zone is indistinguishable from a clear road. |
| R3 | Radar ground-plane returns below the configured ego-frame height are rejected before clustering. | `modules/radar_processor.py` | selftest §radar (pure) | demo clip A (no phantom brake) | implemented, unit-verified, simulation-observed (clip) | Other road geometries (crests, ramps). |
| R4 | Fusion projects LiDAR to camera with 6-DoF extrinsics and associates one-to-one; tracks are ego-motion compensated. | `modules/sensor_fusion.py`, `modules/mot_tracker.py`, `modules/ego_motion.py` | selftest §fusion (pure); `tests/test_safety_geometry.py` | runtime demo | implemented, unit-verified | Association accuracy against ground truth is not measured. |
| R5 | Learned lane is gated (confidence/width/jump), junction-prioritised, with a map fallback. | `modules/learned_lane.py`, `modules/lane_source.py`, `modules/ufldv2_backend.py` | selftest §lane (pure) | demo HUD lane source | implemented, unit-verified | Lane accuracy not evaluated. |
| R6a | ODD: invalid/missing/bool inputs and total range loss give VIOLATION, never NORMAL; boundaries 20/50 m and μ 0.3/0.6 as published. | `modules/odd_monitor.py`, `modules/sensor_health.py` | `tests/test_evidence_remediation.py::OddInputContractTests`, `RangeHealthTests` (pure) | none | implemented, unit-verified | μ < 0.3 is unreachable from CARLA weather (G11); tested with injected values only. visibility/μ/SNR are heuristic proxies, not measured physics. |
| R6b | TOR → MRM → SAFE_STOP with a 10 s project-configured window; a simulated takeover ACK hands control to the driver and does not re-engage automation without an explicit request. | `modules/mrm_controller.py` | selftest §L3 (pure); `TakeoverOwnershipTests` (pure) | demo clip C (`--driver-takeover`, earlier code) | implemented, unit-verified | No real driver or manual input: `human_takeover_verified` is always false. Curved-lane MRM not tested (G10, steer 0). |
| R6c | MRM does not weaken a stronger AEB request: final brake = max(MRM, AEB). | `modules/control_arbitration.py`, `modules/ego_control.py` | `tests/test_evidence_review.py::MrmAebArbitrationTests` (pure); `tests/test_ego_control.py::EgoControllerEvidenceRemediationTests` (mocked) | none | implemented, unit-verified | Command invariant only — not a guaranteed deceleration or stopping distance. |
| R6d | Rain-aware range-fusion weighting. | `modules/sensor_fusion_eval.py` | selftest §9 (pure) | none | implemented as a **helper only** | **Not wired into the runtime** (`chinh.py`, `evaluate_l3.py` do not call it). Not a runtime feature. |
| R7 | The DQN proposes cruise speed only; AEB/MRM override it. | `modules/rl_speed_controller.py` | selftest §RL (pure) | runtime demo | implemented, unit-verified | — |
| R8 | Custom planning/control; any invalid command (NaN, out of range, bool) is a control fault leading to a safe stop, never sent; throttle is 0 whenever brake > 0. | `modules/ego_control.py`, `modules/control_arbitration.py`, planner/controller modules | `tests/test_ego_control.py` (mocked); `tests/test_junction_route.py`, `tests/test_road_waypoints.py` (local) | demo 2026-09-12 (earlier code) | implemented, unit-verified | Steer during AEB/MRM is held at 0: a degraded fallback, not lane keeping. |
| R9a | Synchronous mode: exact-frame camera/LiDAR matching. | `modules/sensor_sync.py` | selftest; `tests/test_sensor_sync.py` | scenario harness (sync) | implemented, unit-verified, simulation-observed | — |
| R9b | Async-stable mode: latest-frame reads with age/skew recorded, 150 ms freshness on neural results; algorithms use the validated simulation step, not the configured one. | `modules/sensor_runtime.py`, `modules/inference_scheduler.py`, `modules/control_timing.py` | `tests/test_inference_telemetry.py`, `tests/test_neural_decoupling.py`, `SimClockTests` | none on current code | implemented, unit-verified | Async is not exact-frame and is not claimed to be. |
| R9c | A LiDAR read timeout reaches sensor health and attempts a safe stop; the outcome (sent or RPC failure) is recorded. | `chinh.py` (`except TimeoutError`), `EgoController.sensor_loss_safe_stop` | `tests/test_ego_control.py` (mocked) | none | implemented, unit-verified (controller part) | The `chinh.py` handler itself is not exercised by a test; live timeout **NOT_RUN**. Python cannot brake a hung server. |
| R10 | The main runtime reports success only after verified cleanup. Scenario cases are final only after their cleanup. | `modules/runtime_cleanup.py`, `chinh.py`, `run_scenarios.py`, `modules/scenario_acceptance.py` | `tests/test_runtime_cleanup.py`, `tests/test_runtime_report.py`; `CaseStatusTests` | demo 2026-09-12 reported cleanup **not verified** → FAIL | implemented, unit-verified; known-failing in the one published demo | Suite-level restore failure → exit code is implemented but not live-tested. |
| R11 | Scenario suites report planned/attempted/passed/failed/invalid/error/not_run, reject duplicates, judge each matrix against its plan, and exit nonzero unless PASS. | `run_scenarios.py`, `modules/l3_report.py`, `modules/scenario_acceptance.py` | `SuiteAccountingTests`, `CaseStatusTests`, `ReactionTimingTests` (pure) | `head_catalog_3seed.json`: gate **FAIL** (core 16/18); `head_core_5weather.json`: 28/30 | implemented, unit-verified; **known-failing** in the latest published runs | Reaction metric v2 and surface clearance have **never been run live**. |
| R12 | CARLA-free checks run in CI. | `selftest.py`, `.github/workflows/selftest.yml` | 210 checks; 306-test offline group | CI | implemented | — |
| R13 | Small reviewed data sample and label pipeline. | `scripts/dataset/*`, `modules/dataset_labels.py` | `tests/test_dataset_labels.py`, `tests/test_validate_dataset.py` | — | implemented, unit-verified | Sample capture pending. |
| R14 | Pretrained COCO YOLO for demo detection. | `modules/neural_perception.py` | — | demo clips | implemented, simulation-observed | Detector accuracy not qualified. |
| R15 | Reported control rate is the measured rate over the active control window, not the HUD EMA or the configured 40 Hz. | `modules/control_timing.py`, `modules/runtime_report.py` | `ControlTimingTests`; `tests/test_runtime_report.py` | none — the published demo predates it (fps_ema 44.9 vs 20.0 frames/wall) | implemented, unit-verified | No measured control rate exists yet. |
| L1 | Native engine stability (B01). | server `edf3e9f5c` | — | `docs/B01_FAILURE_ANALYSIS.md` | known-failing (OPEN) | No project-side fix possible. |
| L2 | Custom 8-class detector accuracy. | `scripts/training/train.py` | — | — | unverified (future work) | — |
| L3 | Full scenario × weather × fault matrix + 30-min soak. | `run_scenarios.py` | — | — | unverified (future work) | — |
| L4 | Runtime throughput/latency qualification. | `modules/control_timing.py`, `modules/inference_scheduler.py` | — | catalog perception p95 up to 132 ms vs 50 ms target (CPU) | known-failing on CPU; unverified on GPU | — |

Scenario IDs available to `run_scenarios.py` (`--scenarios`): core 6
(`FollowLeadingVehicle`, `HardBrake`, `StationaryObjectCrossing`,
`DynamicObjectCrossing`, `ConstructionObstacle`, `CutInFrom_left_Lane` — see
`CORE_SUITE`), plus `FollowLeadingVehicleWithObstacle`, `OtherLeadingVehicle`,
`CutInFrom_right_Lane`, `HighwayCutIn`, `VehicleTurningRight`,
`VehicleTurningLeft`, `NoSignalJunctionCrossing`, `ManeuverOppositeDirection`,
`OppositeVehicleRunningRedLight`. Categories (`--category`):
`lead | crossing | cutin | junction | oncoming`.
