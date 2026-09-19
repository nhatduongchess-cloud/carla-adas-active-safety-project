# Requirement → evidence map

Traces each portfolio claim to the code that implements it and the evidence that verifies it, so a reviewer can check any claim. Evidence types: **self-test** (`selftest.py`, CARLA-free), **unit** (`test_*.py`), **scenario** (`run_scenarios.py`), **demo** (recorded clip + run-report JSON). Status reflects the portfolio scope (`docs/PORTFOLIO_SCOPE_PLAN.md`), not the full Luna certification.

| # | Capability (README claim) | Implemented in | Evidence | Status |
|---|---|---|---|---|
| R1 | Geometry-first AEB: metric LiDAR swept path aligned to the waypoint centerline; detector-independent braking | `modules/active_safety.py`, `modules/road_geometry.py` | self-test (TTC/safe-distance/swept-path); scenario `StationaryObjectCrossing`, `ConstructionObstacle`; demo Clip B | ✅ demonstrated |
| R2 | Committed safety FSM (`NORMAL→FOLLOW→EVADE→EMERGENCY_BRAKE`) with hysteresis, no creep, brake-latch across drop-outs | `modules/active_safety.py` | self-test (FSM transitions, creep-guard, latch); scenario `FollowLeadingVehicle`, `HardBrake` | ✅ demonstrated |
| R3 | Radar ground-plane rejection removes the 9–10 m ray/road false AEB without weakening AEB | `modules/radar_processor.py` | self-test (radar ground rejection/sign); demo Clip A (no phantom brake on clear road) | ✅ demonstrated |
| R4 | Sensor fusion: 6-DoF LiDAR→camera projection, one-to-one association, Mahalanobis updates, ego-motion compensation | `modules/sensor_fusion.py`, `modules/mot_tracker.py`, `modules/ego_motion.py` | self-test (projection/association/fusion); unit `tests/test_safety_geometry.py` | ✅ demonstrated |
| R5 | Audited learned lane (UFLDv2) + confidence/width/jump gates + junction priority + CARLA-map fallback | `modules/learned_lane.py`, `modules/lane_source.py`, `modules/ufldv2_backend.py` | self-test (lane fallback/arbitration); `docs/ufldv2_security_audit.md`; demo Clip A HUD lane source | ✅ demonstrated |
| R6 | L3 ODD monitor + `TOR→MRM→SAFE_STOP` + friction/stopping model + rain-aware fusion weighting | `modules/odd_monitor.py`, `modules/mrm_controller.py`, `modules/friction.py`, `modules/weather_model.py` | self-test (ODD/MRM paths); demo Clip C (`--driver-takeover`) | ✅ demonstrated |
| R7 | Safety-capped RL cruise: DQN proposes speed only; AEB/MRM always override | `modules/rl_speed_controller.py`, `modules/rl_agent.py` | self-test (RL cap / safety override); README design note | ✅ demonstrated |
| R8 | Custom planning/control: waypoint route intent, quintic lane change, pure-pursuit + anti-windup PID | `modules/local_planner.py`, `modules/lateral_controller.py`, `modules/longitudinal_controller.py`, `modules/ego_driving_stack.py` | unit `tests/test_ego_control.py`, `tests/test_junction_route.py`, `tests/test_road_waypoints.py`; demo Clip A | ✅ demonstrated |
| R9 | Exact-frame sensor sync + 150 ms freshness; bounded latest-frame-only inference | `modules/sensor_sync.py`, `modules/inference_scheduler.py`, `modules/perception_contracts.py` | self-test (stale-result rejection); unit `tests/test_inference_telemetry.py`, `tests/test_neural_decoupling.py`, `tests/test_perception_contracts.py` | ✅ demonstrated |
| R10 | Fail-closed teardown: success only after verified cleanup; failed/timeout cleanup fails the run + exit | `modules/runtime_cleanup.py`, `chinh.py` (finally), `modules/runtime_report.py` | unit `tests/test_runtime_cleanup.py` (10 cases), `tests/test_runtime_report.py` | ✅ demonstrated |
| R11 | Reproducible V&V: seeded curated scenario suite with machine-readable acceptance | `run_scenarios.py`, `modules/scenario_library.py`, `modules/kpi.py` | scenario `--scenarios core --seeds 42,1337,2026`; reports under `logs/` | ⏳ run locally (demo runbook §2) |
| R12 | CARLA-free safety self-test in CI | `selftest.py`, `.github/workflows/selftest.yml` | 204 checks / 0 fail; CI on every push | ✅ demonstrated |
| R13 | Small data story: reviewed CARLA capture + labelled pipeline (demonstrator, not training-grade) | `scripts/dataset/collect_carla_dataset.py`, `modules/dataset_labels.py`, `scripts/dataset/review_label_pilot.py`, `scripts/dataset/validate_dataset.py`, `scripts/dataset/audit_training_dataset.py` | sample capture + overlays (run locally); `docs/dataset_label_v3_progress.md` | ⏳ sample pending |
| R14 | Detection (demo) via pretrained COCO YOLO | `modules/neural_perception.py`, `weights/yolov8*.pt` | demo clips | ✅ demonstrated |
| L1 | Native engine stability on the custom build (B01) | server (`edf3e9f5c`) — no project-side fix possible | `docs/B01_FAILURE_ANALYSIS.md`; `output/verification/N01,N02,R01,N04*` | ⚠️ known limitation |
| L2 | Custom 8-class detector accuracy (mAP ≥ 0.75) | `scripts/training/train.py`, dataset pipeline | — | 🔮 future work |
| L3 | Full scenario × weather × fault matrix + 30-min soak | `run_scenarios.py` | — | 🔮 future work |
| L4 | Runtime throughput/latency qualification (20 FPS / p95) on GPU | `modules/inference_scheduler.py` | measured numbers only, not gated | 🔮 future work |

**Legend:** ✅ demonstrated · ⏳ pending a local run · ⚠️ known limitation · 🔮 future work (out of portfolio scope).

Scenario IDs available to `run_scenarios.py` (`--scenarios`): core 6 above, plus `FollowLeadingVehicleWithObstacle`, `OtherLeadingVehicle`, `CutInFrom_right_Lane`, `HighwayCutIn`, `VehicleTurningRight`, `VehicleTurningLeft`, `NoSignalJunctionCrossing`, `ManeuverOppositeDirection`, `OppositeVehicleRunningRedLight`. Categories (`--category`): `lead | crossing | cutin | junction | oncoming`.
