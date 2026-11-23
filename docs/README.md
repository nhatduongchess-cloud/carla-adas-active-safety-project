# Engineering documentation

## Follow a validation case

| Question | Implementation | Verification |
|---|---|---|
| What starts the pedestrian response clock, and when was braking commanded? | [Walker trigger](../modules/scenario_library.py) · [Acceptance logic](../modules/scenario_acceptance.py) · [Command timestamps](../modules/pedestrian_response.py) | [Per-case measurements](benchmarks/pedestrian_deadline_20261123.json) |
| Why did the predicted crossing trigger before direct radar? | [Motion fit and conflict prediction](../modules/radar_crossing.py) · [Threat selection](../modules/active_safety.py) | [Radar regression test](../tests/test_radar_crossing.py) · [Recorded storm input](../tests/fixtures/safety/town02_storm_radar_prefix.json.gz) |
| Can changing only the centerline change the safety decision? | [Lane selection and fallback](../modules/lane_source.py) · [Path geometry](../modules/road_geometry.py) | [Curved-path and fallback test](../tests/test_pedestrian_response.py) |
| How are unintended interventions checked? | [Standing and parallel-walking controls](../modules/scenario_library.py) | [Sidewalk exposure and command records](benchmarks/pedestrian_deadline_20261123.json), under `non_crossing.rows` |

## Run and interpret the tests

- [November requirement-to-test matrix](REQUIREMENT_MAP.md#pedestrian-aeb-follow-up-2026-11-23) — check acceptance criteria for crossings, sidewalk controls and lane selection.
- [Safety metric definitions](METRIC_DEFINITIONS.md#safety-metrics) — read the clocks, command timestamps, clearance and exposure rules.
- [Offline regression and simulator commands](../README.md#getting-started) — reproduce the storm radar decision or run the closed-loop matrix.
- [Python safety replay](PYTHON_SAFETY_REPLAY.md) — capture state-machine inputs and compare decisions tick by tick.
- [Benchmark archive](benchmarks/README.md) — find a report by scenario, date and measurement definition.
- [Demo runbook](DEMO_RUNBOOK.md) — start the Windows simulator and record a dashboard run.

## Inspect the wider system

- [Architecture](ARCHITECTURE.md) — sensor scheduling, safety decisions, command arbitration and teardown.
- [September system baseline](REQUIREMENT_MAP.md#september-2026-system-baseline) — wider system requirements and historical verification results.
- [Native engine failure analysis](B01_FAILURE_ANALYSIS.md) — investigate the CARLA stability limit observed in longer runs.
