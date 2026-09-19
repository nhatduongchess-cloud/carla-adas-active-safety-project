# Benchmark artifacts

The JSON reports behind the numbers in the [main README](../../README.md#results).
They were previously only in `logs/`, which is git-ignored, so the figures could
not be checked by anyone reading the repository. They are published here
unmodified — same bytes the harness wrote.

## Run on HEAD, 2026-09-19

Re-run against a live CARLA server (build `edf3e9f5c`, client `edf3e9f5c`,
Town02, RTX 4070 Laptop) on the current code, after the F01–F12 review fixes.

| Artifact | Suite | Cases | Result |
|---|---|---:|---|
| [`head_catalog_3seed.json`](head_catalog_3seed.json) | Full catalog × 3 seeds, clear | 45 | **43 pass / 2 fail** — target ≥43/45 met |
| [`head_core_5weather.json`](head_core_5weather.json) | Core × 5 weather profiles | 30 | **28 pass / 2 fail** |
| [`head_core_seed42.json`](head_core_seed42.json) | Core, seed 42, clear | 6 | **6 pass / 0 fail** |
| [`head_doc_probe.json`](head_doc_probe.json) | `DynamicObjectCrossing` × 3 seeds × 3 weathers | 9 | 3 pass / 6 fail — see below |
| [`pre_f02_doc_heavyrain.json`](pre_f02_doc_heavyrain.json) | Same scenario on the **pre-fix** commit `47d9274` | 3 | 0 pass / 3 fail — attribution evidence |

**Zero collisions and zero camera–LiDAR frame errors in all 90 runs.** Every
failure above is one criterion — reaction delay — on one scenario.

### The one failing scenario, and why it is not a regression

`DynamicObjectCrossing` misses the `reaction_delay_s <= 1.0` budget about half
the time, at **1.225 s** (49 frames at 0.025 s). It brakes, it never collides,
and clearance stays above 3.1 m; it is simply late against the stated budget.

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

- **The three scenario reports predate the `meta` block** the harness now writes,
  so they do not carry the commit SHA, CARLA build, client version or hardware.
  They were produced on 2026-09-01 on the development machine (RTX 4070 Laptop,
  Windows 11, CARLA 0.9.15) against the radar-ground-filter build — that comes
  from the project log, **not** from inside the files. Reports generated by the
  current harness do embed all of it.
- They therefore cannot be tied byte-for-byte to today's HEAD. That gap is now
  closed a different way: the suites were **re-run on HEAD on 2026-09-19** and the
  results are at the top of this file. Quote those.

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
| `run.simulated_duration_s` | 22.175 (target 30.0) |
| `safety.aeb_triggered` | **true** — 310 of 887 frames commanded BRAKE |
| `criteria.safety_p99_le_25ms` | false |
| `criteria.perception_p95_le_50ms` | false |
| `criteria.inference_age_p95_le_150ms` | false |

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
python run_scenarios.py --town Town02 --scenarios all  --seeds 42,1337,2026 --weather clear --report logs/catalog_clear_3seed.json
python run_scenarios.py --town Town02 --scenarios core --seeds 42,1337,2026 --weather clear --report logs/core_clear_3seed.json
python run_scenarios.py --town Town02 --scenarios core --seed 42 --weathers clear,light_rain,heavy_rain,fog,storm --report logs/core_5weather.json
```

Read the JSON verdicts rather than the exit code, and distinguish a scenario that
failed from a run that never completed because the process crashed.
