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
| [`head_catalog_3seed.json`](head_catalog_3seed.json) | Full catalog × 3 seeds, clear | 45 | **43 pass / 2 fail** — catalog sub-target ≥43/45 met, but the **acceptance gate recorded in the file is FAIL**: core is 16/18 against 18/18 required |
| [`head_core_5weather.json`](head_core_5weather.json) | Core × 5 weather profiles | 30 | **28 pass / 2 fail** |
| [`head_core_seed42.json`](head_core_seed42.json) | Core, seed 42, clear | 6 | **6 pass / 0 fail** |
| [`head_doc_probe.json`](head_doc_probe.json) | `DynamicObjectCrossing` × 3 seeds × 3 weathers | 9 | 3 pass / 6 fail — see below |
| [`pre_f02_doc_heavyrain.json`](pre_f02_doc_heavyrain.json) | Same scenario on the **pre-fix** commit `47d9274` | 3 | 0 pass / 3 fail — attribution evidence |

**Zero collisions and zero camera–LiDAR frame errors in all 90 runs.** Every
failure above is one criterion — reaction delay — on one scenario.

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

**Brake latch release.** After a critical brake, five consecutive frames with an
empty corridor release the latch, and while it holds the brake drops from 1.0 to
0.7. An empty corridor cannot be told apart from an obstacle that has entered the
LiDAR's near-field blind zone, so this can release a brake onto an object that is
still there. It is real. It is not fixed here because every candidate fix changes
live AEB behaviour in the scenarios above, and a change to the AEB that has not
been re-run on the simulator is a change nobody has tested. Open gap **G9**.

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
