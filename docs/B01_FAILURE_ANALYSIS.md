# B01 — Native engine crash: failure analysis

Status: **OPEN / NOT_FIXED (contained).** Last reviewed: 2026-09-12, Asia/Bangkok.
Scope: analysis and containment of an intermittent native crash in the local CARLA server, from a Python client project. This documents what was proven, what was ruled out, and why repair is out of scope — not a fix.

## 1. Summary

Under the heavier sensor-capture workload, the local CARLA server terminates with a native `EXCEPTION_ACCESS_VIOLATION` in skeletal-mesh **scene-proxy render dispatch** (`FSkeletalMeshSceneProxy` / `MeshObject`), reached inline from the camera's scene-capture path. The fault is **intermittent**, reproduces **across graphics back-ends and threading modes**, and is **not** an out-of-memory or a RenderThread-scheduling race. The available crash dumps do not contain the heap pages that would identify the specific freed/corrupted object, and no matching native source or build tree exists for this custom engine build, so the root cause cannot be proven or repaired within project scope. Work is therefore scoped to a **stable capture envelope** in which the pipeline runs reliably, and this fault is carried as a documented limitation.

## 2. Environment

| Item | Value |
|---|---|
| OS / host | Windows, RTX 4070 Laptop (8 GB VRAM), 32 GB RAM |
| GPU driver (R01) | NVIDIA 592.00 |
| CARLA build | custom `edf3e9f5c`, UE4 4.26.2 (not a stock release) |
| Client | CARLA 0.9.16 client, matching build `edf3e9f5c` |
| Server launch | Low quality, windowed 640×360, default RHI, `-fullcrashdumpalways` |

## 3. Symptom and native signatures

Two consistent signatures, both pointing at the same object:

- **Execute access violation at address `0x0`**, return RVA `0x2ec2a4c`, inside skeletal-mesh scene-proxy render dispatch during the camera `SceneCapture` path (N01, N02). Dispatching through a null/garbage virtual-table pointer.
- **Read access violation at `0x00000001e30d0168`** (crash `UE4CC-Windows-A2B2B0B549357FCDFFA124A117BC20CA_0000`, 2026-09-05). The low bits `0x168` match the **`MeshObject` member offset `0x168`** confirmed in the PDB — consistent with dereferencing a member on a corrupted or near-null object base.

Symbolication was done with the **matching local PDB**, signed Windows DbgHelp, and installed objdump. Dump→EXE identity was verified (CodeView GUID / age / size / timestamp), and EXE→PDB identity matched. The native code at the dispatch site **already performs a non-null check**, so a simple missing null-guard is not the cause.

## 3b. Clearest reproduction — fully symbolicated RenderThread stack (2026-09-12)

A clean-driving run (open map, ego at a clear spawn, ~34 km/h, clear road, zero NPCs) crashed at ~20 s with a **fully symbolicated** fatal error — the sharpest evidence to date:

```
LowLevelFatalError [Line: 475] Pure virtual function being called   (GIsRunning == 1)
  FSkeletalMeshSceneProxy::GetMeshElementsConditionallySelectable()  SkeletalMesh.cpp:5129
  FSkeletalMeshSceneProxy::GetDynamicMeshElements()                  SkeletalMesh.cpp:5119
  FSceneRenderer::GatherDynamicMeshElements()                        SceneVisibility.cpp:2908
  FSceneRenderer::ComputeViewVisibility()                            SceneVisibility.cpp:4047
  FDeferredShadingSceneRenderer::InitViews()                         SceneVisibility.cpp:4304
  FDeferredShadingSceneRenderer::Render_CARLA()                      DeferredShadingRenderer.cpp:1696
  UpdateSceneCaptureContentDeferred_RenderThread()                  SceneCaptureRendering.cpp:294
  UpdateSceneCaptureContent_RenderThread()                          SceneCaptureRendering.cpp:421
  RenderingThreadMain()                                             RenderingThread.cpp:373
```

**Interpretation.** "Pure virtual function being called" means the object's vtable pointer is the **base-class (pure-virtual) vtable at the moment of the call** — which only happens while a C++ object is **mid-construction or mid-destruction**. So an `FSkeletalMeshSceneProxy` is being built or torn down **concurrently** while the **camera sensor's scene-capture** render pass (`UpdateSceneCaptureContent_RenderThread` → `Render_CARLA` → `GatherDynamicMeshElements`) iterates the dynamic mesh elements of that same proxy on the RenderThread.

This directly confirms the earlier object-lifetime hypothesis and sharpens it:
- The faulting object is specifically a **skeletal mesh** (animated actor — pedestrian/vehicle with skeletal components) present in the captured view.
- The trigger is the **camera sensor's `SceneCapture`** — i.e., the crash is intrinsic to capturing frames of a scene containing an animated skeletal mesh whose scene-proxy lifecycle races the capture render pass.
- It is a **RenderThread** fault reached through CARLA's `Render_CARLA` scene-capture path, consistent with the earlier `-onethread` and cross-RHI reproductions (the capture is invoked regardless of render-thread scheduling).

This supersedes the heap-less minidumps: it names the faulting function, file:line, and the exact render path, and it occurred on an otherwise-healthy clean-driving run — showing the fault is **intermittent and timing-dependent** (here ~20 s in), not tied to heavy load. Containment therefore relies on **bounded run duration**: ending a capture run before the intermittent window (e.g., short ≤12–15 s demo clips) avoids it, but does not repair it.

## 4. Reproduction matrix

| Trial | Configuration | Outcome |
|---|---|---|
| 09-12 clean drive | Open map, clear spawn (index 2), ~34 km/h, 0 NPC, low-memory | Crash ~20 s — **RenderThread pure-virtual** in `FSkeletalMeshSceneProxy` via camera SceneCapture (fully symbolicated, see §3b) |
| R01 | Low quality, 640×360, default RHI, **D3D12** | Crash after 940 loops / 176.9 m — RenderThread1 **null AV**, then LiDAR timeout |
| N02 | Low, 640×360, **`-onethread`** | Crash after 838 controls / 179.7 m — GameThread execute AV `0x0` at the **same** MeshObject return RVA |
| N01 (dump inventory) | historical crashes | **9 exact-hash** crashes + **6 near-variant** crashes, spanning **D3D11 and D3D12** |
| N04-A | Minimal RGB-only, stationary ego | Survived — 2400 frames / 59.975 s, no crash |
| N04-B / N04-C | Full stack, 60 s, Town10HD_Opt | Survived twice — 2400 frames each, ~325–345 m, no crash |
| N04-F | Minimal RGB, both maps + transition | Survived — 3/3 per map, no crash |

**Reading the matrix:** the fault appears with the heavier / longer capture path and is intermittent; bounded minimal and short full-stack runs survive. Survival of a bounded run is **not** a repair — it only bounds the envelope.

## 5. What was ruled out

- **Not D3D12-specific / not an RHI bug.** The identical faulting site appears under both D3D11 and D3D12, and low-render did not prevent it.
- **Not a RenderThread-scheduling race.** It still crashes under `-onethread`; rendering is invoked inline from the camera `PostPhysTick`, so separate render-thread scheduling is neither the explanation nor a fix.
- **Not out-of-memory.** Sampled device-wide GPU use stayed ~2.4–2.7 GB of 8 GB and system RAM remained ample across the failing trials; no sustained memory-exhaustion evidence.
- **Not a Python-side defect.** A Python reporting/exception fix cannot repair a compiled invalid virtual dispatch; the crash is inside the native server process.

## 6. Leading hypothesis (unproven)

An **object-lifetime / memory-corruption** issue: a skeletal-mesh scene proxy (e.g., an actor/mesh in view) is freed or left partially initialized and then dereferenced during camera scene capture. The specific responsible asset and the freeing site are **not proven**, because the retained minidumps lack the pointed-to heap pages. Establishing this would need a full-memory dump captured at the fault plus the native symbols/source for this exact build.

## 7. Why repair is out of scope

- **No matching native source or build tree** is available for the custom `edf3e9f5c` engine build, so the server cannot be rebuilt or patched, and the object cannot be traced to its allocation.
- The stock official **0.9.16** package was downloaded and security-audited (N03-A), but its shipping EXE/PDB/client **differ** from the installed development build; switching servers is a different-topology decision and is explicitly out of the project's "existing-installation-only" scope.
- Binary-patching the server or forcing driver/registry/TDR changes are out of scope and would not address an invalid (non-null) object.

## 8. Containment — the stable envelope

Demos, captures and scenario runs are scoped to the configuration that reliably survives:

- Low quality, 640×360, default RHI, `-fullcrashdumpalways`.
- Bounded run duration (≈60 s), one map at a time, minimal-to-moderate sensor load.
- Fail-closed teardown (`modules/runtime_cleanup.py`): a run reports success only after cleanup is verified; a crash or timeout fails the run honestly rather than masking it.

Any workload outside this envelope is treated as unverified until a current run shows otherwise.

## 9. What this demonstrates

Native-crash triage driven from a Python client project: matched-symbol symbolication (PDB / DbgHelp / objdump), dump identity verification, controlled single-variable reproduction across RHI / threading / quality, ruling out OOM and RHI-scheduling causes, forming an evidence-bounded hypothesis, and — critically — **containing** an un-repairable engine fault behind a documented stable envelope instead of hiding it or faking a fix.

## 10. Evidence index

- `output/verification/N01_native_symbols_20260906_2357/{SUMMARY.md,verification.json}` — symbolication, disassembly, 26-crash inventory.
- `output/verification/N02_threading_20260907_0040/{SUMMARY.md,verification.json}` — `-onethread` reproduction, inline-render finding.
- `output/verification/R01_low_render_20260906_1927/{SUMMARY.md,verification.json}` — low-render reproduction + copied native XML / minidump.
- `output/security/N03A_20260907_2132/SECURITY_REVIEW.md` — stock 0.9.16 package audit (candidate differs from installed build).
- `output/verification/N04{A,B,C,D,E,F}_*` — bounded survivals that establish the stable envelope.
- Native crash artifact: `%LOCALAPPDATA%\CarlaUE4\Saved\Crashes\UE4CC-Windows-A2B2B0B549357FCDFFA124A117BC20CA_0000\CrashContext.runtime-xml` (2026-09-05).

## 11. Status and future work

**OPEN / NOT_FIXED**, contained. Future investigation (out of portfolio scope) would require a full-memory dump at the fault and native symbols/source for this build, or a controlled migration to a stock CARLA release with re-validation. Until then, the fault is a documented limitation and does not block the portfolio deliverables, which run inside the stable envelope.
