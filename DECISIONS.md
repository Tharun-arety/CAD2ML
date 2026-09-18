# Architectural decisions (append-only)

Format: `D-NNN — date — decision`. **Context / Decision / Consequences.** Newer entries may supersede older
ones, but older entries stay in place.

---

### D-001 — 2026-09-17 — OCCT via CadQuery/OCP from conda-forge
**Context.** Real B-Rep work needs STEP transfer, topology maps, BRepCheck, ShapeFix, BRepMesh, surface
evaluation and projection. pythonOCC and OCP both wrap OCCT.
**Decision.** Use `cadquery=2.5` (OCP 7.7.2) from conda-forge. OCP handles kernel access and CadQuery builds the synthetic corpus.
Before building anything on top, `scripts/spike_occt.py` checked write→read→traverse→volume→mesh (passed, 1.25 s).
**Consequences.** The environment comes from conda (micromamba in Docker). The OCP API uses `_s` static methods and
returns tuples for out-params, as observed (e.g. `BRepTools.UVBounds_s(face)`).

### D-002 — 2026-09-17 — Isolated child processes for kernel work
**Context.** Uploaded STEP is untrusted input to native code that can hang or segfault.
**Decision.** Parsing and extraction run in two separate `spawn` children with hard wall-clock timeouts.
POSIX children also get `RLIMIT_AS`/`RLIMIT_CPU`. Results cross the process boundary as JSON plus files.
**Consequences.** Each child pays roughly 1–2 s of process start and OCP import. The isolation has already paid for itself: a native
BLAS crash (D-008) showed up as `CHILD_CRASHED` instead of killing the worker. Windows only gets the timeout, not the rlimits.

### D-003 — 2026-09-17 — Canonical IDs from quantized geometry, not traversal order
**Decision.** Faces and edges are sorted by (type rank, centroid/midpoint quantized to 1e-3 mm, area/length,
bbox, fingerprint). IDs `F###`/`E###` follow that order, and the kernel traversal index is kept alongside.
**Consequences.** IDs are stable across construction orders of identical geometry, which is tested. They are **not**
stable across edits. The fingerprint-based revision matching is labelled experimental.

### D-004 — 2026-09-17 — Durable state in the DB, Valkey as transport only
**Decision.** Job state lives in PostgreSQL (SQLite for native tests), and every transition is a
compare-and-set (`UPDATE … WHERE state = src`). Valkey only carries job ids (LPUSH/BLMOVE). Leases and
heartbeats live in the DB. A reaper re-queues jobs whose lease expired, and a reconciler re-enqueues DB
`queued` jobs that are missing from Valkey.
**Consequences.** Duplicate deliveries do no harm, and losing the queue loses no work (tested).

### D-005 — 2026-09-17 — Content-addressed samples; manifest written last
**Decision.** `sample_id = sha256(source_sha | pipeline_version | config_hash | schema_version)`. Artifacts are
written to `staging/…`, validated, then promoted with `os.replace`, and the manifest goes last.
**Consequences.** Readers never see a partial sample, and reruns are idempotent hits.

### D-006 — 2026-09-17 — Software rasterizer for multi-view outputs
**Context.** GL-based headless renderers (EGL/OSMesa) depend on host drivers and behave differently on Windows
and in Linux containers.
**Decision.** A NumPy z-buffer rasterizer over the OCCT tessellation. It gives perspective-correct depth, interpolated
surface normals, a uint16 face-ID mask, and a Lambert-shaded RGB image derived from the normals.
**Consequences.** It is exact and identical everywhere, and the face-ID ↔ topology link is checked by back-projecting pixels.
Rendering is about 0.4 s per part for 6 × 256² views. There are no textures or anti-aliasing (not needed for this use).

### D-007 — 2026-09-17 — Ground-truth face labels from analytic cutting tools
**Decision.** Every synthetic feature is cut with an analytic extruded rounded rectangle recorded in the
sidecar. A face gets a feature label when ≥95 % of its surface samples lie on that tool boundary; fillets
are matched by surface type and radius.
**Consequences.** Labels don't depend on the rule recognizer, so the recognizer and the models are evaluated
against independent labels. Hole and fillet radii are kept disjoint in the generator.

### D-008 — 2026-09-17 — Windows BLAS/OpenMP runtime
**Context.** conda-forge MKL `libblas` crashed NumPy matmul (0xC06D007F delay-load failure). The OpenMP
build of OpenBLAS clashed with the PyTorch wheel's `libiomp5md.dll` (OMP Error #15).
**Decision.** Native Windows uses the **pthreads** build of `libopenblas` (installed `--no-deps`) plus the PyTorch
CPU wheel (`scripts/setup_env_windows.ps1`). Linux containers use the default solve.
**Consequences.** NumPy, CadQuery and torch now coexist in one process. The workaround is documented rather than hidden.

### D-009 — 2026-09-17 — Single-threaded BLAS in geometry children
**Context.** During the first corpus run, 2 of 72 parts crashed with "OpenBLAS error: Memory allocation still
failed". That was multi-threaded OpenBLAS buffer exhaustion across many short-lived processes.
**Decision.** `OPENBLAS_NUM_THREADS=1` is set in `cad2ml/__init__.py` (and in the Dockerfile).
**Consequences.** 0 crashes in later full-corpus runs. Geometry work didn't benefit from BLAS threads anyway.

### D-010 — 2026-09-17 — Environment-sensitive failures are not cached as sample outcomes
**Decision.** Deterministic rejections (e.g. `UNSUPPORTED_ASSEMBLY`, `NO_SOLID`) are persisted as the sample
manifest, so resubmitting returns the same result. Timeouts and child crashes are written only to
`quarantine/<sample>/<timestamp>.json`, with the job record in state `timed_out`/`quarantined`.
**Consequences.** A slow or overloaded host can't permanently poison a valid part.

### D-011 — 2026-09-17 — Two split modes, both leakage-checked
**Context.** Six families can't give a meaningful 70/15/15 split with no family crossing splits and all
label classes represented.
**Decision.** `group` mode (the default) splits by design group `family/variant`. For each family one variant is held out
(val or test), and near-duplicate candidate clusters are merged into one unit. `family` mode holds out whole families.
Both modes assert that no split unit crosses splits, and family mode also asserts it per family.
**Consequences.** Some classes are missing from some splits (reported in the quality report). Group mode tests
generalization to unseen design variants of known families; family mode tests unseen families.

### D-012 — 2026-09-17 — Multi-view outputs use the tessellation's own vertex normals from surface UV evaluation
**Decision.** Mesh vertex normals come from evaluating the exact surface at the triangulation UV nodes, with
area-weighted triangle normals as the fallback, so normal maps reflect true curvature.

### D-013 — 2026-09-17 — Generator bug found by evaluation, fixed at the source
**Context.** Comparing the extracted bounding box with generator parameters showed 1–2.8 mm errors on
`shaft_support/rounded`. The rounded cap cylinder poked below the base for short parts. The pipeline measurement was
correct, and the ground-truth parameters were inconsistent.
**Decision.** The generator now requires `H ≥ Wu + Tb + 2`. The corpus was regenerated and reprocessed.
**Consequences.** Max bbox error dropped to 5e-7 mm. This is kept as an example of evaluation catching data bugs.

### D-014 — 2026-09-17 — GNN aggregation chosen on validation data only
**Context.** With sum aggregation the GNN peaked at epochs 4–9 and trailed the per-face MLP.
**Decision.** A 2×2 sweep (aggregation add/mean × lr 3e-3/1e-3), 3 seeds, both datasets, ranked by
**validation** macro-F1 only. Result: mean/3e-3 = 0.874, add/1e-3 = 0.821, add/3e-3 = 0.804, mean/1e-3 = 0.844
(`docs/evidence/gnn_validation_sweep.json`). Default is now `gnn_aggr="mean"`, lr 3e-3.
**Consequences.** Test metrics in the README come from runs made after this choice. Family-mode validation has no
`pocket` training examples, so its validation F1 is structurally capped. That is reported, not tuned around.

### D-015 — 2026-09-18 — RLIMIT_AS is a runaway guard, not a memory budget
**Context.** In the first container run, 71 of 72 parts were quarantined as `CHILD_CRASHED` with "OpenBLAS error:
Memory allocation still failed". A controlled test inside the worker container on the same part gave
`RLIMIT_AS`=4096 MB → crash, 16384 MB → completed, unlimited → completed. `RLIMIT_AS` caps virtual address space,
which OpenBLAS/OCCT reserve well beyond their resident use. Windows never applies the limit, so the native runs
couldn't reveal this.
**Decision.** The cap moved from the hashed `PipelineConfig` to runtime `Settings.child_memory_limit_mb`
(default 16384, env `CAD2ML_CHILD_MEMORY_LIMIT_MB`), because it must not change artifact identity. Real resident memory is
bounded by the container limit (`mem_limit: 3g` per worker in compose).
**Consequences.** The config hash changed (`a0ffc99f5a2ae63d` → `3e6d22c8bd2f5b8b`), so every sample and dataset id
changed and the evidence was regenerated. D-010 (not caching crashes as outcomes) meant none of the 71 crashed
samples was permanently poisoned.

### D-016 — 2026-09-18 — Container image built once
**Context.** The first compose build crashed Docker Desktop (SIGBUS in the VM). Three services exported the same
6 GB image in parallel while the Docker disk sat on a nearly full C: drive.
**Decision.** Only `api` has a `build:` section; `worker` and `migrate` reuse `cad2ml:1.0.0` (`pull_policy: never`).
The base image is pinned to `mambaorg/micromamba:2.9.0-ubuntu24.04`, since the originally chosen tag
`2.0.5-jammy` does not exist.
**Consequences.** The image is 6.23 GB (CadQuery/VTK + PyTorch). Slimming it is future work.
