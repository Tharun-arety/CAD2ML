# CAD2ML roadmap

Releases run in order. A release is done only when its gate has been demonstrated by a command whose output is
recorded (see `STATUS.md` for where each gate stands and how it was verified).

| Release | Scope | Depends on | Acceptance gate |
|---|---|---|---|
| **R0 Foundation** | repo layout, conda/Docker environment, OCCT spike, config + hashing, schemas, structured logging, storage interface, synthetic generator, compose, CI | — | A generated STEP file can be created and parsed in the target container (`scripts/spike_occt.py`; also runs during `docker build`) |
| **R1 Canonical CAD processing** | intake/hashing/limits, isolated parser with timeout, assembly/multi-body policy, units, metrics, validation + recorded repair, quarantine, canonical faces/edges, manifest | R0 | The generated corpus is processed with schema-valid manifests and explicit failure codes (`cad2ml process-corpus`) |
| **R2 Aligned representations** | face graph, tessellation, point cloud, multi-view, lineage index + verification | R1 | A selected face traces through every representation, and each link is verified against the arrays (`test_trace_face_across_all_representations`) |
| **R3 Semantics & datasets** | rule recognizer with evidence, ground-truth sidecars + face labels, recognizer evaluation, dedupe, family-aware splits, immutable dataset versions, quality report | R2 | A reproducible dataset rebuilds to the same id with no split leakage (`test_dataset_rebuild_is_identical_and_leak_free`) |
| **R4 Production operation** | API, durable jobs, leases/heartbeats, retry and idempotency, metrics, reaper/reconciler, benchmark, fault injection | R1–R3 | The pipeline recovers from a worker kill, and the benchmark reports measured p50/p95 (`test_worker_killed_mid_job_is_recovered`, `scripts/benchmark.py`) |
| **R5 ML proof & reviewer experience** | GNN + MLP baselines, evaluation report, prediction→face traceability, inspection UI, README, demo | R3–R4 | A fresh reviewer can start the stack, process CAD, build a dataset, train, run inference and inspect (`docker compose up --build` + `scripts/demo.py`) |

## Next releases (not implemented)

- **R6 Real-world robustness.** Evaluate on public, permissively licensed STEP collections (e.g. ABC/Fusion 360
  Gallery subsets, license permitting). Measure recognizer precision where no synthetic ground truth exists
  (manual audit sample). Add BSpline-heavy and imported/tolerance-degraded geometry to the failure corpus.
- **R7 Recognizer v2.** Counterbores/countersinks, chamfers, bosses, ribs, non-axis-aligned slots, and pockets with
  islands. Calibrated confidence (reliability curves against audited labels).
- **R8 Learning.** A pose-invariant feature set for the GNN, UV-grid face encodings (UV-Net style), a point-cloud
  segmentation baseline that uses per-point face ids, and multi-view segmentation from face-ID masks.
- **R9 Revision matching.** Validate fingerprint matching on parameter-perturbed revision pairs; report matching
  precision/recall before removing the "experimental" label.
- **R10 Adapters & scale.** An S3 backend for `ArtifactStore`, an IGES adapter behind `CadAdapter`, assembly
  decomposition into per-part samples, horizontal worker scaling with a measured throughput curve.
