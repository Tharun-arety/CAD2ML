# STATUS

_Last updated: 2026-09-18. Every result below comes from a command run in this repository. The raw outputs are in
`docs/evidence/`._

## Where each release stands

All gates are verified in two environments:
* **native**: Windows 11, conda env `cad2ml`, SQLite + fakeredis TCP
* **container**: Docker Desktop 29.8 / WSL2, `docker compose` with PostgreSQL 16 + Valkey 8 + 2 workers

| Release | State | How the gate was verified |
|---|---|---|
| R0 Foundation | ✅ native · ✅ container | `scripts/spike_occt.py` passed natively and during `docker build` (`SPIKE OK` inside the image). |
| R1 Canonical CAD processing | ✅ · ✅ | 83 files → 75 completed, 7 rejected, 1 quarantined, each with its expected code (`docs/evidence/corpus_report.json`; same outcomes in the compose demo and benchmark). |
| R2 Aligned representations | ✅ · ✅ | `test_trace_face_across_all_representations` passes in both environments. The demo trace shows `all links verified: True` under config `3e6d22c8bd2f5b8b`. |
| R3 Semantics & datasets | ✅ · ✅ | Identical-rebuild and leakage tests pass. Group and family datasets are built (`docs/evidence/baselines.json`). |
| R4 Production operation | ✅ · ✅ | Worker hard-kill recovery, queue-loss reconciliation and duplicate delivery pass against **real PostgreSQL + Valkey** (5/5). Compose benchmark: p50 2.96 s / p95 3.70 s, 41.0 files/min. |
| R5 ML proof & reviewer experience | ✅ · ✅ | `docker compose up --build` followed by `docker compose run --rm api python scripts/demo.py --api-url http://api:8000` ran end to end (`docs/evidence/demo_compose_run.txt`). The inspector UI was checked in a browser against the native stack. |

## Tests

* Native: **66 passed** (151 s). Linux container: **66 passed** (91 s). Jobs/API/worker suite against PostgreSQL +
  Valkey: **5 passed**. ruff format, ruff check and mypy (57 files) are clean.
* The golden geometry values reviewed on Windows match exactly on Linux (6/6).

| Suite | Tests | Covers |
|---|---|---|
| unit/test_core_units | 23 | hashing, sanitization, intake rejections, units, config hash, storage path safety and atomic promote, canonical ordering, fingerprints, schema drift |
| unit/test_math_and_splits | 8 | property tests: normalization inverse, allocation, split leakage, tool SDF, state machine, metrics, graph invariants |
| geometry/test_brep_geometry | 11 | measurements, concavity, traversal-invariant IDs, repair vs quarantine, correspondence, 0.2 mm hole, determinism, face-ID views |
| integration/test_pipeline_and_lineage | 9 | upload → manifest, idempotency, lineage trace, invalid inputs, all failure fixtures, timeout, no partial artifacts |
| integration/test_jobs_api_worker | 5 | API idempotency, correlation ids, stage events, worker hard-kill recovery, queue loss, rejection/timeout, metrics |
| integration/test_dataset_and_training | 4 | identical rebuild, leakage, label alignment, recognizer vs ground truth, training determinism, prediction → face |
| golden/test_golden_outputs | 6 | reviewed outputs for six fixed parts |

## Measured results

* **Recognizer vs synthetic ground truth** (72 parts, 1094 faces): face-label accuracy 1.000; holes 283/283, slots
  27/27, pockets 12/12, no false positives, hole diameter error 0.0 mm, max bbox error 5.0e-7 mm. The rules were
  developed on these families, so this is a consistency check, not a generalization claim.
* **Baselines** (3 seeds, test accuracy, `docs/evidence/baselines.json`):

  | Split | Rule | MLP | GNN |
  |---|---|---|---|
  | group | 1.000 | 1.000 ± 0.000 | 0.934 ± 0.004 |
  | family | 1.000 | 0.971 ± 0.024 | 0.953 ± 0.052 |

  The GNN is sensitive to data order and platform (an earlier run gave 0.983/0.922, and the Linux seed-0 run gave 0.918).
  It does not beat the MLP.
* **Benchmark** (2 workers, 83 files):

  | Stack | Throughput | Latency p50 / p95 | Idempotent rerun |
  |---|---|---|---|
  | compose | 41.0 files/min | 2.96 / 3.70 s | 0 new jobs |
  | native | 18.3 files/min | 3.39 / 3.84 s | 0 new jobs |

## Issues found by the container run (fixed)

1. The base image tag `micromamba:2.0.5-jammy` didn't exist. Pinned `2.9.0-ubuntu24.04`.
2. Parallel export of the same 6 GB image by three services, with Docker's disk on a nearly full C:, crashed the VM (SIGBUS).
   The image is now built once (D-016), and the Docker disk was moved to A:.
3. **71 of 72 parts crashed in containers**: a 4 GB `RLIMIT_AS` broke OpenBLAS address-space reservations (Linux
   only). A controlled experiment confirmed it (D-015). The cap moved to runtime settings (16 GB), and the compose
   `mem_limit` bounds real memory. The config hash changed, so all evidence was regenerated.
4. `schemas/` wasn't copied into the image (the schema-drift test failed in the container). Fixed.

## Known limitations

* Synthetic, axis-aligned, prismatic parts only (6 families, 18 variants). There has been no real-world CAD evaluation yet.
* The recognizer has a limited vocabulary and is tuned on the evaluated families.
* Canonical IDs aren't stable across edits. Revision matching is experimental.
* The image is 6.23 GB. Peak memory inside compose isn't measured.
* Test sets are 12 parts, and classes are unevenly present across splits.
* The GitHub Actions workflow is written but has not run (no remote repository has been configured).

## Next actions

1. Push to a Git remote so the CI workflow runs.
2. R6: real-world STEP evaluation with an audited label sample.
3. A pre-forked child pool to cut the per-job isolation overhead.
