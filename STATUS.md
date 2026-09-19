# STATUS

_Last updated: 2026-09-19 (pipeline 1.1.0). Every result below comes from a command run in this repository. The raw outputs are in
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
| R4 Production operation | ✅ · ✅ | Worker hard-kill recovery, queue-loss reconciliation and duplicate delivery pass against **real PostgreSQL + Valkey** (5/5). Compose benchmark: p50 2.23 s / p95 2.41 s, 54.9 files/min. |
| R5 ML proof & reviewer experience | ✅ · ✅ | `docker compose up --build` followed by `docker compose run --rm api python scripts/demo.py --api-url http://api:8000` ran end to end (`docs/evidence/demo_compose_run.txt`). The inspector UI was checked in a browser against the native stack. |
| R6 Real-world robustness | 🔶 in progress | NIST MBE PMI models: 32/33 complete, 1 rejected (`UNSUPPORTED_TESSELLATED`). Cross-export consistency is measured (`docs/evidence/r6_nist_eval.json`, [docs/r6_real_world.md](docs/r6_real_world.md)). Open: a hole-by-hole audit against the NIST drawings/PMI. |

GitHub Actions: [run 35409127088](https://github.com/Tharun-arety/CAD2ML/actions/runs/35409127088) on `5b12e52` is
green. `quality-and-tests` covers format, lint, types, dependency audit, and unit/geometry/integration tests on
ubuntu-latest. `container` covers the image build, the compose smoke test and the PostgreSQL/Valkey suite. (The first
run's container job failed at setup on an invalid `trivy-action` tag, since fixed.)

## Tests

* Native: **72 passed** (109 s). Linux container: **72 passed** (85 s). Jobs/API/worker suite against PostgreSQL +
  Valkey: **5 passed**. ruff format, ruff check and mypy (57 files) are clean.
* R6 added: `.stp` discovery, auxiliary-geometry isolation, cone-tip blind holes, counterbored holes, forced-reprocess
  persistence, and tolerant near-duplicate matching.
* The golden geometry values reviewed on Windows match exactly on Linux (6/6).

| Suite | Tests | Covers |
|---|---|---|
| unit/test_core_units | 24 | `.stp` discovery, hashing, sanitization, intake rejections, units, config hash, storage path safety and atomic promote, canonical ordering, fingerprints, schema drift |
| unit/test_math_and_splits | 9 | property tests: normalization inverse, allocation, split leakage, near-duplicate matching, tool SDF, state machine, metrics, graph invariants |
| geometry/test_brep_geometry | 14 | measurements, concavity, traversal-invariant IDs, repair vs quarantine, auxiliary-geometry isolation, cone-tip blind hole, counterbored hole, correspondence, 0.2 mm hole, determinism, face-ID views |
| integration/test_pipeline_and_lineage | 10 | upload → manifest, idempotency, lineage trace, invalid inputs, all failure fixtures, timeout, no partial artifacts, forced reprocess |
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
  | group | 1.000 | 0.997 ± 0.005 | 0.986 ± 0.019 |
  | family | 1.000 | 0.971 ± 0.013 | 0.917 ± 0.054 |

  The GNN varies with data order and platform (0.934–0.986 group, 0.917–0.953 family across reruns). It does not beat
  the MLP.
* **Benchmark** (2 workers, 83 files):

  | Stack | Throughput | Latency p50 / p95 | Idempotent rerun |
  |---|---|---|---|
  | compose | 54.9 files/min | 2.23 / 2.41 s | 0 new jobs |
  | native | 21.0 files/min | 2.98 / 3.30 s | 0 new jobs |

* **Real-world (NIST, R6):** 32/33 files complete, volume agrees across exports for 10/11 parts, hole counts for 9/11,
  canonical IDs for 0/11 (exporters split faces differently), 15.3 % of faces `unknown`.

## Issues found by the container run (fixed)

1. The base image tag `micromamba:2.0.5-jammy` didn't exist. Pinned `2.9.0-ubuntu24.04`.
2. Parallel export of the same 6 GB image by three services, with Docker's disk on a nearly full C:, crashed the VM (SIGBUS).
   The image is now built once (D-016), and the Docker disk was moved to A:.
3. **71 of 72 parts crashed in containers**: a 4 GB `RLIMIT_AS` broke OpenBLAS address-space reservations (Linux
   only). A controlled experiment confirmed it (D-015). The cap moved to runtime settings (16 GB), and the compose
   `mem_limit` bounds real memory. The config hash changed, so all evidence was regenerated.
4. `schemas/` wasn't copied into the image (the schema-drift test failed in the container). Fixed.

## Known limitations

* Real-world evaluation covers 11 NIST parts, with no hole-by-hole ground truth yet. The training data is synthetic only.
* The recognizer has a limited vocabulary and is tuned on the evaluated families.
* Canonical IDs aren't stable across edits or across STEP exporters (D-017). Revision matching is experimental.
* The image is 6.23 GB. Peak memory inside compose isn't measured.
* Test sets are 12 parts, and classes are unevenly present across splits.

## Next actions

1. R6: hole-by-hole comparison against the NIST drawings / AP242 semantic PMI (hole callouts).
2. R6: the Fusion 360 Gallery segmentation subset (real per-face operation labels) once download is approved.
3. A pre-forked child pool to cut per-job isolation overhead.
