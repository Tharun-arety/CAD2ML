# Rules for coding agents working on CAD2ML

1. **Evidence over claims.** Every number in README/STATUS must come from a command in this repository, and the
   output artifact behind it lives in `docs/evidence/`. Never type in accuracy, latency or counts by hand.
2. **Provenance vocabulary is part of the schema.** Keep `observed` (read from STEP), `computed`
   (deterministic geometry), `inferred` (heuristic), `ground_truth` (synthetic sidecar) and `unavailable`
   separate. Never put semantic claims in `observations`, and never put measurements in `features`.
3. **Do not claim parametric-history recovery from STEP**, assembly support, or formats other than STEP.
4. **Kernel code runs in isolated children** (`cad2ml/parsers/isolation.py`). Don't import OCP in the API
   process or at module import time of `cad2ml.pipeline`.
5. **Never break lineage.** Node order equals canonical face order; point and triangle ranges are contiguous per face.
   If you change any representation, update `lineage.py` verification and `tests/integration/test_pipeline_and_lineage.py`.
6. **Identity and immutability.** Anything that changes derived bytes must change `PipelineConfig` (so the
   config hash changes) or bump `PIPELINE_VERSION`/`SCHEMA_VERSION`. Never overwrite `sources/`,
   `samples/<id>/` or `datasets/<id>/`.
7. **Job state changes only through `Database.transition`** (compare-and-set + event row). Add new states to
   `cad2ml/jobs/states.py` and its tests.
8. **Tests avoid traversal order.** Compare geometry with tolerances. Golden files (`tests/golden/expected.json`)
   are regenerated only via `scripts/update_golden.py`, and the diff must be reviewed.
9. **Model selection uses validation data only.** Test metrics are reported, never tuned against.
10. **Before finishing:** `ruff format --check`, `ruff check`, `mypy cad2ml apps`, `pytest`. Update `STATUS.md`,
    and append to `DECISIONS.md` for material changes.

Environment: see `README.md` (Docker) or `scripts/setup_env_windows.ps1` (native Windows).
