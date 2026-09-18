"""End-to-end reviewer demo against a running CAD2ML API.

    docker compose up --build -d
    docker compose run --rm api python scripts/demo.py --api-url http://api:8000

Steps (all through the public HTTP API, except corpus generation which only writes STEP files):
  1. generate the synthetic STEP corpus + ground-truth sidecars into CAD2ML_CORPUS_DIR
  2. upload every STEP file, submit processing jobs, wait for workers
  3. show explicit outcomes for failure fixtures
  4. trace one face across all representations
  5. build a versioned dataset (family-group split) and show quality
  6. train GNN and MLP baselines as worker jobs; wait
  7. run inference on a held-out test sample and save the prediction-vs-ground-truth panel
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TERMINAL = {"completed", "rejected", "failed_terminal", "timed_out", "quarantined"}


def step(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def wait(c: httpx.Client, job_ids: list[str], timeout: float) -> dict[str, dict[str, Any]]:
    end, out = time.time() + timeout, {}
    pending = set(job_ids)
    while pending and time.time() < end:
        for j in list(pending):
            d = c.get(f"/v1/jobs/{j}").json()
            if d["state"] in TERMINAL:
                out[j] = d
                pending.discard(j)
        if pending:
            time.sleep(2)
    if pending:
        raise TimeoutError(f"{len(pending)} jobs still running")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("CAD2ML_API_URL", "http://localhost:8000"))
    ap.add_argument("--corpus", default=os.environ.get("CAD2ML_CORPUS_DIR", "fixtures/corpus"))
    ap.add_argument("--per-variant", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default=os.environ.get("CAD2ML_DEMO_OUT", "data/demo_outputs"))
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()
    corpus, out = Path(a.corpus), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = httpx.Client(base_url=a.api_url, timeout=300)
    print(json.dumps(c.get("/health/ready").json()))

    step("1. synthetic corpus")
    if not (corpus / "corpus_index.json").exists():
        from cad2ml.synthetic.corpus import generate_corpus

        idx = generate_corpus(corpus, per_variant=a.per_variant)
        print(f"generated {sum(1 for p in idx['parts'] if 'file' in p)} parts into {corpus}")
    else:
        print(f"reusing corpus at {corpus}")

    step("2. upload + asynchronous processing")
    parts = sorted((corpus / "parts").glob("*.step"))
    jobs = {}
    for p in parts:
        f = c.post("/v1/files", files={"file": (p.name, p.read_bytes(), "application/step")}).json()
        j = c.post("/v1/jobs", json={"file_id": f["file_id"]}).json()
        jobs[j["job_id"]] = p.name
    t0 = time.time()
    done = wait(c, list(jobs), a.timeout)
    states: dict[str, int] = {}
    for d in done.values():
        states[d["state"]] = states.get(d["state"], 0) + 1
    print(f"{len(parts)} parts -> {states} in {time.time() - t0:.0f}s")

    step("3. failure fixtures")
    fjobs = {}
    for p in sorted((corpus / "failures").glob("*")):
        if p.name in ("expected.json",) or p.name.startswith("_"):
            continue
        r = c.post("/v1/files", files={"file": (p.name, p.read_bytes(), "application/step")})
        if r.status_code >= 400:
            print(f"  {p.name:40s} upload rejected: {r.json()['error']['code']}")
            continue
        j = c.post("/v1/jobs", json={"file_id": r.json()["file_id"]}).json()
        fjobs[j["job_id"]] = (p.name, j["created"])
    for jid, d in wait(c, list(fjobs), a.timeout).items():
        name, created = fjobs[jid]
        print(
            f"  {name:40s} {d['state']:12s} {d['error_code'] or ''}{'' if created else '  (idempotent: existing job)'}"
        )

    step("4. entity lineage for one face")
    sample_id = next(d["sample_id"] for d in done.values() if d["state"] == "completed")
    m = c.get(f"/v1/samples/{sample_id}").json()
    feat = next(
        (f for f in m["features"] if f["feature_type"] in ("through_hole", "blind_hole")), m["features"][0]
    )
    face = feat["participating_faces"][0]
    trace = c.get(f"/v1/samples/{sample_id}/faces/{face}/trace").json()
    (out / "trace.json").write_text(json.dumps(trace, indent=1))
    ch = trace["chain"]
    print(
        f"  {ch['point_cloud']['range_text']}\n  -> B-Rep face {face} ({ch['brep_face']['surface_type']})\n"
        f"  -> {feat['feature_id']} {feat['feature_type']} (confidence {feat['confidence']})\n"
        f"  -> source {ch['source']['filename']} ({ch['source']['sha256'][:12]}...)\n"
        f"  -> pipeline {ch['processing']['pipeline_version']}, config {ch['processing']['configuration_hash']}\n"
        f"  all links verified: {trace['all_links_verified']}"
    )
    (out / "lineage_face.png").write_bytes(c.get(f"/v1/samples/{sample_id}/views/0/highlight/{face}").content)

    step("5. dataset version")
    ds = c.post("/v1/datasets", json={"seed": 7, "split_mode": "group"}).json()
    q = c.get(f"/v1/datasets/{ds['dataset_id']}/quality").json()["quality"]
    (out / "dataset_quality.json").write_text(json.dumps(q, indent=1))
    print(
        json.dumps(
            {
                "dataset_id": ds["dataset_id"],
                "split_sizes": q["split_sizes"],
                "leakage_passed": q["leakage"]["passed"],
                "excluded": q["excluded"],
            },
            indent=1,
        )
    )

    step("6. baseline training (worker jobs)")
    runs = {}
    for model in ("gnn", "mlp"):
        r = c.post(
            "/v1/training-runs", json={"dataset_id": ds["dataset_id"], "model": model, "epochs": a.epochs}
        ).json()
        runs[model] = r
    wait(c, [r["job_id"] for r in runs.values()], a.timeout)
    results = {}
    for model, r in runs.items():
        exp = c.get(f"/v1/training-runs/{r['run_id']}").json()
        results[model] = exp
        t = exp["experiment"]["metrics"]["test"]["summary"]
        print(
            f"  {model}: run {r['run_id']} test accuracy {t['accuracy']} macro-F1 {t['macro_f1_present_classes']} "
            f"({t['n_faces']} faces)"
        )
    (out / "training_runs.json").write_text(json.dumps(results, indent=1))

    step("7. inference on a held-out test sample")
    dsd = c.get(f"/v1/datasets/{ds['dataset_id']}").json()
    test_id = dsd["splits"]["test"][0]
    run_id = runs["gnn"]["run_id"]
    pred = c.get(f"/v1/training-runs/{run_id}/predictions/{test_id}").json()
    wrong = [p for p in pred["predictions"] if p["ground_truth"] != p["predicted"]]
    (out / "prediction.json").write_text(json.dumps(pred, indent=1))
    (out / "prediction_panel.png").write_bytes(
        c.get(f"/v1/training-runs/{run_id}/predictions/{test_id}/panel.png").content
    )
    print(
        f"  sample {test_id} (split={pred['dataset_split']}): {len(pred['predictions'])} faces, {len(wrong)} wrong"
    )
    print(f"\nOutputs written to {out.resolve()}. Inspector UI: {a.api_url}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
