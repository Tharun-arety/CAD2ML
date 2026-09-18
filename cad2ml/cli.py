"""Command-line interface.

cad2ml generate-corpus --out fixtures/corpus --per-variant 4
cad2ml process FILE [--data data]
cad2ml process-corpus --corpus fixtures/corpus [--report docs/corpus_report.json]
cad2ml trace SAMPLE_ID FACE_ID
cad2ml build-dataset --corpus fixtures/corpus [--seed 7]
cad2ml train --dataset DATASET_ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from cad2ml.config import get_settings
from cad2ml.observability.logging import configure_logging


def _store(data_dir: str | None) -> Any:
    from cad2ml.storage.base import LocalFSStore

    return LocalFSStore(Path(data_dir) if data_dir else get_settings().data_dir)


def cmd_generate(a: argparse.Namespace) -> int:
    from cad2ml.synthetic.corpus import generate_corpus

    idx = generate_corpus(Path(a.out), a.per_variant, a.seed)
    made = sum(1 for p in idx["parts"] if "file" in p)
    skipped = [p for p in idx["parts"] if "skipped" in p]
    print(
        json.dumps({"parts": made, "skipped_draws": len(skipped), "failure_fixtures": len(idx["failures"])})
    )
    return 0


def cmd_process(a: argparse.Namespace) -> int:
    from cad2ml.pipeline import process_source

    p = Path(a.file)
    m = process_source(p.name, p.read_bytes(), _store(a.data), reuse_existing=not a.force)
    print(m.model_dump_json(include={"sample_id", "status", "rejection", "geometry", "validation"}, indent=1))
    return 0 if m.status == "completed" else 2


def cmd_process_corpus(a: argparse.Namespace) -> int:
    from cad2ml.pipeline import process_source

    store = _store(a.data)
    corpus = Path(a.corpus)
    files = sorted((corpus / "parts").glob("*.step")) + sorted(
        p for p in (corpus / "failures").glob("*") if p.name != "expected.json" and not p.name.startswith("_")
    )
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for f in files:
        t0 = time.perf_counter()
        m = process_source(f.name, f.read_bytes(), store)
        dup = seen.get(m.source.sha256)
        seen.setdefault(m.source.sha256, f.name)
        rows.append(
            {
                "file": str(f.relative_to(corpus)),
                "sample_id": m.sample_id,
                "status": m.status.value,
                "code": m.rejection.code if m.rejection else None,
                "duplicate_of": dup,
                "seconds": round(time.perf_counter() - t0, 3),
                "faces": m.geometry.face_count if m.geometry else None,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    summary: dict[str, Any] = {"files": len(rows)}
    for r in rows:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    summary["duplicates"] = sum(1 for r in rows if r["duplicate_of"])
    report = {"summary": summary, "rows": rows}
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(summary))
    return 0


def cmd_trace(a: argparse.Namespace) -> int:
    from cad2ml.lineage import trace_face

    print(json.dumps(trace_face(_store(a.data), a.sample_id, a.face_id), indent=1))
    return 0


def cmd_build_dataset(a: argparse.Namespace) -> int:
    from cad2ml.datasets.builder import build_dataset

    ds = build_dataset(_store(a.data), Path(a.corpus), seed=a.seed, split_mode=a.split_mode)
    print(
        json.dumps(
            {
                "dataset_id": ds["dataset_id"],
                "splits": {k: len(v) for k, v in ds["splits"].items()},
                "excluded": len(ds["excluded"]),
            }
        )
    )
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from cad2ml.training.train import TrainConfig, run_training

    cfg = TrainConfig(model=a.model, epochs=a.epochs, seed=a.seed)
    res = run_training(_store(a.data), a.dataset, cfg)
    print(json.dumps({"run_id": res["run_id"], "test": res["metrics"]["test"]["summary"]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    ap = argparse.ArgumentParser(prog="cad2ml")
    ap.add_argument("--data", default=None, help="artifact store root (default: CAD2ML_DATA_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-corpus")
    g.add_argument("--out", default="fixtures/corpus")
    g.add_argument("--per-variant", type=int, default=4)
    g.add_argument("--seed", type=int, default=1000)
    g.set_defaults(fn=cmd_generate)
    p = sub.add_parser("process")
    p.add_argument("file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_process)
    pc = sub.add_parser("process-corpus")
    pc.add_argument("--corpus", default="fixtures/corpus")
    pc.add_argument("--report", default=None)
    pc.set_defaults(fn=cmd_process_corpus)
    t = sub.add_parser("trace")
    t.add_argument("sample_id")
    t.add_argument("face_id")
    t.set_defaults(fn=cmd_trace)
    b = sub.add_parser("build-dataset")
    b.add_argument("--corpus", default="fixtures/corpus")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--split-mode", default="group", choices=["group", "family"])
    b.set_defaults(fn=cmd_build_dataset)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--model", default="gnn", choices=["gnn", "mlp"])
    tr.add_argument("--epochs", type=int, default=150)
    tr.add_argument("--seed", type=int, default=0)
    tr.set_defaults(fn=cmd_train)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
