"""Regenerate tests/golden/expected.json. Review the diff before committing: golden files are reviewed fixtures."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad2ml.pipeline import process_source  # noqa: E402
from cad2ml.storage.base import LocalFSStore  # noqa: E402
from cad2ml.synthetic.corpus import export_step  # noqa: E402
from cad2ml.synthetic.families import FAMILIES  # noqa: E402
from tests.golden.golden_cases import CASES, summarize  # noqa: E402


def main() -> int:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        store = LocalFSStore(Path(td) / "store")
        for fam, var, seed in CASES:
            solid, _ = FAMILIES[fam][0](np.random.default_rng(seed), seed, var)
            p = Path(td) / f"{fam}_{var}_{seed}.step"
            export_step(solid, p)
            m = process_source(p.name, p.read_bytes(), store)
            assert m.status == "completed", m.rejection
            out[f"{fam}/{var}/{seed}"] = summarize(m)
    (ROOT / "tests/golden/expected.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
