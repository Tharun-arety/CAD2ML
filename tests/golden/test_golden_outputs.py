"""Compare pipeline outputs for fixed generated parts against reviewed golden values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cad2ml.pipeline import process_source
from cad2ml.storage.base import LocalFSStore
from tests.golden.golden_cases import CASES, close, summarize

EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text())


@pytest.mark.parametrize("fam,var,seed", CASES)
def test_golden(fam: str, var: str, seed: int, tmp_path: Path) -> None:
    from cad2ml.synthetic.corpus import export_step
    from cad2ml.synthetic.families import FAMILIES

    solid, _ = FAMILIES[fam][0](np.random.default_rng(seed), seed, var)
    p = tmp_path / "part.step"
    export_step(solid, p)
    m = process_source(p.name, p.read_bytes(), LocalFSStore(tmp_path / "s"))
    assert m.status == "completed", m.rejection
    got = summarize(m)
    exp = EXPECTED[f"{fam}/{var}/{seed}"]
    assert close(got, exp), json.dumps({"got": got, "expected": exp}, indent=1)
