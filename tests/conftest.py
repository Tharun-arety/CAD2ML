"""Shared fixtures. Heavy fixtures are session-scoped and built through the public pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _cq():
    import cadquery as cq

    return cq


@pytest.fixture(scope="session")
def box_with_hole_step(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from cad2ml.synthetic.corpus import export_step

    cq = _cq()
    d = tmp_path_factory.mktemp("prims")
    solid = cq.Workplane("XY").box(40, 30, 10).faces(">Z").workplane().hole(8).val()
    p = d / "box_with_hole.step"
    export_step(solid, p)
    return p


@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One part per family/variant (18 parts) + failure fixtures."""
    from cad2ml.synthetic.corpus import generate_corpus

    out = tmp_path_factory.mktemp("corpus")
    generate_corpus(out, per_variant=1, base_seed=500)
    return out


@pytest.fixture(scope="session")
def processed_store(mini_corpus: Path, tmp_path_factory: pytest.TempPathFactory):
    from cad2ml.pipeline import process_source
    from cad2ml.storage.base import LocalFSStore

    store = LocalFSStore(tmp_path_factory.mktemp("store"))
    manifests = {}
    for f in sorted((mini_corpus / "parts").glob("*.step")):
        manifests[f.name] = process_source(f.name, f.read_bytes(), store)
    store.manifests = manifests  # type: ignore[attr-defined]
    return store
