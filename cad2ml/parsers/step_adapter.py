"""STEP adapter (the only CAD format implemented in v1).

``CadAdapter`` is the boundary for future IGES / native-CAD adapters; none exist yet.
``parse_step_child`` runs inside an isolated process (see ``parsers.isolation``). It
reads the file with OCCT, classifies the top-level structure, converts to millimetres
and writes the transferred shape as an OCCT ``.brep`` for the extraction stage.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

from cad2ml.errors import PipelineError

_UNIT_PATTERNS: list[tuple[str, str]] = [
    (r"SI_UNIT\s*\(\s*\.MILLI\.\s*,\s*\.METRE\.\s*\)", "mm"),
    (r"SI_UNIT\s*\(\s*\.CENTI\.\s*,\s*\.METRE\.\s*\)", "cm"),
    (r"SI_UNIT\s*\(\s*\.MICRO\.\s*,\s*\.METRE\.\s*\)", "um"),
    (r"SI_UNIT\s*\(\s*\$\s*,\s*\.METRE\.\s*\)", "m"),
    (r"CONVERSION_BASED_UNIT\s*\(\s*'INCH'", "inch"),
    (r"CONVERSION_BASED_UNIT\s*\(\s*'FOOT'", "foot"),
]


class CadAdapter(Protocol):
    format_name: str
    extensions: tuple[str, ...]

    def parse(self, source: Path, out_brep: Path, timeout_s: float) -> dict[str, Any]: ...


def kernel_version() -> str:
    try:
        from OCP.Standard import Standard_Version  # type: ignore[attr-defined]

        return f"OCCT {Standard_Version.String_s()}"
    except Exception:
        try:
            import OCP

            return f"OCP {getattr(OCP, '__version__', 'unknown')}"
        except Exception:
            return "unknown"


def detect_length_unit(text: str) -> str | None:
    """Best-effort observation of the declared STEP length unit (first match in file order)."""
    best: tuple[int, str] | None = None
    for pat, unit in _UNIT_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), unit)
    return best[1] if best else None


def count_entities(text: str, name: str) -> int:
    return len(re.findall(rf"=\s*{name}\s*\(", text, re.IGNORECASE))


def _face_has_no_surface(face: Any) -> bool:
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS

    try:
        return BRep_Tool.Surface_s(TopoDS.Face_s(face)) is None
    except Exception:
        return True


def parse_step_child(source: str, out_brep: str, fault_sleep_s: float = 0.0) -> dict[str, Any]:
    """Isolated-child entry point. Returns observations; writes ``out_brep``."""
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    if fault_sleep_s > 0:  # test-only fault injection (enabled explicitly by the caller)
        time.sleep(fault_sleep_s)

    t0 = time.perf_counter()
    raw = Path(source).read_bytes()
    text = raw.decode("latin-1", errors="replace")
    nauo = count_entities(text, "NEXT_ASSEMBLY_USAGE_OCCURRENCE")
    declared_unit = detect_length_unit(text)

    Interface_Static.SetCVal_s("xstep.cad.unit", "MM")
    reader = STEPControl_Reader()
    status = reader.ReadFile(source)
    if status != IFSelect_RetDone:
        raise PipelineError("STEP_PARSE_FAILED", f"STEPControl_Reader status {int(status)}", "parsing")
    n_roots = reader.NbRootsForTransfer()
    if n_roots == 0:
        raise PipelineError("STEP_NO_SHAPES", "no transferable roots", "parsing")
    if nauo > 0:
        raise PipelineError(
            "UNSUPPORTED_ASSEMBLY",
            f"{nauo} NEXT_ASSEMBLY_USAGE_OCCURRENCE entities; assemblies are unsupported in v1",
            "parsing",
        )
    reader.TransferRoots()
    n_shapes = reader.NbShapes()
    if n_shapes == 0:
        raise PipelineError("STEP_NO_SHAPES", "transfer produced no shapes", "parsing")
    shape = reader.OneShape()
    if shape.IsNull():
        raise PipelineError("STEP_NO_SHAPES", "transferred shape is null", "parsing")

    def n_sub(kind: Any) -> int:
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, m)
        return m.Extent()

    solids, shells, faces = n_sub(TopAbs_SOLID), n_sub(TopAbs_SHELL), n_sub(TopAbs_FACE)
    if solids > 1:
        raise PipelineError("MULTI_BODY", f"{solids} solids found; v1 accepts exactly one solid", "parsing")
    if faces == 0:
        raise PipelineError("STEP_NO_SHAPES", "no faces in transferred shape", "parsing")
    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    no_surface = sum(1 for i in range(1, fmap.Extent() + 1) if _face_has_no_surface(fmap.FindKey(i)))
    if no_surface:
        raise PipelineError(
            "UNSUPPORTED_TESSELLATED",
            f"{no_surface}/{faces} faces carry no surface geometry (tessellated representation)",
            "parsing",
        )

    os.makedirs(os.path.dirname(out_brep), exist_ok=True)
    if not BRepTools.Write_s(shape, out_brep):
        raise PipelineError("INTERNAL_ERROR", "failed to serialise BRep", "parsing")
    return {
        "parser_version": kernel_version(),
        "declared_length_unit": declared_unit,
        "roots": int(n_roots),
        "transferred_shapes": int(n_shapes),
        "solids": solids,
        "shells": shells,
        "faces": faces,
        "assembly_occurrences": nauo,
        "parse_seconds": round(time.perf_counter() - t0, 6),
    }
