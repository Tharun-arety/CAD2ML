"""R0 dependency spike: prove the CAD kernel stack works before committing to it.

CadQuery box with a through hole -> STEP -> re-read with OCP -> B-Rep traversal ->
surface typing -> tessellation -> triangle/face correspondence. Exits non-zero on failure.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    t0 = time.perf_counter()
    import cadquery as cq
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    part = cq.Workplane("XY").box(40, 30, 10).faces(">Z").workplane().hole(8)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "spike.step"
        cq.exporters.export(part, str(path))
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(path))
        assert status == IFSelect_RetDone, f"read failed: {status}"
        reader.TransferRoots()
        shape = reader.OneShape()

    def count(kind: object) -> int:
        exp = TopExp_Explorer(shape, kind)
        n = 0
        while exp.More():
            n += 1
            exp.Next()
        return n

    counts = {
        k: count(v)
        for k, v in [
            ("solid", TopAbs_SOLID),
            ("face", TopAbs_FACE),
            ("edge", TopAbs_EDGE),
            ("vertex", TopAbs_VERTEX),
        ]
    }
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = props.Mass()
    expected = 40 * 30 * 10 - np.pi * 16 * 10
    valid = BRepCheck_Analyzer(shape).IsValid()

    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    types: dict[str, int] = {}
    tri_total = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        st = BRepAdaptor_Surface(face).GetType()
        name = GeomAbs_SurfaceType(st).name if hasattr(GeomAbs_SurfaceType(st), "name") else str(st)
        types[name] = types.get(name, 0) + 1
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        assert tri is not None, "face without triangulation"
        tri_total += tri.NbTriangles()
        exp.Next()

    print(
        {
            "counts": counts,
            "volume": round(volume, 3),
            "expected": round(expected, 3),
            "valid": valid,
            "surface_types": types,
            "triangles": tri_total,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    )
    assert counts["solid"] == 1 and counts["face"] == 7
    assert abs(volume - expected) / expected < 1e-3
    assert valid
    print("SPIKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
