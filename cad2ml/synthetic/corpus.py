"""Synthetic corpus + failure fixtures, written as STEP files with ground-truth sidecars.

Layout:
    <out>/parts/<family>__<variant>__<seed>.step
    <out>/parts/<family>__<variant>__<seed>.gt.json   (authoritative ground truth)
    <out>/failures/<case>.<ext>  + failures/expected.json (expected error code per case)
    <out>/corpus_index.json

The generator only writes files. Processing must go through the public pipeline.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from cad2ml.synthetic.families import FAMILIES


def _cq() -> Any:
    import cadquery as cq

    return cq


def export_step(shape: Any, path: Path) -> None:
    cq = _cq()
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(cq.Workplane("XY").add(shape), str(path), exportType="STEP")
    _freeze_header_timestamp(path)


def export_step_single_product(shape: Any, path: Path) -> None:
    """Write all bodies into ONE product (no assembly structure), unlike the default OCCT writer mode
    which turns a compound into a NEXT_ASSEMBLY_USAGE_OCCURRENCE product tree."""
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    Interface_Static.SetIVal_s("write.step.assembly", 0)
    w = STEPControl_Writer()
    w.Transfer(shape.wrapped, STEPControl_AsIs)
    path.parent.mkdir(parents=True, exist_ok=True)
    w.Write(str(path))
    Interface_Static.SetIVal_s("write.step.assembly", 2)
    _freeze_header_timestamp(path)


def _freeze_header_timestamp(path: Path) -> None:
    """Replace the export timestamp in FILE_NAME so regenerated corpora are byte-identical."""
    import re

    text = path.read_bytes().decode("latin-1")
    text = re.sub(r"(FILE_NAME\s*\(\s*'[^']*'\s*,\s*)'[^']*'", r"\g<1>'2000-01-01T00:00:00'", text, count=1)
    path.write_bytes(text.encode("latin-1"))


def generate_parts(out: Path, per_variant: int, base_seed: int = 1000) -> list[dict[str, Any]]:
    entries = []
    for fi, (family, (gen, variants)) in enumerate(sorted(FAMILIES.items())):
        for vi, variant in enumerate(variants):
            made = 0
            attempt = 0
            while made < per_variant and attempt < per_variant * 5:
                seed = base_seed + fi * 10000 + vi * 1000 + attempt
                attempt += 1
                rng = np.random.default_rng(seed)
                try:
                    solid, gt = gen(rng, seed, variant)
                except Exception as exc:  # OCCT boolean/fillet failure for this parameter draw
                    entries.append(
                        {"family": family, "variant": variant, "seed": seed, "skipped": str(exc)[:200]}
                    )
                    continue
                if not solid.isValid() or len(solid.Solids()) != 1:
                    entries.append(
                        {
                            "family": family,
                            "variant": variant,
                            "seed": seed,
                            "skipped": "generator produced invalid or multi-solid shape",
                        }
                    )
                    continue
                name = f"{family}__{variant}__{seed}"
                step = out / "parts" / f"{name}.step"
                export_step(solid, step)
                gtd = gt.to_dict()
                gtd["step_file"] = step.name
                gtd["source_sha256"] = hashlib.sha256(step.read_bytes()).hexdigest()
                gtd["group"] = f"{family}/{variant}"
                (out / "parts" / f"{name}.gt.json").write_text(json.dumps(gtd, indent=1))
                entries.append(
                    {
                        "family": family,
                        "variant": variant,
                        "seed": seed,
                        "file": f"parts/{step.name}",
                        "sha256": gtd["source_sha256"],
                    }
                )
                made += 1
    return entries


def generate_failures(out: Path, sample_part: Path | None) -> dict[str, dict[str, str]]:
    cq = _cq()
    d = out / "failures"
    d.mkdir(parents=True, exist_ok=True)
    expected: dict[str, dict[str, str]] = {}

    good = d / "_good_plate.step"
    export_step(cq.Solid.makeBox(40, 30, 5), good)
    data = good.read_bytes()

    (d / "empty.step").write_bytes(b"")
    expected["empty.step"] = {"code": "EMPTY_FILE", "why": "zero bytes"}

    cut = int(len(data) * 0.55)
    (d / "corrupt_truncated.step").write_bytes(data[:cut] + b"\n#99999=GARBAGE((((;\n")
    expected["corrupt_truncated.step"] = {
        "code": "STEP_PARSE_FAILED|STEP_NO_SHAPES|NO_SOLID|INVALID_GEOMETRY",
        "why": "truncated DATA section with junk entity",
    }

    (d / "wrong_extension.txt").write_bytes(data)
    expected["wrong_extension.txt"] = {
        "code": "UNSUPPORTED_EXTENSION",
        "why": "valid STEP content, .txt name",
    }

    (d / "png_renamed.step").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    expected["png_renamed.step"] = {"code": "NOT_STEP_CONTENT", "why": "PNG magic with .step extension"}

    asm = cq.Assembly(name="asm")
    asm.add(cq.Workplane().box(20, 20, 5), name="base")
    asm.add(cq.Workplane().cylinder(15, 4), name="pin", loc=cq.Location(cq.Vector(0, 0, 10)))
    asm.save(str(d / "assembly_two_parts.step"), exportType="STEP")
    _freeze_header_timestamp(d / "assembly_two_parts.step")
    expected["assembly_two_parts.step"] = {
        "code": "UNSUPPORTED_ASSEMBLY",
        "why": "XCAF assembly with 2 components",
    }

    comp = cq.Compound.makeCompound(
        [cq.Solid.makeBox(10, 10, 10), cq.Solid.makeBox(10, 10, 10).translate(cq.Vector(30, 0, 0))]
    )
    export_step_single_product(comp, d / "multi_body.step")
    expected["multi_body.step"] = {"code": "MULTI_BODY", "why": "two disjoint solids in one product"}
    export_step(comp, d / "compound_exported_as_assembly.step")
    expected["compound_exported_as_assembly.step"] = {
        "code": "UNSUPPORTED_ASSEMBLY",
        "why": "same two solids, written by OCCT default mode as a 2-product tree",
    }

    box = cq.Solid.makeBox(20, 20, 20)
    open_shell = cq.Shell.makeShell(box.Faces()[:-1])
    export_step(open_shell, d / "open_shell.step")
    expected["open_shell.step"] = {"code": "NO_SOLID", "why": "box shell with one face removed"}

    if sample_part is not None and sample_part.exists():
        shutil.copyfile(sample_part, d / "duplicate_of_part.step")
        expected["duplicate_of_part.step"] = {
            "code": "DUPLICATE",
            "why": "byte-identical copy of a corpus part (same sha256)",
        }

    tiny = cq.Workplane("XY").box(50, 50, 5).faces(">Z").workplane().hole(0.2).val()
    export_step(tiny, d / "tiny_feature_hole_0p2mm.step")
    expected["tiny_feature_hole_0p2mm.step"] = {
        "code": "COMPLETED",
        "why": "0.2 mm through hole must be sampled",
    }

    big = cq.Workplane("XY").box(400, 400, 10).faces(">Z").workplane().rarray(25, 25, 14, 14).hole(6).val()
    export_step(big, d / "large_plate_196_holes.step")
    expected["large_plate_196_holes.step"] = {"code": "COMPLETED", "why": "large face count (≈200 faces)"}

    expected["timeout_simulation"] = {
        "code": "PARSER_TIMEOUT",
        "why": "any valid file with test-only fault_parse_sleep_s > parse timeout",
    }
    (d / "expected.json").write_text(json.dumps(expected, indent=1))
    return expected


def generate_corpus(out: Path, per_variant: int = 4, base_seed: int = 1000) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    parts = generate_parts(out, per_variant, base_seed)
    first = next((out / e["file"] for e in parts if "file" in e), None)
    failures = generate_failures(out, first)
    index = {"per_variant": per_variant, "base_seed": base_seed, "parts": parts, "failures": failures}
    (out / "corpus_index.json").write_text(json.dumps(index, indent=1))
    return index
