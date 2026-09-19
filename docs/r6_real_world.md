# R6: real-world STEP evaluation (NIST MBE PMI test models)

**Data.** The NIST MBE PMI test models: [NIST-PMI-STEP-Files.zip](https://www.nist.gov/document/nist-pmi-step-files)
(14.0 MB, SHA-256 `8fa78429…1911`). NIST states they "can be used without any restrictions". The files are
**not** redistributed in this repository. The archive has **33 STEP files for 11 distinct parts**: 5 complex (CTC)
and 6 full/simplified test cases (FTC/STC). Each part comes in several exports: AP203 geometry only, AP203 with PMI,
AP242, and one AP242 tessellated file. These are real, PMI-heavy industrial files written by several CAD exporters.
Parts are up to 800 × 450 × 150 mm with up to 663 faces.

**Reproduce** (after downloading and unzipping the files into `data/r6_corpus/parts/`):

```bash
python -m cad2ml.cli --data data/r6 process-corpus --corpus data/r6_corpus --report docs/evidence/r6_nist_process_report.json
```

```bash
python scripts/r6_nist_eval.py --store data/r6 --report docs/evidence/r6_nist_eval.json
```

There are no hand-made face labels. Two measurements need no labels:
* **robustness:** outcomes, codes, warnings and timings;
* **cross-export consistency:** exports of the same part should give the same measured geometry and the same
  recognized features.

Feature correctness was also spot-checked visually on renders (see the audit below).

## What the first run found

The first run used the pipeline exactly as it was after the synthetic releases (v1.0.0).

| Outcome | Files | Cause |
|---|---|---|
| batch command found **0 files** | 33 | `process-corpus` globbed only `*.step`; NIST uses `.stp` (the upload API already accepted both) |
| completed | 6 / 33 | |
| `INTERNAL_ERROR` | 15 | edges belonging to no face (PMI annotation curves) crashed edge-convexity code |
| `INVALID_GEOMETRY` (quarantined) | 11 | auxiliary reference surfaces made the compound look like an "open shell" |
| `CHILD_CRASHED` | 1 | AP242 **tessellated** file: 273 faces with no surface geometry and 0 edges |

Synthetic parts never contain auxiliary geometry, so none of these showed up before R6.

## Fixes (pipeline 1.1.0)

1. `step_files()`: batch commands accept `.step`/`.stp` in any case (test added).
2. **Single-solid isolation.** When a file holds exactly one solid plus auxiliary geometry, only the solid is processed.
   What was ignored is **recorded as a warning**, e.g. `auxiliary geometry outside the solid ignored: 78 edges, 22
   vertices` (test added). 27 of 32 completed files had such geometry.
3. **Tessellated STEP** is rejected explicitly with the new code `UNSUPPORTED_TESSELLATED`.
4. **Point-on-surface check** now uses the chord deflection OCCT *achieved* on each face (stored in `mesh.npz`),
   not a fixed 0.1 mm. On two large freeform parts the mesher exceeded the target deflection; the samples were
   correct but the check was wrong.
5. **Forced reprocessing** no longer silently discards the new result when a failure record exists; the old sample
   is kept under `superseded/` (test added).
6. **Recognizer:** blind-hole floors may be *several* coaxial faces (exporters split drill-point cones), and
   **counterbored/stepped holes** are one feature: coaxial bores linked by annular steps (tests added).
7. **Near-duplicate detection** no longer depends on face counts, which differ between exporters. It uses tolerant
   pairwise matching on volume, area and sorted bounding box (test added).

## Results after the fixes

Source: `docs/evidence/r6_nist_eval.json`.

| Measure | Result |
|---|---|
| files processed | **32 completed**, 1 rejected (`UNSUPPORTED_TESSELLATED`, correct) |
| files with auxiliary geometry isolated and reported | 27 / 32 |
| files needing repair | 1 (`ShapeFix_Shape`; source flagged invalid by BRepCheck; repaired solid valid) |
| processing time per file | p50 8.3 s, max 27.9 s (native Windows, isolated children) |
| faces processed | 7,865 (cylinder 3,707 · plane 2,383 · cone 922 · torus 365 · sphere 312 · B-spline 176) |
| faces left `unknown` | 15.3 % (cones, spheres, tori and B-splines outside the v1 vocabulary) |
| hole walls not resolved to through/blind | 82 (down from ≈200 before fix 6) |
| **volume agreement across exports** | 10 / 11 parts within 0.1 %; `ftc_11` 0.11 % |
| **hole count identical across exports** | 9 / 11 parts |
| **hole diameters identical across exports** | 7 / 11 parts |
| **canonical face IDs identical across exports** | **0 / 11 parts** (fingerprint Jaccard 0.07–0.73) |
| near-duplicate detection (same part = duplicate) | 31/31 pairs found, 0 false matches (see caveat) |

### How to read these numbers

* **Canonical IDs are exporter-specific.** Exporters split the same surfaces differently: `ctc_02` has 487 / 635 / 663
  faces across its three exports. Canonical IDs are deterministic for a given B-Rep (tested), but they do **not**
  identify the same physical face across different exporters. The earlier docs said IDs are reproducible "across
  re-exports of identical geometry"; that holds only when the B-Rep topology is identical, and DECISIONS D-017
  corrects it.
* **The hole-diameter disagreements are real differences in the source files, not pipeline errors.** `ctc_03` holds a
  #10 drill as 4.775 mm (0.188") in AP203 and 4.763 mm (0.1875") in AP242. `ctc_04`'s native export has 6.647 / 10.106 mm
  where the others have 6.65 / 10.2 mm. STC_10 is a *simplified* model with extra hole sizes, so it isn't the same
  geometry as FTC_10.
* **Near-duplicate tolerances were set from the spreads measured on this same data** (volume ≤ 0.11 %, area ≤ 0.11 %).
  The 31/31 result shows the method works here; it isn't an independent validation.

## Visual audit (limited)

`docs/images/r6_nist_ctc_01_asme1_ap203_rule_labels.png` shows the rule labels (red hole, orange slot, green fillet)
on four views of `ctc_01`.
* Holes and slots are where the part has them.
* The large green regions looked like mislabelled rounded lug ends. Measuring them showed every "fillet" is a 90°
  tangent blend: concave inner fillets of r = 5 / 10 / 25 mm and 50 mm convex corner rounds on an 800 mm part. So the
  labels are geometrically right.
* At 256 px the thin plate `ftc_09` is too small to audit hole by hole.

**No hole-by-hole comparison against the NIST drawings was done.** That's the main open item for R6.

## What remains open

* A labelled real-world set: compare recognizer output hole by hole against the NIST drawings / semantic PMI (hole
  callouts in the AP242 files). Then the larger Fusion 360 Gallery segmentation set (real per-face operation labels;
  non-commercial licence).
* Features outside the v1 vocabulary: countersinks and chamfers (cones), spherical/toroidal blends, threads.
* Holes interrupted by other features (82 remaining walls, mostly with 3–17 rim positions along the axis).
* A face-matching method that survives exporter re-splitting (e.g. merge coplanar/co-surface faces before
  fingerprinting), measured against these NIST export pairs.
