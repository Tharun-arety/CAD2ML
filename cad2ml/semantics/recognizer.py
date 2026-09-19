"""Rule-based engineering-feature recognizer (``deterministic_rule_v1``).

Input: canonical face/edge records only (kernel-free, unit-testable).
Output: (a) ``GeometricObservation`` - measured facts, provenance *computed*;
        (b) ``FeatureRecord`` - semantic hypotheses, provenance *inferred*.

Confidence policy (documented, not learned):
    confidence = 0.95 * passed_checks / total_checks
Every check is recorded as evidence. 0.95 is a hard cap: a rule system with
geometric checks only cannot justify certainty. Features whose checks do not all pass
are downgraded (e.g. ``through_hole`` -> ``cylindrical_hole_wall``) or left ``unknown``.

Recognition order (earlier rules claim faces): holes -> slots -> pockets -> fillets
-> planar faces -> unknown.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from cad2ml.config import RECOGNIZER_VERSION, RecognizerConfig
from cad2ml.schemas.manifest import EdgeRecord, Evidence, FaceRecord, FeatureRecord, GeometricObservation

CONF_CAP = 0.95
FEATURE_TO_FACE_LABEL = {
    "through_hole": "hole",
    "blind_hole": "hole",
    "cylindrical_hole_wall": "hole",
    "slot": "slot",
    "pocket": "pocket",
    "fillet": "fillet",
    "planar_face": "planar",
    "unknown": "other",
}


def _conf(ev: Sequence[Evidence]) -> float:
    return round(CONF_CAP * sum(e.passed for e in ev) / max(len(ev), 1), 4)


def _u(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, float)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


@dataclass
class _Ctx:
    faces: list[FaceRecord]
    edges: dict[str, EdgeRecord]
    fmap: dict[str, FaceRecord]
    cfg: RecognizerConfig
    claimed: dict[str, str] = field(default_factory=dict)

    @property
    def cos_tol(self) -> float:
        return math.cos(math.radians(self.cfg.angle_tol_deg))

    def parallel(self, a: Any, b: Any) -> bool:
        return abs(float(_u(a) @ _u(b))) >= self.cos_tol

    def perpendicular(self, a: Any, b: Any) -> bool:
        return abs(float(_u(a) @ _u(b))) <= math.sin(math.radians(self.cfg.angle_tol_deg))

    def edges_between(self, a: str, b: str) -> list[EdgeRecord]:
        fa = self.fmap[a]
        return [self.edges[e] for e in fa.boundary_edge_ids if b in self.edges[e].adjacent_face_ids]


def angular_span(face: FaceRecord) -> float:
    """Swept angle (rad) of a cylinder (u) or torus minor circle (v)."""
    ub = face.uv_bounds
    if face.surface_type == "cylinder":
        return ub[1] - ub[0]
    if face.surface_type == "torus":
        return ub[3] - ub[2]
    return 0.0


def material_side(face: FaceRecord) -> Literal["inside", "outside", "n/a"]:
    H = face.curvature.get("mean_curvature_mean")
    if face.surface_type not in ("cylinder", "cone", "sphere", "torus") or H is None:
        return "n/a"
    return "outside" if H < 0 else "inside"  # concave surface: material outside the surface


def observe(faces: Sequence[FaceRecord]) -> list[GeometricObservation]:
    obs = []
    for f in faces:
        m: dict[str, float | list[float] | str | bool] = {"area_mm2": f.area_mm2}
        ap = f.analytic_params
        if f.surface_type == "cylinder":
            m.update(
                {
                    "radius_mm": ap["radius"],
                    "axis": ap["axis"],
                    "axis_origin_mm": ap["axis_origin"],  # type: ignore[dict-item]
                    "angular_span_deg": round(math.degrees(angular_span(f)), 4),
                    "concave": material_side(f) == "outside",
                }
            )
        elif f.surface_type == "torus":
            m.update(
                {
                    "major_radius_mm": ap["major_radius"],
                    "minor_radius_mm": ap["minor_radius"],  # type: ignore[dict-item]
                    "minor_span_deg": round(math.degrees(angular_span(f)), 4),
                    "concave": material_side(f) == "outside",
                }
            )
        elif f.surface_type == "plane":
            m.update({"normal": f.normal_at_centroid or ap.get("axis", [0.0, 0.0, 0.0])})  # type: ignore[dict-item]
        obs.append(
            GeometricObservation(
                face_id=f.face_id, surface_type=f.surface_type, material_side=material_side(f), measurements=m
            )
        )  # type: ignore[arg-type]
    return obs


# ------------------------------------------------------------------------------ holes


def _axis_key(f: FaceRecord) -> tuple[np.ndarray, np.ndarray, float]:
    ax = _u(f.analytic_params["axis"])  # type: ignore[arg-type]
    if ax[np.argmax(np.abs(ax))] < 0:
        ax = -ax
    o = np.asarray(f.analytic_params["axis_origin"], float)
    o = o - (o @ ax) * ax  # closest point of axis line to origin
    return ax, o, float(f.analytic_params["radius"])  # type: ignore[arg-type]


def _group_coaxial(ctx: _Ctx, cand: list[FaceRecord], same_radius: bool = True) -> list[list[FaceRecord]]:
    groups: list[list[FaceRecord]] = []
    keys: list[tuple[np.ndarray, np.ndarray, float]] = []
    tol = max(ctx.cfg.distance_tol_mm, 1e-4) * 10
    for f in cand:
        ax, o, r = _axis_key(f)
        for gi, (gax, go, gr) in enumerate(keys):
            if (
                ctx.parallel(ax, gax)
                and np.linalg.norm(o - go) < tol
                and (not same_radius or abs(r - gr) < tol)
            ):
                groups[gi].append(f)
                break
        else:
            groups.append([f])
            keys.append((ax, o, r))
    return groups


def _connected(ctx: _Ctx, faces: list[FaceRecord]) -> list[list[FaceRecord]]:
    ids = {f.face_id for f in faces}
    seen: set[str] = set()
    comps = []
    for f in faces:
        if f.face_id in seen:
            continue
        stack, comp = [f.face_id], []
        seen.add(f.face_id)
        while stack:
            cur = stack.pop()
            comp.append(ctx.fmap[cur])
            for n in ctx.fmap[cur].adjacent_face_ids:
                if n in ids and n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def _closing_floor(ctx: _Ctx, start: set[str], wall: set[str], ax: np.ndarray, o: np.ndarray) -> list[str]:
    """Faces that close one end of a hole: planes perpendicular to the axis and/or cones coaxial with it.

    Exporters often split a drill-point cone (or a flat floor) into several faces, so the floor is the
    connected set of such faces reachable from the rim whose neighbours are only the wall or each other.
    Returns [] if the end is not closed this way.
    """
    floor: set[str] = set()
    frontier = list(start)
    while frontier:
        fid = frontier.pop()
        if fid in floor or fid in wall:
            continue
        f = ctx.fmap[fid]
        if f.surface_type == "plane":
            ok = ctx.parallel(f.normal_at_centroid or [0, 0, 0], ax)
        elif f.surface_type == "cone":
            cax = _u(f.analytic_params["axis"])  # type: ignore[arg-type]
            apex = np.asarray(f.analytic_params.get("apex", f.analytic_params["axis_origin"]), float)
            ok = ctx.parallel(cax, ax) and bool(np.linalg.norm((apex - o) - ((apex - o) @ ax) * ax) < 1e-2)
        else:
            ok = False
        if not ok or len(floor) > 8:
            return []
        floor.add(fid)
        frontier.extend(n for n in f.adjacent_face_ids if n not in wall and n not in floor)
    closes = all(set(ctx.fmap[f].adjacent_face_ids) <= (wall | floor) for f in floor)
    return sorted(floor) if floor and closes else []


def _coaxial_transition_face(ctx: _Ctx, f: FaceRecord, ax: np.ndarray, o: np.ndarray) -> bool:
    """Plane perpendicular to the axis, or cone coaxial with it (floors, counterbore steps, countersinks)."""
    if f.surface_type == "plane":
        return ctx.parallel(f.normal_at_centroid or [0, 0, 0], ax)
    if f.surface_type == "cone":
        cax = _u(f.analytic_params["axis"])  # type: ignore[arg-type]
        apex = np.asarray(f.analytic_params.get("apex", f.analytic_params["axis_origin"]), float)
        return ctx.parallel(cax, ax) and bool(np.linalg.norm((apex - o) - ((apex - o) @ ax) * ax) < 1e-2)
    return False


def _step_to_segment(
    ctx: _Ctx, start: set[str], own: set[str], ax: np.ndarray, o: np.ndarray, seg_of: dict[str, int]
) -> tuple[list[str], int] | None:
    """A counterbore/stepped transition: coaxial annular faces linking this bore to exactly one other
    full coaxial bore. Returns (transition faces, other segment index) or None."""
    trans: set[str] = set()
    others: set[int] = set()
    frontier = list(start)
    while frontier:
        fid = frontier.pop()
        if fid in trans or fid in own:
            continue
        if fid in seg_of:
            others.add(seg_of[fid])
            continue
        f = ctx.fmap[fid]
        if not _coaxial_transition_face(ctx, f, ax, o) or len(trans) > 8:
            return None
        trans.add(fid)
        frontier.extend(n for n in f.adjacent_face_ids if n not in own and n not in trans)
    if len(others) != 1 or not trans:
        return None
    other_ids = {k for k, v in seg_of.items() if v in others}
    if not all(set(ctx.fmap[t].adjacent_face_ids) <= (own | trans | other_ids) for t in trans):
        return None
    return sorted(trans), next(iter(others))


def _recognize_holes(ctx: _Ctx) -> list[FeatureRecord]:
    """Holes as stacks of full, concave, coaxial cylindrical segments.

    Each segment end is classified as open (convex/smooth rim), floor (coaxial faces closing the end),
    step (coaxial annular faces leading to another bore: counterbore / stepped hole) or other. Segments
    linked by steps form one hole; the stack's terminal ends decide through vs. blind.
    """
    cand = [f for f in ctx.faces if f.surface_type == "cylinder" and material_side(f) == "outside"]
    segments: list[list[FaceRecord]] = []
    for g in _group_coaxial(ctx, cand, same_radius=True):
        segments.extend(_connected(ctx, g))
    full: list[list[FaceRecord]] = []
    for seg in segments:
        span = sum(angular_span(f) for f in seg)
        if abs(span - 2 * math.pi) < math.radians(ctx.cfg.angle_tol_deg):
            full.append(seg)
    seg_of = {f.face_id: i for i, seg in enumerate(full) for f in seg}
    info: list[dict[str, Any]] = []
    for seg in full:
        ids = {f.face_id for f in seg}
        ax, o, r = _axis_key(seg[0])
        rims: dict[int, list[tuple[EdgeRecord, str]]] = {}
        for f in seg:
            for eid in f.boundary_edge_ids:
                e = ctx.edges[eid]
                others = [a for a in e.adjacent_face_ids if a not in ids]
                if e.convexity == "seam" or not others:
                    continue
                pos = round(float(np.asarray(e.start_mm) @ ax) / 0.01)
                rims.setdefault(pos, []).append((e, others[0]))
        ends: list[dict[str, Any]] = []
        for pos in sorted(rims):
            rim = rims[pos]
            neighbours = {o_ for _, o_ in rim}
            conv = {e.convexity for e, _ in rim}
            end: dict[str, Any] = {"pos": pos * 0.01, "kind": "other", "faces": [], "nb": neighbours}
            if conv <= {"convex", "smooth"}:
                end["kind"] = "open"
            elif conv == {"concave"}:
                step = _step_to_segment(ctx, neighbours, ids, ax, o, seg_of)
                closing = [] if step else _closing_floor(ctx, neighbours, ids, ax, o)
                if step:
                    end.update(kind="step", faces=step[0], to=step[1])
                elif closing:
                    end.update(kind="floor", faces=closing)
            ends.append(end)
        info.append({"seg": seg, "ax": ax, "o": o, "r": r, "ends": ends})

    # union segments linked by steps into stacks
    parent = list(range(len(full)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, inf in enumerate(info):
        if len(inf["ends"]) == 2:
            for end in inf["ends"]:
                if end["kind"] == "step" and len(info[end["to"]]["ends"]) == 2:
                    parent[find(i)] = find(end["to"])
    stacks: dict[int, list[int]] = {}
    for i in range(len(full)):
        stacks.setdefault(find(i), []).append(i)

    feats: list[tuple[list[FaceRecord], dict[str, Any], list[Evidence], str]] = []
    for members in stacks.values():
        members.sort(key=lambda k: info[k]["r"])
        base = info[members[0]]
        ax, o, r = base["ax"], base["o"], base["r"]
        faces = [f for k in members for f in info[k]["seg"]]
        extra: set[str] = set()
        for k in members:
            for end in info[k]["ends"]:
                if end["kind"] == "step" and find(end["to"]) == find(k):
                    extra.update(end["faces"])
        # an end is internal if it is a linking step, or if its rim opens onto a linking step face
        # (the smaller bore's rim on a counterbore floor is convex, i.e. looks "open" from its side)
        terminal: list[dict[str, Any]] = []
        for k in members:
            for end in info[k]["ends"]:
                linking = end["kind"] == "step" and find(end["to"]) == find(k)
                if not linking and not (end["nb"] & extra):
                    terminal.append(end)
        ev = [
            Evidence(
                check="concave_cylindrical_surface",
                passed=True,
                detail=f"{len(faces)} face(s) in {len(members)} coaxial bore(s), mean curvature < 0",
            ),
            Evidence(check="full_360_degree_coverage", passed=True, detail="every bore covers 360 deg"),
            Evidence(
                check="two_axial_ends_found",
                passed=len(terminal) == 2,
                detail=f"{len(terminal)} terminal rim positions along axis",
            ),
        ]
        open_ends = sum(e["kind"] == "open" for e in terminal)
        floors = [fid for e in terminal if e["kind"] == "floor" for fid in e["faces"]]
        n_other = sum(e["kind"] == "other" for e in terminal)
        axial = [e["pos"] for e in terminal]
        length = (max(axial) - min(axial)) if len(axial) == 2 else float("nan")
        params: dict[str, Any] = {
            "diameter_mm": round(2 * r, 6),
            "axis": [round(float(x), 6) for x in ax],
            "axis_point_mm": [round(float(x), 6) for x in o],
        }
        if len(members) > 1:
            params["counterbore_diameters_mm"] = [round(2 * info[k]["r"], 6) for k in members[1:]]
            ev.append(
                Evidence(
                    check="coaxial_bores_linked_by_annular_steps",
                    passed=True,
                    detail=f"{len(members)} bores, {len(extra)} step face(s)",
                )
            )
        if len(terminal) == 2 and open_ends == 2:
            ev.append(Evidence(check="both_ends_open_with_convex_rims", passed=True, detail="through"))
            kind = "through_hole"
            params["length_mm"] = round(length, 6)
        elif len(terminal) == 2 and open_ends == 1 and floors:
            ev.append(
                Evidence(check="one_open_end_and_one_closing_floor", passed=True, detail=f"floor {floors[0]}")
            )
            kind = "blind_hole"
            params["depth_mm"] = round(length, 6)
        else:
            ev.append(
                Evidence(
                    check="end_conditions_consistent",
                    passed=False,
                    detail=f"open_ends={open_ends}, floors={floors}, other={n_other}",
                )
            )
            kind = "cylindrical_hole_wall"
        feats.append((faces + [ctx.fmap[x] for x in sorted(extra | set(floors))], params, ev, kind))
    # merge collinear segments of the same through hole (e.g. a pin bore through two clevis lugs)
    out: list[FeatureRecord] = []
    merged: list[int] = []
    for i, (faces_i, params_i, ev_i, kind_i) in enumerate(feats):
        if i in merged:
            continue
        faces_all, ev_all = list(faces_i), list(ev_i)
        if kind_i == "through_hole":
            for j in range(i + 1, len(feats)):
                faces_j, params_j, _, kind_j = feats[j]
                if (
                    kind_j == "through_hole"
                    and params_j["diameter_mm"] == params_i["diameter_mm"]
                    and ctx.parallel(params_i["axis"], params_j["axis"])
                    and np.linalg.norm(  # type: ignore[arg-type]
                        np.asarray(params_i["axis_point_mm"]) - np.asarray(params_j["axis_point_mm"])
                    )
                    < 1e-2
                ):
                    faces_all += faces_j
                    merged.append(j)
            if len(faces_all) > len(faces_i):
                ev_all.append(
                    Evidence(
                        check="collinear_segments_merged",
                        passed=True,
                        detail=f"{len(faces_all)} faces across interrupted bore",
                    )
                )
        out.append(
            FeatureRecord(
                feature_id="",
                feature_type=kind_i,  # type: ignore[arg-type]
                participating_faces=sorted({f.face_id for f in faces_all}),
                parameters=params_i,  # type: ignore[arg-type]
                inference_method=RECOGNIZER_VERSION,
                confidence=_conf(ev_all),
                confidence_basis="0.95 * passed_checks / total_checks",
                evidence=ev_all,
            )
        )
    return out


# ------------------------------------------------------------------------------ slots


def _recognize_slots(ctx: _Ctx) -> list[FeatureRecord]:
    out: list[FeatureRecord] = []
    free = [f for f in ctx.faces if f.face_id not in ctx.claimed]
    halves = [
        f
        for f in free
        if f.surface_type == "cylinder"
        and material_side(f) == "outside"
        and abs(angular_span(f) - math.pi) < math.radians(ctx.cfg.angle_tol_deg)
    ]
    used: set[str] = set()
    # (1) stadium slots: two concave half-cylinders + two parallel planar walls joined by smooth edges
    for i, a in enumerate(halves):
        for b in halves[i + 1 :]:
            if a.face_id in used or b.face_id in used:
                continue
            ra, rb = float(a.analytic_params["radius"]), float(b.analytic_params["radius"])  # type: ignore[arg-type]
            if abs(ra - rb) > 1e-3 or not ctx.parallel(a.analytic_params["axis"], b.analytic_params["axis"]):  # type: ignore[arg-type]
                continue
            walls = [
                ctx.fmap[w]
                for w in set(a.adjacent_face_ids) & set(b.adjacent_face_ids)
                if ctx.fmap[w].surface_type == "plane"
                and w not in ctx.claimed
                and all(e.convexity == "smooth" for e in ctx.edges_between(w, a.face_id))
            ]
            if len(walls) != 2:
                continue
            n1, n2 = (
                _u(walls[0].normal_at_centroid or [0, 0, 0]),
                _u(walls[1].normal_at_centroid or [0, 0, 0]),
            )
            gap = abs(float((np.asarray(walls[0].centroid_mm) - np.asarray(walls[1].centroid_mm)) @ n1))
            ev = [
                Evidence(
                    check="two_concave_half_cylinders_equal_radius", passed=True, detail=f"r={ra:.4f} mm"
                ),
                Evidence(
                    check="two_planar_walls_tangent_to_both_ends",
                    passed=True,
                    detail=f"{walls[0].face_id},{walls[1].face_id}",
                ),
                Evidence(
                    check="walls_parallel_and_facing",
                    passed=bool(float(n1 @ n2) < -ctx.cos_tol),
                    detail=f"n1.n2={float(n1 @ n2):.4f}",
                ),
                Evidence(
                    check="wall_gap_equals_diameter",
                    passed=abs(gap - 2 * ra) < 1e-2,
                    detail=f"gap {gap:.4f} vs 2r {2 * ra:.4f}",
                ),
            ]
            ids = {a.face_id, b.face_id, walls[0].face_id, walls[1].face_id}
            floor = [
                f
                for f in free
                if f.face_id not in ids
                and f.surface_type == "plane"
                and {a.face_id, b.face_id} <= set(f.adjacent_face_ids)
                and all(e.convexity == "concave" for e in ctx.edges_between(f.face_id, a.face_id))
            ]
            ca = np.asarray(a.analytic_params["axis_origin"], float)
            cb = np.asarray(b.analytic_params["axis_origin"], float)
            ax = _u(a.analytic_params["axis"])  # type: ignore[arg-type]
            d = (cb - ca) - ((cb - ca) @ ax) * ax
            params: dict[str, Any] = {
                "width_mm": round(2 * ra, 6),
                "length_mm": round(float(np.linalg.norm(d)) + 2 * ra, 6),
                "profile": "stadium",
                "through": "no" if floor else "yes",
                "depth_direction": [round(float(x), 6) for x in ax],
            }
            ids |= {f.face_id for f in floor}
            used |= ids
            out.append(
                FeatureRecord(
                    feature_id="",
                    feature_type="slot",
                    participating_faces=sorted(ids),
                    parameters=params,
                    inference_method=RECOGNIZER_VERSION,  # type: ignore[arg-type]
                    confidence=_conf(ev),
                    confidence_basis="0.95 * passed_checks / total_checks",
                    evidence=ev,
                )
            )
    # (2) open rectangular slots: planar floor + two parallel, facing planar walls joined by concave edges
    #     (directly or through a tangent concave blend face, which stays a separate fillet feature)
    for fl in free:
        if fl.surface_type != "plane" or fl.face_id in used or fl.face_id in ctx.claimed:
            continue
        nf = _u(fl.normal_at_centroid or [0, 0, 0])
        wall_map: dict[str, FaceRecord] = {}
        via: set[str] = set()
        for nb_id in fl.adjacent_face_ids:
            nb = ctx.fmap[nb_id]
            es = ctx.edges_between(fl.face_id, nb_id)
            if (
                nb.surface_type == "plane"
                and es
                and all(e.convexity == "concave" for e in es)
                and ctx.perpendicular(nb.normal_at_centroid or [0, 0, 0], nf)
            ):
                wall_map[nb_id] = nb
            elif (
                _is_blend(ctx, nb)
                and material_side(nb) == "outside"
                and all(e.convexity == "smooth" for e in es)
            ):
                for w_id in nb.adjacent_face_ids:
                    w = ctx.fmap[w_id]
                    tangent = ctx.edges_between(nb_id, w_id)
                    if (
                        w_id != fl.face_id
                        and w.surface_type == "plane"
                        and tangent
                        and all(e.convexity == "smooth" for e in tangent)
                        and ctx.perpendicular(w.normal_at_centroid or [0, 0, 0], nf)
                    ):
                        wall_map[w_id] = w
                        via.add(nb_id)
        if len(wall_map) != 2:
            continue
        w0, w1 = wall_map.values()
        n1, n2 = _u(w0.normal_at_centroid or [0, 0, 0]), _u(w1.normal_at_centroid or [0, 0, 0])
        c0, c1 = np.asarray(w0.centroid_mm), np.asarray(w1.centroid_mm)
        facing = float(n1 @ n2) < -ctx.cos_tol and float((c1 - c0) @ n1) > 0
        if not facing:
            continue
        gap = abs(float((c0 - c1) @ n1))
        connected = set(wall_map) | via
        rest = [
            ctx.edges[x]
            for x in fl.outer_loop_edge_ids
            if not set(ctx.edges[x].adjacent_face_ids) & connected
        ]
        open_ends = bool(rest) and all(e.convexity == "convex" for e in rest)
        ev = [
            Evidence(
                check="planar_floor_with_two_perpendicular_walls",
                passed=True,
                detail=f"floor {fl.face_id}, walls {w0.face_id},{w1.face_id}"
                + (f" via blends {sorted(via)}" if via else ""),
            ),
            Evidence(check="walls_parallel_and_facing_each_other", passed=True, detail=f"gap {gap:.4f} mm"),
            Evidence(
                check="floor_open_at_both_ends",
                passed=open_ends,
                detail=f"{len(rest)} remaining outer floor edges, convex={open_ends}",
            ),
        ]
        if not open_ends:
            continue  # closed on more sides: pocket rule decides
        ids = {fl.face_id, w0.face_id, w1.face_id}
        used |= ids
        out.append(
            FeatureRecord(
                feature_id="",
                feature_type="slot",
                participating_faces=sorted(ids),
                parameters={
                    "width_mm": round(gap, 6),
                    "profile": "rectangular_open",
                    "through": "open",
                    "floor_normal": [round(float(x), 6) for x in nf],
                    "blend_faces": ",".join(sorted(via)),
                },
                inference_method=RECOGNIZER_VERSION,
                confidence=_conf(ev),
                confidence_basis="0.95 * passed_checks / total_checks",
                evidence=ev,
            )
        )
    return out


# ------------------------------------------------------------------------------ pockets / fillets


def _is_blend(ctx: _Ctx, f: FaceRecord) -> bool:
    """Partial cylinder/torus whose sweep boundaries are tangent (smooth) to neighbours."""
    if f.surface_type not in ("cylinder", "torus"):
        return False
    span = angular_span(f)
    if not (0 < span <= 0.6 * math.pi):
        return False
    smooth = [ctx.edges[e] for e in f.boundary_edge_ids if ctx.edges[e].convexity == "smooth"]
    return len(smooth) >= 2


def _recognize_pockets(ctx: _Ctx) -> list[FeatureRecord]:
    out: list[FeatureRecord] = []
    for fl in ctx.faces:
        if fl.surface_type != "plane" or fl.face_id in ctx.claimed:
            continue
        nf = _u(fl.normal_at_centroid or [0, 0, 0])
        outer = [ctx.edges[x] for x in fl.outer_loop_edge_ids]
        if len(outer) < 3 or not all(e.convexity == "concave" for e in outer):
            continue
        walls = sorted({a for e in outer for a in e.adjacent_face_ids if a != fl.face_id})
        if any(w in ctx.claimed for w in walls):
            continue
        # walls may continue through tangent corner faces
        ring = set(walls)
        frontier = list(walls)
        while frontier:
            w = frontier.pop()
            for n in ctx.fmap[w].adjacent_face_ids:
                if n in ring or n == fl.face_id or n in ctx.claimed:
                    continue
                if all(e.convexity == "smooth" for e in ctx.edges_between(w, n)) and ctx.fmap[
                    n
                ].surface_type in ("plane", "cylinder"):
                    ring.add(n)
                    frontier.append(n)
        wall_ok = all(
            (
                ctx.fmap[w].surface_type == "plane"
                and ctx.perpendicular(ctx.fmap[w].normal_at_centroid or [0, 0, 0], nf)
            )
            or (
                ctx.fmap[w].surface_type == "cylinder"
                and ctx.parallel(ctx.fmap[w].analytic_params["axis"], nf)  # type: ignore[arg-type]
                and material_side(ctx.fmap[w]) == "outside"
            )
            for w in ring
        )
        rim_convex = all(
            any(
                e.convexity == "convex"
                for e in (ctx.edges[x] for x in ctx.fmap[w].boundary_edge_ids)
                if fl.face_id not in e.adjacent_face_ids
            )
            for w in ring
        )
        if not wall_ok:
            continue
        depth = max(
            float(np.ptp([ctx.fmap[w].bbox_max_mm, ctx.fmap[w].bbox_min_mm], axis=0) @ np.abs(nf))
            for w in ring
        )
        ev = [
            Evidence(check="planar_floor_outer_loop_all_concave", passed=True, detail=f"{len(outer)} edges"),
            Evidence(
                check="walls_perpendicular_or_axis_parallel_to_floor_normal",
                passed=True,
                detail=f"{len(ring)} wall faces",
            ),
            Evidence(
                check="walls_open_with_convex_rim",
                passed=rim_convex,
                detail="opening detected" if rim_convex else "no convex rim edge on some wall",
            ),
        ]
        corner_r = sorted(
            {
                float(ctx.fmap[w].analytic_params["radius"])  # type: ignore[arg-type]
                for w in ring
                if ctx.fmap[w].surface_type == "cylinder"
            }
        )
        ids = sorted(ring | {fl.face_id})
        out.append(
            FeatureRecord(
                feature_id="",
                feature_type="pocket",
                participating_faces=ids,
                parameters={
                    "depth_mm": round(depth, 6),
                    "floor_face": fl.face_id,
                    "floor_normal": [round(float(x), 6) for x in nf],
                    "corner_radii_mm": corner_r,
                    "floor_area_mm2": fl.area_mm2,
                },
                inference_method=RECOGNIZER_VERSION,
                confidence=_conf(ev),
                confidence_basis="0.95 * passed_checks / total_checks",
                evidence=ev,
            )
        )
    return out


def _recognize_fillets(ctx: _Ctx, max_radius_fraction: float = 0.1) -> list[FeatureRecord]:
    out: list[FeatureRecord] = []
    lo = np.min([f.bbox_min_mm for f in ctx.faces], axis=0)
    hi = np.max([f.bbox_max_mm for f in ctx.faces], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    # Exporters split a 180 deg rounded end (lug/boss) into several <= 90 deg faces; judge the sweep of the
    # whole connected coaxial same-radius group, not the single face (NIST R6 audit finding).
    partial = [
        c
        for c in ctx.faces
        if c.surface_type == "cylinder"
        and 0 < angular_span(c) < 2 * math.pi - 1e-6
        and c.face_id not in ctx.claimed
    ]
    group_span: dict[str, float] = {}
    for g in _group_coaxial(ctx, partial, same_radius=True):
        for comp in _connected(ctx, g):
            sides = {material_side(c) for c in comp}
            span = sum(angular_span(c) for c in comp) if len(sides) == 1 else 0.0
            for c in comp:
                group_span[c.face_id] = span
    for f in ctx.faces:
        if f.face_id in ctx.claimed or not _is_blend(ctx, f):
            continue
        if group_span.get(f.face_id, 0.0) > 0.6 * math.pi:
            continue  # rounded end / boss, not a blend: left for later rules (becomes `unknown`)
        r = float(f.analytic_params.get("radius", f.analytic_params.get("minor_radius", 0.0)))  # type: ignore[arg-type]
        smooth_nb = sorted(
            {
                a
                for e in f.boundary_edge_ids
                for a in ctx.edges[e].adjacent_face_ids
                if ctx.edges[e].convexity == "smooth" and a != f.face_id
            }
        )
        ev = [
            Evidence(
                check="partial_cylinder_or_torus",
                passed=True,
                detail=f"{f.surface_type}, span {math.degrees(angular_span(f)):.2f} deg",
            ),
            Evidence(
                check="tangent_to_two_neighbours", passed=len(smooth_nb) >= 2, detail=",".join(smooth_nb)
            ),
            Evidence(
                check="radius_small_relative_to_part",
                passed=r < max_radius_fraction * diag,
                detail=f"r={r:.4f} mm, part diagonal {diag:.2f} mm",
            ),
        ]
        out.append(
            FeatureRecord(
                feature_id="",
                feature_type="fillet",
                participating_faces=[f.face_id],
                parameters={
                    "radius_mm": round(r, 6),
                    "blend": "concave" if material_side(f) == "outside" else "convex",
                },
                inference_method=RECOGNIZER_VERSION,
                confidence=_conf(ev),
                confidence_basis="0.95 * passed_checks / total_checks",
                evidence=ev,
            )
        )
    return out


def recognize(
    faces: Sequence[FaceRecord], edges: Sequence[EdgeRecord], cfg: RecognizerConfig | None = None
) -> tuple[list[GeometricObservation], list[FeatureRecord]]:
    ctx = _Ctx(
        list(faces), {e.edge_id: e for e in edges}, {f.face_id: f for f in faces}, cfg or RecognizerConfig()
    )
    feats: list[FeatureRecord] = []
    for rule in (_recognize_holes, _recognize_slots, _recognize_pockets, _recognize_fillets):
        for feat in rule(ctx):
            if any(fid in ctx.claimed for fid in feat.participating_faces):
                continue
            for fid in feat.participating_faces:
                ctx.claimed[fid] = feat.feature_type
            feats.append(feat)
    for f in faces:
        if f.face_id in ctx.claimed:
            continue
        if f.surface_type == "plane":
            ev = [
                Evidence(check="analytic_plane_surface", passed=True, detail="surface type read from B-Rep"),
                Evidence(check="not_part_of_recognized_feature", passed=True, detail="no rule claimed face"),
            ]
            feats.append(
                FeatureRecord(
                    feature_id="",
                    feature_type="planar_face",
                    participating_faces=[f.face_id],
                    parameters={"normal": f.normal_at_centroid or [0.0, 0.0, 0.0], "area_mm2": f.area_mm2},
                    inference_method=RECOGNIZER_VERSION,
                    confidence=_conf(ev),
                    confidence_basis="0.95 * passed_checks / total_checks",
                    evidence=ev,
                )
            )
        else:
            ev = [
                Evidence(check="any_rule_matched", passed=False, detail=f"{f.surface_type} face unexplained")
            ]
            feats.append(
                FeatureRecord(
                    feature_id="",
                    feature_type="unknown",
                    participating_faces=[f.face_id],
                    parameters={"surface_type": f.surface_type},
                    inference_method=RECOGNIZER_VERSION,
                    confidence=0.0,
                    confidence_basis="no rule matched",
                    evidence=ev,
                )
            )
    prefix = {
        "through_hole": "H",
        "blind_hole": "H",
        "cylindrical_hole_wall": "W",
        "slot": "S",
        "pocket": "P",
        "fillet": "R",
        "planar_face": "PL",
        "unknown": "U",
    }
    feats.sort(key=lambda x: (list(prefix).index(x.feature_type), x.participating_faces))
    counters: dict[str, int] = {}
    for feat in feats:
        p = prefix[feat.feature_type]
        counters[p] = counters.get(p, 0) + 1
        feat.feature_id = f"{p}{counters[p]:03d}"
    return observe(faces), feats


def face_labels_from_features(faces: Sequence[FaceRecord], feats: Sequence[FeatureRecord]) -> dict[str, str]:
    lab = {f.face_id: ("planar" if f.surface_type == "plane" else "other") for f in faces}
    for feat in feats:
        for fid in feat.participating_faces:
            lab[fid] = FEATURE_TO_FACE_LABEL[feat.feature_type]
    return lab
