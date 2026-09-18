"""Entity-level lineage queries across representations.

``trace_face`` resolves a canonical face through every stored artifact and *verifies*
each link against the actual arrays (not only the lineage index), e.g. that the point
range really carries that face index and that mask pixels exist in the stated views.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np

from cad2ml.pipeline import load_manifest
from cad2ml.storage.base import ArtifactStore


class LineageError(LookupError):
    pass


def _npz(store: ArtifactStore, key: str) -> dict[str, Any]:
    with np.load(io.BytesIO(store.get_bytes(key)), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def trace_face(store: ArtifactStore, sample_id: str, face_id: str) -> dict[str, Any]:
    m = load_manifest(store, sample_id)
    if m is None:
        raise LineageError(f"unknown sample {sample_id}")
    if m.status != "completed":
        raise LineageError(f"sample {sample_id} is {m.status}: {m.rejection.code if m.rejection else ''}")
    base = f"samples/{sample_id}"
    lineage = json.loads(store.get_bytes(f"{base}/lineage.json"))
    if face_id not in lineage["faces"]:
        raise LineageError(f"face {face_id} not in sample {sample_id}")
    entry = lineage["faces"][face_id]
    brep = json.loads(store.get_bytes(f"{base}/brep.json"))
    face = next(f for f in brep["faces"] if f["face_id"] == face_id)
    graph = _npz(store, f"{base}/graph.npz")
    pc = _npz(store, f"{base}/pointcloud.npz")
    mesh = _npz(store, f"{base}/mesh.npz")
    views = _npz(store, f"{base}/views.npz")
    feats = json.loads(store.get_bytes(f"{base}/features.json"))
    node = entry["graph_node"]
    ps, pe = entry["point_range"]
    ts, te = entry["triangle_range"]
    pix = {int(k): int(np.count_nonzero(views["face_id"][int(k)] == node + 1)) for k in entry["view_pixels"]}
    neighbours = graph["edge_index"][1][graph["edge_index"][0] == node].tolist()
    checks = {
        "graph_node_maps_to_face": str(graph["node_face_ids"][node]) == face_id,
        "points_carry_face_index": bool(np.all(pc["face_ids"][ps:pe] == node)) and pe > ps,
        "no_points_outside_range": int(np.count_nonzero(pc["face_ids"] == node)) == pe - ps,
        "triangles_carry_face_index": bool(np.all(mesh["tri_face_index"][ts:te] == node)) and te > ts,
        "view_pixels_match_index": pix == {int(k): v for k, v in entry["view_pixels"].items()},
        "graph_neighbours_match_brep_adjacency": sorted(str(graph["node_face_ids"][n]) for n in neighbours)
        == sorted(face["adjacent_face_ids"]),
    }
    features = [f for f in feats["features"] if face_id in f["participating_faces"]]
    observation = next((o for o in feats["observations"] if o["face_id"] == face_id), None)
    return {
        "chain": {
            "source": {
                "filename": m.source.filename,
                "sha256": m.source.sha256,
                "original_units": m.source.original_units,
                "source_key": m.lineage.get("source_key"),
            },
            "processing": {
                "pipeline_version": m.processing.pipeline_version,
                "extractor_version": m.processing.extractor_version,
                "parser_version": m.processing.parser_version,
                "configuration_hash": m.processing.configuration_hash,
            },
            "brep_face": {
                k: face[k]
                for k in (
                    "face_id",
                    "surface_type",
                    "area_mm2",
                    "centroid_mm",
                    "analytic_params",
                    "adjacent_face_ids",
                    "fingerprint",
                )
            },
            "graph": {
                "node_index": node,
                "neighbour_nodes": sorted(neighbours),
                "feature_vector_head": [round(float(x), 5) for x in graph["face_features"][node][:8]],
            },
            "point_cloud": {
                "point_range": [ps, pe],
                "n_points": pe - ps,
                "range_text": f"points {ps}-{pe - 1} -> {face_id}",
            },
            "mesh": {"triangle_range": [ts, te], "n_triangles": te - ts},
            "views": {"pixels_per_view": pix},
            "observation (computed)": observation,
            "features (inferred)": [
                {
                    k: f[k]
                    for k in ("feature_id", "feature_type", "parameters", "confidence", "inference_method")
                }
                for f in features
            ],
        },
        "verified": checks,
        "all_links_verified": all(checks.values()),
    }
