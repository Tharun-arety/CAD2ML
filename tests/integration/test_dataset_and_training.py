"""Dataset versioning (R3 gate), recognizer evaluation, training smoke test and prediction traceability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad2ml.datasets.builder import build_dataset, check_leakage


@pytest.fixture(scope="module")
def dataset(processed_store, mini_corpus: Path) -> dict:
    return build_dataset(processed_store, mini_corpus, seed=3)


def test_dataset_rebuild_is_identical_and_leak_free(
    processed_store, mini_corpus: Path, dataset: dict
) -> None:
    again = build_dataset(processed_store, mini_corpus, seed=3)
    assert again["dataset_id"] == dataset["dataset_id"]
    assert again["created_at"] == dataset["created_at"]  # immutable: existing version returned, not rewritten
    assert check_leakage(dataset["samples"], "split_unit") == {}
    assert dataset["quality"]["leakage"]["passed"]
    all_ids = [s for ids in dataset["splits"].values() for s in ids]
    assert len(all_ids) == len(set(all_ids)) == len(dataset["samples"]) == 18
    other_seed = build_dataset(processed_store, mini_corpus, seed=4)
    assert other_seed["dataset_id"] != dataset["dataset_id"]
    fam = build_dataset(processed_store, mini_corpus, seed=3, split_mode="family")
    assert check_leakage(fam["samples"], "family") == {}
    assert fam["quality"]["representation_completeness"]["graph"] == 1.0


def test_labels_align_with_graph_nodes(processed_store, dataset: dict) -> None:
    labels = json.loads(processed_store.get_bytes(f"datasets/{dataset['dataset_id']}/labels.json"))
    import io

    import numpy as np

    for sid in dataset["samples"]:
        with np.load(io.BytesIO(processed_store.get_bytes(f"samples/{sid}/graph.npz"))) as g:
            assert [str(x) for x in g["node_face_ids"]] == labels[sid]["face_ids"]
    q = dataset["quality"]
    assert {"planar", "hole"} <= set(q["class_distribution"]["train"])
    assert q["rule_recognizer_face_label_agreement"] > 0.9


def test_recognizer_evaluation_against_ground_truth(processed_store, mini_corpus: Path) -> None:
    from cad2ml.evaluation.recognizer_eval import evaluate

    r = evaluate(processed_store, mini_corpus)
    assert r["samples"] == 18
    assert r["global_dimensions"]["max_abs_bbox_error_mm"] < 1e-3
    assert r["feature_level"]["through_hole"]["recall"] == 1.0
    assert r["feature_level"]["through_hole"]["precision"] == 1.0
    assert r["hole_diameter_abs_error_mm"]["max"] < 1e-6


def test_training_smoke_and_prediction_traceability(processed_store, dataset: dict) -> None:
    torch = pytest.importorskip("torch")
    from cad2ml.training.train import TrainConfig, infer_sample, run_training

    exp = run_training(processed_store, dataset["dataset_id"], TrainConfig(model="gnn", epochs=3, seed=0))
    assert exp["metrics"]["test"]["summary"]["n_faces"] > 0
    preds = json.loads(processed_store.get_bytes(f"training_runs/{exp['run_id']}/predictions.json"))
    test_ids = set(dataset["splits"]["test"])
    assert {p["sample_id"] for p in preds["test"]} == test_ids
    brep = json.loads(processed_store.get_bytes(f"samples/{preds['test'][0]['sample_id']}/brep.json"))
    assert preds["test"][0]["face_id"] in {f["face_id"] for f in brep["faces"]}
    rows = infer_sample(processed_store, exp["run_id"], preds["test"][0]["sample_id"])
    assert [r["face_id"] for r in rows] == [f["face_id"] for f in brep["faces"]]
    assert all(0.0 <= r["confidence"] <= 1.0 for r in rows)
    # determinism: same config -> identical metrics
    exp2 = run_training(processed_store, dataset["dataset_id"], TrainConfig(model="gnn", epochs=3, seed=0))
    assert exp2["metrics"] == exp["metrics"]
    assert torch.get_num_threads() >= 1
