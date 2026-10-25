import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.evaluate_reranker_test import (
    CLIP_SELECTOR,
    ZERO_SHOT,
    RerankerFeatureTables,
    build_selection_tasks,
    checkpoint_pool_size,
    checkpoint_provenance_mismatches,
    clip_candidate_pool,
    mcnemar_exact_p,
    paired_accuracy_difference,
    rerank_pools,
    reranker_inputs,
    resolve_primary_selector,
    summarize_test_results,
    write_query_csv,
)
from scripts.export_reranker_predictions import POOLED_INPUTS
from scripts.generate_reranker_teacher_data import build_teacher_tasks
from src.data.dataclasses import RerankerTestEvalResult
from src.data.reranker_dataset import RerankerTeacherDataset, collate_reranker_queries
from src.models.reranker import LabelAwareReranker, RerankerConfig


def _unit(rows):
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def _feature_space(seed=0, num_query=4, num_retrieval=12, num_classes=5):
    rng = np.random.default_rng(seed)
    return {
        "query_clip": _unit(rng.normal(size=(num_query, 6))),
        "retrieval_clip": _unit(rng.normal(size=(num_retrieval, 6))),
        "query_siglip": _unit(rng.normal(size=(num_query, 7))),
        "retrieval_siglip": _unit(rng.normal(size=(num_retrieval, 7))),
        "label_siglip": _unit(rng.normal(size=(num_classes, 7))),
        "retrieval_classes": np.arange(num_retrieval) % num_classes,
        "query_classes": np.arange(num_query) % num_classes,
    }


def _tables(space):
    return RerankerFeatureTables(
        query_clip=torch.as_tensor(space["query_clip"]),
        query_siglip=torch.as_tensor(space["query_siglip"]),
        retrieval_clip=torch.as_tensor(space["retrieval_clip"]),
        retrieval_siglip=torch.as_tensor(space["retrieval_siglip"]),
        label_siglip=torch.as_tensor(space["label_siglip"]),
        retrieval_classes=torch.as_tensor(space["retrieval_classes"], dtype=torch.long),
    )


def test_clip_pool_matches_teacher_generation_for_a_disjoint_query_split():
    space = _feature_space()
    examples = lambda classes: [SimpleNamespace(label=int(c)) for c in classes]
    query_split = SimpleNamespace(
        examples=examples(space["query_classes"]), clip_embeddings=space["query_clip"]
    )
    retrieval = SimpleNamespace(
        examples=examples(space["retrieval_classes"]), clip_embeddings=space["retrieval_clip"]
    )
    cfg = OmegaConf.create({
        "experiment": {"seed": 0},
        "targets": {"k": 3},
        "retrieval": {"candidate_pool_size": 5, "exclude_same_class": False},
        "dataset": {"query_splits": ["val"], "retrieval_split": "train"},
        "limits": {"max_queries_per_split": {"val": None}},
    })
    teacher_tasks, _ = build_teacher_tasks(
        cfg, {"val": query_split}, retrieval,
        {"val": space["query_siglip"]}, space["label_siglip"],
    )

    for task in teacher_tasks:
        order, similarities = clip_candidate_pool(
            space["retrieval_clip"], space["query_clip"][task["query_idx"]], 5
        )
        np.testing.assert_array_equal(order, task["candidate_indices"])
        np.testing.assert_array_equal(similarities, task["candidate_similarities"])


def test_reranker_inputs_match_training_dataset_and_collation():
    space = _feature_space()
    pool_size = 4
    pools = [
        clip_candidate_pool(space["retrieval_clip"], space["query_clip"][q], pool_size)
        for q in range(2)
    ]
    metrics = {
        name: np.zeros(pool_size, dtype=np.float32)
        for name in ("margin", "true_probability", "correct")
    }
    artifact = {
        "method": "reranker_teacher_data",
        "immutable_args": {"schema_version": 2, "retrieval_split": "train"},
        "records": [
            SimpleNamespace(
                query_split="test",
                query_idx=q,
                true_class_idx=int(space["query_classes"][q]),
                candidate_indices=order.astype(np.int32),
                candidate_class_indices=space["retrieval_classes"][order].astype(np.int16),
                candidate_similarities=similarities,
                candidate_metrics=metrics,
            )
            for q, (order, similarities) in enumerate(pools)
        ],
        "feature_tables": {
            "clip_image_embeddings_by_split": {
                "train": space["retrieval_clip"], "test": space["query_clip"],
            },
            "siglip_image_embeddings_by_split": {
                "train": space["retrieval_siglip"], "test": space["query_siglip"],
            },
            "siglip_class_text_embeddings": space["label_siglip"],
            "class_names": list("abcde"),
        },
    }
    dataset = RerankerTeacherDataset(artifact, split="test", target="margin")
    expected = collate_reranker_queries([dataset[0], dataset[1]])

    actual = reranker_inputs(
        _tables(space),
        torch.as_tensor([0, 1]),
        torch.as_tensor(np.stack([order for order, _ in pools])),
        torch.as_tensor(np.stack([similarities for _, similarities in pools])),
    )

    assert set(actual) == set(POOLED_INPUTS)
    for name in POOLED_INPUTS:
        assert actual[name].dtype == expected[name].dtype, name
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_rerank_pools_returns_the_argmax_position_for_every_query():
    space = _feature_space()
    torch.manual_seed(0)
    model = LabelAwareReranker(RerankerConfig(clip_dim=6, siglip_dim=7)).eval()
    tables = _tables(space)
    pools = [clip_candidate_pool(space["retrieval_clip"], space["query_clip"][q], 5) for q in range(4)]
    pool_indices = np.stack([order for order, _ in pools])
    pool_similarities = np.stack([similarities for _, similarities in pools])

    # A batch size that leaves a short final batch must not change selections.
    positions = rerank_pools(
        model, tables, np.arange(4), pool_indices, pool_similarities, 3, torch.device("cpu")
    )

    with torch.inference_mode():
        scores = model(**reranker_inputs(
            tables, torch.arange(4), torch.as_tensor(pool_indices),
            torch.as_tensor(pool_similarities),
        ))
    np.testing.assert_array_equal(positions, scores.argmax(dim=1).numpy())


def test_selection_tasks_deduplicate_exemplars_and_respect_each_pool_size():
    pool_indices = np.asarray([[7, 3, 9, 4], [5, 6, 8, 2]])
    retrieval_classes = np.zeros(10, dtype=int)
    retrieval_classes[[4, 6]] = 1
    base = [
        {"query_idx": 11, "distractor_class_indices": {2: [0, 1]}},
        {"query_idx": 12, "distractor_class_indices": {2: [0, 1]}},
    ]

    tasks = build_selection_tasks(
        base,
        true_classes=np.asarray([1, 1]),
        pool_indices=pool_indices,
        retrieval_classes=retrieval_classes,
        selected_positions={"m2": np.asarray([0, 1]), "m4": np.asarray([3, 1])},
        pool_sizes={"m2": 2, "m4": 4},
    )

    assert tasks[0]["selected_example_indices"] == {CLIP_SELECTOR: 7, "m2": 7, "m4": 4}
    assert tasks[0]["example_indices"] == [4, 7]
    # The true-class exemplar at position 3 is outside the M=2 pool.
    assert tasks[0]["pool_contains_true_class"] == {"m2": False, "m4": True}
    assert tasks[1]["example_indices"] == [5, 6]
    assert tasks[1]["selected_positions"] == {CLIP_SELECTOR: 0, "m2": 1, "m4": 1}
    assert tasks[1]["distractor_class_indices"] == {2: [0, 1]}

    with pytest.raises(ValueError, match="outside its pool"):
        build_selection_tasks(
            base[:1], np.asarray([1]), pool_indices[:1], retrieval_classes,
            {"m2": np.asarray([2])}, {"m2": 2},
        )


def test_mcnemar_exact_p_matches_binomial_tail():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact_p(3, 3) == 1.0
    # Large discordant counts must not underflow or overflow.
    assert 0 <= mcnemar_exact_p(900, 700) < 1e-5


def test_paired_bootstrap_is_reproducible_and_brackets_the_difference():
    rng = np.random.default_rng(1)
    control = rng.random(500) < 0.5
    treatment = control | (rng.random(500) < 0.3)

    first = paired_accuracy_difference(treatment, control, 2000, 0.95, seed=7)
    second = paired_accuracy_difference(treatment, control, 2000, 0.95, seed=7)

    assert first == second
    assert first["accuracy_difference"] == pytest.approx(treatment.mean() - control.mean())
    assert first["ci_low"] < first["accuracy_difference"] < first["ci_high"]
    assert first["ci_low"] > 0
    assert first["control_only_correct"] == 0
    assert sum(first[key] for key in (
        "treatment_only_correct", "control_only_correct", "both_correct", "neither_correct"
    )) == 500


def test_provenance_check_reports_each_mismatched_input():
    checkpoint = {"provenance": {"artifact_immutable_args": {
        "clip_model": "ViT-B/32",
        "retrieval_split": "train",
        "siglip_image_embeddings_sha256_by_split": {"train": "aaa", "val": "bbb"},
        "load_in_8bit": False,
    }}}
    expected = {
        "clip_model": "ViT-B/32",
        "retrieval_split": "train",
        "siglip_retrieval_image_embeddings_sha256": "aaa",
        "load_in_8bit": True,
    }

    mismatches = checkpoint_provenance_mismatches(checkpoint, expected)

    assert mismatches == ["load_in_8bit: checkpoint=False, evaluation=True"]
    assert checkpoint_provenance_mismatches({}, expected) == [
        "checkpoint records no teacher-artifact provenance"
    ]


def test_pool_size_prefers_training_prefix_over_artifact_width():
    provenance = {"artifact_immutable_args": {"candidate_pool_size": 100}}
    assert checkpoint_pool_size({
        "resolved_config": {"data": {"max_candidates": 50}}, "provenance": provenance,
    }) == 50
    assert checkpoint_pool_size({
        "resolved_config": {"data": {"max_candidates": None}}, "provenance": provenance,
    }) == 100


def test_primary_selector_must_be_explicit_for_several_selectors():
    assert resolve_primary_selector(None, ["m30"]) == "m30"
    assert resolve_primary_selector("m50", ["m30", "m50"]) == "m50"
    with pytest.raises(ValueError, match="chosen on validation"):
        resolve_primary_selector(None, ["m30", "m50"])
    with pytest.raises(ValueError, match="not an enabled selector"):
        resolve_primary_selector(CLIP_SELECTOR, ["m30"])


def _records():
    # Three classes; K=2 sets hold the true class plus class 2.
    def record(query_idx, true_class, clip_example, reranker_example, scores, pool_has_true):
        return RerankerTestEvalResult(
            query_idx=query_idx,
            true_class_idx=true_class,
            zero_shot_scores=[0.0, 0.0, 1.0],
            selected_example_indices={CLIP_SELECTOR: clip_example, "rr": reranker_example},
            selected_positions={CLIP_SELECTOR: 0, "rr": 0 if clip_example == reranker_example else 2},
            pool_contains_true_class={"rr": pool_has_true},
            example_class_indices={idx: cls for idx, (cls, _) in scores.items()},
            example_scores={idx: values for idx, (_, values) in scores.items()},
            distractor_class_indices={2: sorted({true_class, 2})},
        )

    return [
        # Reranker fixes CLIP's error with a same-class exemplar.
        record(0, 0, 10, 11, {10: (2, [0.0, -1.0, 0.5]), 11: (0, [0.9, 0.1, 0.0])}, True),
        # Both selectors agree and share one score vector.
        record(1, 1, 12, 12, {12: (1, [0.0, 0.8, 0.1])}, True),
        # Pool lacks the true class; both are wrong.
        record(2, 0, 13, 14, {13: (2, [0.1, 0.0, 0.9]), 14: (1, [0.2, 0.6, 0.3])}, False),
    ]


def test_summary_reports_paired_selection_and_restricted_metrics(tmp_path):
    records = _records()
    specs = [{"name": "rr", "pool_size": 30}]

    results = summarize_test_results(
        records, specs, "rr", temperature=1.0, k_values=[2],
        bootstrap_samples=500, confidence_level=0.95, seed=0,
    )

    assert results["primary_selector"] == "rr"
    assert results["accuracy"] == pytest.approx(2 / 3)
    assert results["conditions"][CLIP_SELECTOR]["accuracy"] == pytest.approx(1 / 3)
    assert results["conditions"][ZERO_SHOT]["accuracy"] == 0.0
    paired = results["paired_vs_clip_top1"]["rr"]
    assert paired["accuracy_difference"] == pytest.approx(1 / 3)
    assert (paired["treatment_only_correct"], paired["control_only_correct"]) == (1, 0)
    selection = results["selection"]["rr"]
    assert selection["same_class_rate"] == pytest.approx(2 / 3)
    assert selection["agreement_with_clip_top1"] == pytest.approx(1 / 3)
    assert selection["pool_contains_true_class_rate"] == pytest.approx(2 / 3)
    assert selection["accuracy_when_pool_contains_true"] == 1.0
    assert selection["clip_top1_accuracy_when_pool_contains_true"] == 0.5
    assert selection["accuracy_when_pool_lacks_true"] == 0.0
    assert results["mean_distinct_exemplars_per_query"] == pytest.approx(5 / 3)
    # Query 2 restricted to {0, 2}: the reranker exemplar now ranks class 2 first.
    restricted = results["restricted_by_k"][2]["conditions"]
    assert restricted["rr"]["accuracy"] == pytest.approx(2 / 3)
    assert restricted[CLIP_SELECTOR]["accuracy"] == pytest.approx(1 / 3)

    path = tmp_path / "queries.csv"
    write_query_csv(path, records, ["rr"], [2], 1.0)
    rows = list(csv.DictReader(path.open()))
    assert [row["rr_correct"] for row in rows] == ["True", "True", "False"]
    assert rows[0]["clip_top1_example_idx"] == "10"
    assert rows[2]["rr_pool_contains_true_class"] == "False"


def test_load_selectors_restores_checkpoint_and_skips_disabled_entries(tmp_path):
    from dataclasses import asdict

    from scripts.evaluate_reranker_test import load_selectors

    config = RerankerConfig(clip_dim=6, siglip_dim=7)
    immutable = {
        "clip_model": "ViT-B/32", "retrieval_split": "train", "k": 32,
        "candidate_pool_size": 100, "query_splits": ["train", "val"],
    }
    path = tmp_path / "best.pt"
    torch.save({
        "experiment_name": "m50",
        "model_state_dict": LabelAwareReranker(config).state_dict(),
        "model_config": asdict(config),
        "resolved_config": {"data": {"max_candidates": 50}},
        "epoch": 3,
        "early_stopping": {"best_epoch": 3},
        "metrics": {"restricted_selected_accuracy": 0.74},
        "provenance": {"artifact_immutable_args": immutable},
    }, path)
    cfg = OmegaConf.create({
        "dataset": {"eval_split": "test"},
        "selectors": {
            "m50": {"checkpoint": str(path), "pool_size": None},
            "disabled": {"checkpoint": None, "pool_size": None},
        },
        "reranker": {"allow_non_best": False, "require_matching_provenance": True},
    })
    expected = {"clip_model": "ViT-B/32", "retrieval_split": "train"}

    specs, models = load_selectors(cfg, expected, [16, 32], torch.device("cpu"))

    assert [spec["name"] for spec in specs] == ["m50"]
    assert specs[0]["pool_size"] == specs[0]["training_pool_size"] == 50
    assert specs[0]["training_k"] == 32
    assert not models["m50"].training
    with pytest.raises(ValueError, match="teacher K=32"):
        load_selectors(cfg, expected, [16], torch.device("cpu"))
    with pytest.raises(ValueError, match="differ from its training data"):
        load_selectors(cfg, {**expected, "clip_model": "ViT-L/14"}, [32], torch.device("cpu"))
    cfg.dataset.eval_split = "val"
    with pytest.raises(ValueError, match="trained or selected on split"):
        load_selectors(cfg, expected, [32], torch.device("cpu"))


def test_fp16_tied_pools_match_teacher_and_fp32_queries_reorder_them():
    # fp16 dot products of unit vectors round to many exactly tied
    # similarities, as in the real CLIP files. Ranking must use the stored
    # dtype and a stable sort; an fp32 recomputation breaks the ties.
    rng = np.random.default_rng(3)
    retrieval = _unit(rng.normal(size=(200, 64))).astype(np.float16)
    queries = _unit(rng.normal(size=(8, 64))).astype(np.float16)
    examples = lambda n: [SimpleNamespace(label=i % 3) for i in range(n)]
    cfg = OmegaConf.create({
        "experiment": {"seed": 0},
        "targets": {"k": 2},
        "retrieval": {"candidate_pool_size": 20, "exclude_same_class": False},
        "dataset": {"query_splits": ["val"], "retrieval_split": "train"},
        "limits": {"max_queries_per_split": {"val": None}},
    })
    teacher_tasks, _ = build_teacher_tasks(
        cfg, {"val": SimpleNamespace(examples=examples(8), clip_embeddings=queries)},
        SimpleNamespace(examples=examples(200), clip_embeddings=retrieval),
        {"val": _unit(rng.normal(size=(8, 5)))}, _unit(rng.normal(size=(3, 5))),
    )

    reordered = False
    for task in teacher_tasks:
        query = queries[task["query_idx"]]
        order, _ = clip_candidate_pool(retrieval, query, 20)
        np.testing.assert_array_equal(order, task["candidate_indices"])
        fp32_order, _ = clip_candidate_pool(
            retrieval.astype(np.float32), query.astype(np.float32), 20
        )
        reordered |= not np.array_equal(fp32_order, order)
    assert reordered


def test_teacher_artifact_verification_accepts_matching_and_rejects_drift(tmp_path):
    import pickle

    from scripts.evaluate_reranker_test import verify_teacher_artifact

    space = _feature_space()
    retrieval_clip = space["retrieval_clip"].astype(np.float16)
    val_clip = space["query_clip"].astype(np.float16)
    pools = [clip_candidate_pool(retrieval_clip, val_clip[q], 5) for q in range(4)]
    artifact = {
        "immutable_args": {"retrieval_split": "train"},
        "records": [
            SimpleNamespace(
                query_split="val", query_idx=q,
                candidate_indices=order, candidate_similarities=similarities,
            )
            for q, (order, similarities) in enumerate(pools)
        ],
        "feature_tables": {
            "clip_image_embeddings_by_split": {
                "train": retrieval_clip.astype(np.float32), "val": val_clip.astype(np.float32),
            },
            "siglip_image_embeddings_by_split": {"train": space["retrieval_siglip"]},
            "siglip_class_text_embeddings": space["label_siglip"],
        },
    }
    path = tmp_path / "teacher.pkl"
    path.write_bytes(pickle.dumps(artifact))
    space["retrieval_clip"] = retrieval_clip.astype(np.float32)
    tables = _tables(space)

    result = verify_teacher_artifact(path, [{"name": "rr", "artifact_sha256": None}], retrieval_clip, tables, 3)
    assert result["teacher_verified_pools"] == 3

    with pytest.raises(ValueError, match="different teacher artifact"):
        verify_teacher_artifact(path, [{"name": "rr", "artifact_sha256": "0" * 64}], retrieval_clip, tables, 3)
    with pytest.raises(ValueError, match="Rebuilt CLIP pool differs"):
        verify_teacher_artifact(path, [], retrieval_clip.astype(np.float32) * 2, tables, 3)
    drifted = _tables({**space, "label_siglip": space["label_siglip"] + 1e-3})
    with pytest.raises(ValueError, match="SigLIP class text"):
        verify_teacher_artifact(path, [], retrieval_clip, drifted, 3)


def test_resume_rejects_checkpoints_from_different_settings(tmp_path):
    import pickle

    from scripts.evaluate_reranker_test import (
        METHOD, RESUME_INVARIANT_ARGS, SCHEMA_VERSION, _load_resume_records,
    )

    records = _records()[:1]
    cfg = OmegaConf.create({
        "dataset": {"eval_split": "test"},
        "scoring": {"mode": "prefix_kv_cache", "candidate_batch_size": 8},
    })
    run_args = {
        "schema_version": SCHEMA_VERSION,
        "selectors": [{"name": "rr", "checkpoint_sha256": "abc", "pool_size": 50}],
        "resolved_config": OmegaConf.to_container(cfg),
        **{key: f"value-{key}" for key in RESUME_INVARIANT_ARGS},
    }
    task_by_query = {0: {
        "selected_example_indices": records[0].selected_example_indices,
        "distractor_class_indices": records[0].distractor_class_indices,
    }}

    def resume(saved_args, current_cfg=cfg, tasks=task_by_query):
        path = tmp_path / "resume.pkl"
        path.write_bytes(pickle.dumps({
            "method": METHOD, "args": saved_args, "candidate_labels": ["a", "b", "c"],
            "query_indices": [0], "records": records,
        }))
        return _load_resume_records(path, current_cfg, run_args, ["a", "b", "c"], [0, 1], tasks)

    assert resume(run_args) == records
    with pytest.raises(ValueError, match="different selectors or pool sizes"):
        resume({**run_args, "selectors": [{"name": "rr", "checkpoint_sha256": "abc", "pool_size": 100}]})
    with pytest.raises(ValueError, match="differs in clip_embedding_dtype"):
        resume({**run_args, "clip_embedding_dtype": "float32"})
    with pytest.raises(ValueError, match="different scoring.mode"):
        resume(run_args, OmegaConf.merge(cfg, {"scoring": {"mode": "full_sequence_batch"}}))
    with pytest.raises(ValueError, match="selected_example_indices differ"):
        resume(run_args, tasks={0: {**task_by_query[0], "selected_example_indices": {}}})
