from copy import deepcopy

from omegaconf import OmegaConf

from scripts.train_reranker import (
    _resolve_experiment_name,
    _stratified_training_subset,
)


def _config():
    return {
        "experiment": {"name": "auto", "seed": 40},
        "data": {"artifact_path": "/machine-a/teacher.pkl", "target": "margin"},
        "model": {"architecture": "interaction_mlp", "hidden_dim": 256},
        "objective": {
            "name": "hybrid_listwise_pairwise",
            "hybrid_listwise_weight": 0.1,
        },
        "optimization": {"learning_rate": 3e-4},
    }


def test_auto_experiment_name_is_stable_and_ignores_artifact_location():
    first = _config()
    second = deepcopy(first)
    second["data"]["artifact_path"] = "/machine-b/copied-teacher.pkl"
    first_name = _resolve_experiment_name(OmegaConf.create(first))
    second_name = _resolve_experiment_name(OmegaConf.create(second))
    assert first_name == second_name
    assert first_name.startswith(
        "interaction_mlp-margin-hybrid_listwise_pairwise-seed40-"
    )


def test_auto_experiment_name_ignores_visual_cache_location():
    first = _config()
    first["data"]["visual_token_cache_path"] = "/machine-a/tokens"
    second = deepcopy(first)
    second["data"]["visual_token_cache_path"] = "/machine-b/tokens"
    assert _resolve_experiment_name(OmegaConf.create(first)) == (
        _resolve_experiment_name(OmegaConf.create(second))
    )


def test_auto_experiment_name_changes_with_ablation_and_honors_explicit_name():
    baseline = _config()
    ablation = deepcopy(baseline)
    ablation["objective"]["hybrid_listwise_weight"] = 0.5
    assert _resolve_experiment_name(OmegaConf.create(baseline)) != (
        _resolve_experiment_name(OmegaConf.create(ablation))
    )

    baseline["experiment"]["name"] = "my-explicit-run"
    assert _resolve_experiment_name(OmegaConf.create(baseline)) == "my-explicit-run"


def test_training_subset_is_class_stratified_and_reproducible():
    class Dataset:
        records = [
            type("Record", (), {"true_class_idx": label})()
            for label in ([0] * 8 + [1] * 2)
        ]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return index

    first = _stratified_training_subset(
        Dataset(), max_queries=None, fraction=0.5, seed=7
    )
    second = _stratified_training_subset(
        Dataset(), max_queries=None, fraction=0.5, seed=7
    )
    labels = [Dataset.records[index].true_class_idx for index in first.indices]

    assert first.indices == second.indices
    assert labels.count(0) == 4
    assert labels.count(1) == 1
