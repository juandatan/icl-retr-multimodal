"""Dataset metadata for the production CUB-200 pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FineGrainedDatasetSpec:
    hf_repo_ids: tuple[str, ...]
    data_dir: str
    display_name: str


FINE_GRAINED_DATASETS = {
    "cub_200": FineGrainedDatasetSpec(
        hf_repo_ids=("Multimodal-Fatima/CUB_train", "Multimodal-Fatima/CUB_test"),
        data_dir="data/cub_200",
        display_name="CUB-200-2011",
    ),
}


def get_dataset_spec(name: str) -> FineGrainedDatasetSpec:
    if name not in FINE_GRAINED_DATASETS:
        raise ValueError(
            f"Unknown fine-grained dataset: {name}. "
            f"Available: {list(FINE_GRAINED_DATASETS.keys())}"
        )
    return FINE_GRAINED_DATASETS[name]
