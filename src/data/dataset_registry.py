"""
Registry of fine-grained datasets built on FineGrainedHFDataset.

Adding a new candidate dataset (e.g. FGVC Aircraft, Oxford-IIIT Pets) requires
only a new entry here — no new dataset class or script changes needed.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FineGrainedDatasetSpec:
    hf_repo_ids: List[str]
    data_dir: str
    display_name: str
    # Roughly how visually similar/fine-grained the classes are; informs
    # priors about how much CLIP-retrieval is expected to help. Not used
    # programmatically — documentation only.
    notes: str = field(default="")


FINE_GRAINED_DATASETS = {
    "cub_200": FineGrainedDatasetSpec(
        hf_repo_ids=["Multimodal-Fatima/CUB_train", "Multimodal-Fatima/CUB_test"],
        data_dir="data/cub_200",
        display_name="CUB-200-2011",
        notes="200 bird species; fine-grained but with more distinct visual cues "
              "(color, pattern) than car models — candidate for CLIP retrieval helping some.",
    ),
    "fgvc_aircraft": FineGrainedDatasetSpec(
        hf_repo_ids=["Multimodal-Fatima/FGVC_Aircraft_train", "Multimodal-Fatima/FGVC_Aircraft_test"],
        data_dir="data/fgvc_aircraft",
        display_name="FGVC Aircraft",
        notes="100 aircraft variants; many models nearly identical in silhouette — "
              "strong candidate for CLIP retrieval providing little lift over 0-shot.",
    ),
    "oxford_pets": FineGrainedDatasetSpec(
        hf_repo_ids=["Multimodal-Fatima/OxfordPets_train", "Multimodal-Fatima/OxfordPets_test"],
        data_dir="data/oxford_pets",
        display_name="Oxford-IIIT Pets",
        notes="37 cat/dog breeds; coarser-grained than cars/aircraft/birds — "
              "likely easiest for both 0-shot and CLIP retrieval.",
    ),
}


def get_dataset_spec(name: str) -> FineGrainedDatasetSpec:
    if name not in FINE_GRAINED_DATASETS:
        raise ValueError(
            f"Unknown fine-grained dataset: {name}. "
            f"Available: {list(FINE_GRAINED_DATASETS.keys())}"
        )
    return FINE_GRAINED_DATASETS[name]
