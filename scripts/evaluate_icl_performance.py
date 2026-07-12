"""
Evaluate ICL performance with a learned reranker.

Retrieves K in-context examples per query using the reranker model and measures
Idefics2 classification accuracy.  For 0-shot and CLIP-retrieval baselines on
Stanford Cars, see evaluate_stanford_cars_baselines.py.

Usage:
    # Stanford Cars with image-level split
    python scripts/evaluate_icl_performance.py \
        --dataset stanford_cars \
        --reranker-checkpoint outputs/reranker_checkpoints/best_model.pt \
        --image-split-path data/stanford_cars/image_split.json \
        --k 1

    # Mini-ImageNet
    python scripts/evaluate_icl_performance.py \
        --dataset mini_imagenet \
        --reranker-checkpoint outputs/reranker_checkpoints/best_model.pt \
        --k 1
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import random
import shutil

import numpy as np
import torch
from tqdm import tqdm
import clip

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.stanford_cars import StanfordCarsDataset
from data.mini_imagenet import MiniImageNetDataset
from data.fine_grained_hf_dataset import FineGrainedHFDataset
from data.dataset_registry import FINE_GRAINED_DATASETS, get_dataset_spec
from data.marginal_utility_dataset import InteractionFeaturesConfig
from data.base_dataset import ClassificationExample
from models.mlp_reranker import MLPReranker
from models.patch_cross_attention_reranker import PatchCrossAttentionReranker
from models.idefics2_wrapper import Idefics2Wrapper
from data.marginal_utility_image_dataset import clip_transform
from utils.multigpu_utils import MultiGPUManager, merge_dict_results
from utils.imagenet_names import get_readable_name, get_synset_id, IMAGENET_SYNSET_TO_NAME
from utils.eval_utils import save_eval_results

_CLIP_TRANSFORM = clip_transform(224)


def top_k_by_score(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores, sorted descending. O(n) via argpartition."""
    k = min(k, len(scores))
    partition_idx = np.argpartition(scores, -k)[-k:]
    return partition_idx[np.argsort(scores[partition_idx])[::-1]]


class CombinedDataset:
    """Combines multiple datasets for evaluation."""
    def __init__(self, datasets):
        self.examples = []
        embeddings_list = []
        self._datasets = datasets
        self._dataset_offsets = [0]

        for ds in datasets:
            # Reindex examples
            for ex in ds.examples:
                new_ex = ClassificationExample(
                    index=len(self.examples),
                    image_path=ex.image_path,
                    label=ex.label,
                    label_name=ex.label_name,
                    split=ex.split,
                    _hf_index=ex._hf_index
                )
                self.examples.append(new_ex)
            embeddings_list.append(ds.clip_embeddings)
            self._dataset_offsets.append(len(self.examples))

        self.clip_embeddings = np.vstack(embeddings_list)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i in range(len(self._datasets)):
            if idx < self._dataset_offsets[i + 1]:
                local_idx = idx - self._dataset_offsets[i]
                return self._datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")

    def __len__(self):
        return len(self.examples)


def determine_retrieval_split(eval_split: str, retrieval_split: Optional[str] = None) -> str:
    """Determine retrieval split based on eval split if not explicitly provided."""
    if retrieval_split is not None:
        return retrieval_split
    if eval_split == "test":
        return "train"
    return eval_split


def get_cache_path(
    dataset_name: str,
    method: str,
    k: int,
    num_queries: int,
    seed: int,
    reranker_checkpoint: str = None,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None,
    use_all_classes: bool = False,
    candidate_pool_size: Optional[int] = None,
) -> Path:
    """Generate cache path for evaluation results."""
    cache_dir = Path("outputs/icl_evaluation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_id = f"{dataset_name}_{method}_k{k}_n{num_queries}_seed{seed}"

    if reranker_checkpoint and method == "reranker":
        ckpt_name = Path(reranker_checkpoint).stem
        config_id += f"_{ckpt_name}"

    if candidate_pool_size is not None:
        config_id += f"_pool{candidate_pool_size}"

    if use_generative:
        config_id += "_generative"

    if prefilter_topk is not None:
        config_id += f"_prefilter{prefilter_topk}"

    if use_all_classes:
        config_id += "_allclasses"

    return cache_dir / f"{config_id}.pkl"


def load_cached_results(cache_path: Path) -> Optional[Dict]:
    """Load cached evaluation results if they exist.

    Returns a dict with 'predictions' list and 'completed_queries' set.
    """
    if cache_path.exists():
        print(f"Found cached results at {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)

        # Handle old cache format (convert to incremental format)
        if 'predictions' in cached and 'completed_queries' not in cached:
            completed = {pred['query_idx'] for pred in cached['predictions']}
            cached['completed_queries'] = completed

        num_completed = len(cached.get('completed_queries', set()))
        print(f"✓ Loaded cache with {num_completed} completed queries")
        return cached
    return None


def save_cached_results(cache_path: Path, results: Dict):
    """Save evaluation results to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Cached results saved to {cache_path}")


def resolve_embeddings_cache_path(dataset_name: str, split: str,
                                   embeddings_kaggle_dataset: Optional[str]) -> Optional[Path]:
    """
    Resolve a local path to clip_embeddings_{split}.pkl sourced from a Kaggle dataset,
    so scripts can reuse pre-built embeddings instead of rebuilding them locally.

    Checks the Kaggle-mounted input path first (when running inside a Kaggle notebook
    with the dataset attached), then falls back to downloading via the Kaggle CLI.

    Returns None if embeddings_kaggle_dataset is not provided, in which case callers
    should fall back to the dataset's own local default path.
    """
    if not embeddings_kaggle_dataset:
        return None

    filename = f"clip_embeddings_{split}.pkl"
    slug = embeddings_kaggle_dataset.split('/')[-1]

    # Matches the mount path Kaggle notebooks actually use when a dataset is attached
    # as input (see scripts/upload_embeddings_to_kaggle.py's printed guidance).
    mounted_path = Path(f"/kaggle/input/{slug}/{filename}")
    if mounted_path.exists():
        print(f"✓ Using mounted Kaggle input: {mounted_path}")
        return mounted_path

    return download_from_kaggle_dataset(
        kaggle_dataset=embeddings_kaggle_dataset,
        filename=filename,
        cache_dir="./cache/embeddings",
    )


def load_dataset(dataset_name: str, split: str = "test",
                 image_split_path: Optional[str] = None,
                 embeddings_kaggle_dataset: Optional[str] = None):
    """Load dataset for evaluation."""
    print(f"\nLoading {dataset_name} dataset ({split} split)...")

    if dataset_name == "stanford_cars":
        dataset = StanfordCarsDataset(
            split=split,
            data_dir="data/stanford_cars",
            class_split_seed=42,
            image_split_path=image_split_path,
        )
    elif dataset_name == "mini_imagenet":
        dataset = MiniImageNetDataset(
            split=split,
            data_dir="data/mini_imagenet",
            class_split_seed=42
        )
    elif dataset_name in FINE_GRAINED_DATASETS:
        spec = get_dataset_spec(dataset_name)
        dataset = FineGrainedHFDataset(
            hf_repo_ids=spec.hf_repo_ids,
            split=split,
            data_dir=spec.data_dir,
            class_split_seed=42,
            image_split_path=image_split_path,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load CLIP embeddings, preferring a Kaggle-sourced cache when configured
    embeddings_cache_path = resolve_embeddings_cache_path(
        dataset_name, split, embeddings_kaggle_dataset
    )
    success = dataset.load_clip_embeddings(cache_path=embeddings_cache_path)
    if not success:
        raise FileNotFoundError(
            f"CLIP embeddings not found. Please run: "
            f"python scripts/build_clip_embeddings.py --dataset {dataset_name} --splits {split}"
        )

    print(f"✓ Loaded {len(dataset)} examples")
    print(f"✓ Embeddings shape: {dataset.clip_embeddings.shape}")
    print(f"✓ Num classes: {dataset.num_classes}")

    return dataset


def load_eval_datasets(dataset_name: str, eval_split: str, retrieval_split: str,
                       image_split_path: Optional[str] = None,
                       embeddings_kaggle_dataset: Optional[str] = None):
    """Load and return (test_dataset, retrieval_dataset), reusing objects where splits overlap."""
    if eval_split == "val+test":
        test_dataset = CombinedDataset([
            load_dataset(dataset_name, split="val", image_split_path=image_split_path,
                        embeddings_kaggle_dataset=embeddings_kaggle_dataset),
            load_dataset(dataset_name, split="test", image_split_path=image_split_path,
                        embeddings_kaggle_dataset=embeddings_kaggle_dataset),
        ])
    else:
        test_dataset = load_dataset(dataset_name, split=eval_split,
                                    image_split_path=image_split_path,
                                    embeddings_kaggle_dataset=embeddings_kaggle_dataset)

    if retrieval_split == eval_split and eval_split != "val+test":
        retrieval_dataset = test_dataset
    else:
        retrieval_dataset = load_dataset(dataset_name, split=retrieval_split,
                                         image_split_path=image_split_path,
                                         embeddings_kaggle_dataset=embeddings_kaggle_dataset)

    return test_dataset, retrieval_dataset


def build_all_classes_label_mapping(dataset_name: str, preloaded: Dict = None,
                                    image_split_path: Optional[str] = None,
                                    embeddings_kaggle_dataset: Optional[str] = None) -> Dict:
    """Build label_name -> label mapping from all splits, reusing any pre-loaded datasets."""
    preloaded = preloaded or {}
    mapping = {}
    for split in ("train", "val", "test"):
        ds = preloaded.get(split) or load_dataset(dataset_name, split=split,
                                                   image_split_path=image_split_path,
                                                   embeddings_kaggle_dataset=embeddings_kaggle_dataset)
        for ex in ds.examples:
            if ex.label_name not in mapping:
                mapping[ex.label_name] = ex.label
    return mapping


def download_from_kaggle_dataset(kaggle_dataset: str, filename: str, cache_dir: str = "./cache") -> Path:
    """
    Download a file from Kaggle dataset, with local caching.

    Args:
        kaggle_dataset: Kaggle dataset name (username/dataset-name)
        filename: Filename to download
        cache_dir: Local cache directory

    Returns:
        Path to cached file
    """
    import subprocess
    import tempfile

    cache_path = Path(cache_dir) / kaggle_dataset.replace('/', '_') / filename

    # Return cached file if it exists
    if cache_path.exists():
        print(f"✓ Using cached file: {cache_path}")
        return cache_path

    print(f"Downloading from Kaggle dataset: {kaggle_dataset}/{filename}")

    # Create cache directory
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Download to temporary location first
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download entire dataset
        cmd = ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(temp_path), "--unzip"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download from Kaggle: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")

        # Move the specific file to cache
        downloaded_file = temp_path / filename
        if not downloaded_file.exists():
            raise FileNotFoundError(f"File {filename} not found in dataset {kaggle_dataset}")

        shutil.move(str(downloaded_file), str(cache_path))

    print(f"✓ Downloaded and cached to: {cache_path}")
    return cache_path


def load_reranker(checkpoint_path: str, device: str, kaggle_dataset: str = None, force_refresh: bool = False):
    """
    Load trained reranker model from checkpoint.

    Supports both MLP and patch cross-attention architectures.

    Args:
        checkpoint_path: Local path or filename of checkpoint
        device: Device to load model on
        kaggle_dataset: Optional Kaggle dataset (username/dataset-name) to download from
        force_refresh: If True, re-download from Kaggle even if cached

    Returns:
        Tuple of (model, interaction_features_or_None)
        For MLP: (MLPReranker, InteractionFeaturesConfig)
        For patch cross-attention: (PatchCrossAttentionReranker, None)
    """
    # Handle Kaggle dataset download
    if kaggle_dataset:
        cache_dir = "./cache/checkpoints"
        cache_path = Path(cache_dir) / kaggle_dataset.replace('/', '_') / Path(checkpoint_path).name

        if force_refresh and cache_path.exists():
            print(f"Force refresh: removing cached file {cache_path}")
            cache_path.unlink()

        # Also check mounted Kaggle input
        kaggle_input_path = Path(f"/kaggle/input/datasets/{kaggle_dataset}") / Path(checkpoint_path).name
        if kaggle_input_path.exists():
            checkpoint_path = str(kaggle_input_path)
            print(f"✓ Using mounted Kaggle input: {checkpoint_path}")
        else:
            checkpoint_path = str(download_from_kaggle_dataset(
                kaggle_dataset=kaggle_dataset,
                filename=Path(checkpoint_path).name,
                cache_dir=cache_dir
            ))

    checkpoint_path = Path(checkpoint_path)
    print(f"\nLoading reranker from {checkpoint_path}...")

    # Load checkpoint (weights_only=False needed for full checkpoint with config/optimizer state)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    config = checkpoint['config']
    model_config = config['model']

    # Detect architecture
    architecture = model_config.get('architecture', 'mlp')
    print(f"  Architecture: {architecture}")

    if architecture in ('cross_attention', 'patch_attention'):
        model = PatchCrossAttentionReranker(
            clip_model_name=model_config.get('clip_model_name', 'ViT-B/32'),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_attention_heads=model_config.get('num_attention_heads', 8),
            num_attention_layers=model_config.get('num_attention_layers', 2),
            feedforward_dims=model_config.get('feedforward_dims', [256, 128]),
            dropout=model_config.get('dropout', 0.1),
            use_sigmoid=model_config.get('use_sigmoid', False),
            freeze_clip=True,
            pooling_method=model_config.get('pooling_method', 'cls')
        )
        interaction_features = None
    else:
        interaction_features = InteractionFeaturesConfig(
            use_product=model_config.get('use_product', False),
            use_difference=model_config.get('use_difference', False),
            use_l2_distance=model_config.get('use_l2_distance', False)
        )
        model = MLPReranker(
            embedding_dim=model_config['embedding_dim'],
            hidden_dims=model_config['hidden_dims'],
            dropout=model_config.get('dropout', 0.1),
            interaction_features=interaction_features,
            use_sigmoid=model_config.get('use_sigmoid', False)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded {architecture} model with {model.get_num_parameters():,} parameters")
    if interaction_features:
        print(f"✓ Interaction features: {interaction_features}")

    return model, interaction_features


def compute_similarity(query_emb: np.ndarray, candidate_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and candidates."""
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    candidate_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)

    # Cosine similarity
    similarities = candidate_norms @ query_norm
    return similarities


def retrieve_by_clip(
    query_emb: np.ndarray,
    train_dataset,
    k: int,
    exclude_indices: Optional[List[int]] = None
) -> List[int]:
    """Retrieve top-K examples using CLIP similarity.

    Args:
        query_emb: Query embedding
        train_dataset: Dataset to retrieve from
        k: Number of examples to retrieve
        exclude_indices: Indices to exclude from retrieval (e.g., the query itself)
    """
    candidate_embs = train_dataset.clip_embeddings

    # Compute similarities
    similarities = compute_similarity(query_emb, candidate_embs)

    # Exclude specified indices by setting their similarity to -inf
    if exclude_indices:
        for idx in exclude_indices:
            if 0 <= idx < len(similarities):
                similarities[idx] = -np.inf

    top_k_indices = top_k_by_score(similarities, k)

    return top_k_indices.tolist()


def retrieve_by_reranker(
    query_emb: np.ndarray,
    train_dataset,
    reranker: MLPReranker,
    interaction_features: InteractionFeaturesConfig,
    device: str,
    k: int,
    exclude_indices: Optional[List[int]] = None
) -> List[int]:
    """Retrieve top-K examples using learned reranker.

    Args:
        query_emb: Query embedding
        train_dataset: Dataset to retrieve from
        reranker: Trained reranker model
        interaction_features: Interaction features config
        device: Device to run on
        k: Number of examples to retrieve
        exclude_indices: Indices to exclude from retrieval (e.g., the query itself)
    """
    candidate_embs = train_dataset.clip_embeddings

    # Compute CLIP similarities first (needed as input to reranker)
    similarities = compute_similarity(query_emb, candidate_embs)

    # Prepare batch inputs for reranker
    query_emb_tensor = torch.from_numpy(query_emb).float().to(device)
    candidate_embs_tensor = torch.from_numpy(candidate_embs).float().to(device)
    similarities_tensor = torch.from_numpy(similarities).float().to(device)

    # Expand query embedding to match batch size
    query_emb_batch = query_emb_tensor.unsqueeze(0).expand(len(candidate_embs), -1)

    # Compute interaction features if needed
    product = None
    difference = None
    l2_distance = None

    if interaction_features.use_product:
        product = query_emb_batch * candidate_embs_tensor
    if interaction_features.use_difference:
        difference = query_emb_batch - candidate_embs_tensor
    if interaction_features.use_l2_distance:
        l2_distance = torch.norm(query_emb_batch - candidate_embs_tensor, dim=1, keepdim=True)

    # Get predictions
    with torch.no_grad():
        utilities = reranker(
            query_emb_batch,
            candidate_embs_tensor,
            similarities_tensor.unsqueeze(1),
            product=product,
            difference=difference,
            l2_distance=l2_distance
        ).squeeze().cpu().numpy()

    # Exclude specified indices by setting their utility to -inf
    if exclude_indices:
        for idx in exclude_indices:
            if 0 <= idx < len(utilities):
                utilities[idx] = -np.inf

    top_k_indices = top_k_by_score(utilities, k)

    return top_k_indices.tolist()


def extract_patch_features_for_dataset(
    dataset,
    reranker: PatchCrossAttentionReranker,
    device: str,
    batch_size: int = 64
) -> torch.Tensor:
    """Pre-extract patch features for all images in a dataset.

    Args:
        dataset: Dataset supporting __getitem__ returning (example, image) and __len__
        reranker: Model with extract_patch_features method
        device: Device for extraction
        batch_size: Batch size for extraction

    Returns:
        Tensor of shape (N, num_patches, embed_dim) on CPU
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot extract patch features from empty dataset")

    # Probe shape from first image
    _, first_img = dataset[0]
    with torch.no_grad():
        probe = reranker.extract_patch_features(
            _CLIP_TRANSFORM(first_img.convert('RGB')).unsqueeze(0).to(device)
        )
    num_patches, embed_dim = probe.shape[1], probe.shape[2]

    # Pre-allocate output tensor to avoid double memory peak from list + cat
    result = torch.empty(n, num_patches, embed_dim)

    print(f"  Pre-extracting patch features for {n} images...")
    for batch_start in tqdm(range(0, n, batch_size), desc="Extracting patches"):
        batch_end = min(batch_start + batch_size, n)
        batch_images = []
        for idx in range(batch_start, batch_end):
            _, img = dataset[idx]
            batch_images.append(_CLIP_TRANSFORM(img.convert('RGB')))

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            features = reranker.extract_patch_features(batch_tensor)
            result[batch_start:batch_end] = features.cpu()

    print(f"  ✓ Extracted patch features: {result.shape}")
    return result


def retrieve_by_patch_reranker(
    query_image,
    query_emb: np.ndarray,
    train_dataset,
    reranker: PatchCrossAttentionReranker,
    device: str,
    k: int,
    exclude_indices: Optional[List[int]] = None,
    prefilter_n: int = 50,
    batch_size: int = 32,
    candidate_patch_features: Optional[torch.Tensor] = None
) -> List[int]:
    """Retrieve top-K examples using patch cross-attention reranker.

    Uses CLIP similarity to pre-filter candidates, then reranks with
    the cross-attention model using patch-level features.

    Args:
        query_image: Query PIL image
        query_emb: Query CLIP embedding (for pre-filtering)
        train_dataset: Dataset to retrieve from
        reranker: Trained patch cross-attention model
        device: Device to run on
        k: Number of examples to retrieve
        exclude_indices: Indices to exclude from retrieval
        prefilter_n: Number of candidates to pre-filter with CLIP before reranking
        batch_size: Batch size for reranker scoring
        candidate_patch_features: Pre-extracted patch features for the retrieval dataset
            (shape: N x num_patches x embed_dim). If None, extracts on the fly.
    """
    candidate_embs = train_dataset.clip_embeddings

    # Step 1: Pre-filter with CLIP similarity using argpartition (O(n) vs O(n log n))
    similarities = compute_similarity(query_emb, candidate_embs)

    if exclude_indices:
        for idx in exclude_indices:
            if 0 <= idx < len(similarities):
                similarities[idx] = -np.inf

    prefilter_indices = top_k_by_score(similarities, prefilter_n)
    prefilter_similarities = similarities[prefilter_indices]

    # Step 2: Extract query patch features
    query_tensor = _CLIP_TRANSFORM(query_image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        query_patches = reranker.extract_patch_features(query_tensor)  # (1, num_patches, embed_dim)

    # Step 3: Score candidates in batches
    all_utilities = []

    for batch_start in range(0, len(prefilter_indices), batch_size):
        batch_indices = prefilter_indices[batch_start:batch_start + batch_size]
        batch_sims = prefilter_similarities[batch_start:batch_start + batch_size]

        # Get candidate patch features (pre-extracted or on-the-fly)
        if candidate_patch_features is not None:
            candidate_patches = candidate_patch_features[batch_indices].to(device)
        else:
            candidate_images = []
            for idx in batch_indices:
                _, img = train_dataset[idx]
                candidate_images.append(_CLIP_TRANSFORM(img.convert('RGB')))
            candidate_batch = torch.stack(candidate_images).to(device)
            with torch.no_grad():
                candidate_patches = reranker.extract_patch_features(candidate_batch)

        with torch.no_grad():
            query_patches_expanded = query_patches.expand(len(batch_indices), -1, -1)
            sim_tensor = torch.from_numpy(batch_sims).float().unsqueeze(1).to(device)
            utilities = reranker(query_patches_expanded, candidate_patches, sim_tensor)
            all_utilities.append(utilities.squeeze(1).cpu().numpy())

    all_utilities = np.concatenate(all_utilities)

    # Step 4: Return top-K by utility
    top_k_local = top_k_by_score(all_utilities, k)
    top_k_indices = prefilter_indices[top_k_local]

    return top_k_indices.tolist()


def get_oracle_candidates(
    image,
    true_class: str,
    all_classes: List[str],
    clip_text_features: torch.Tensor,
    clip_model,
    preprocess,
    device: str,
    top_k: int = 10
) -> List[str]:
    """
    Get top-(K-1) candidates using CLIP, then append true class.

    This guarantees the true class is always included without conditional logic.

    Args:
        image: PIL Image
        true_class: True class name (readable)
        all_classes: List of all candidate class names
        clip_text_features: Precomputed CLIP text embeddings for all classes
        clip_model: CLIP model
        preprocess: CLIP preprocessing function
        device: Device to run on
        top_k: Number of candidates to return

    Returns:
        List of K candidates with true class always at the end
    """
    # Get CLIP embedding for image
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Compute similarities
    similarities = (image_features @ clip_text_features.T).squeeze(0)

    # Get top-(K-1) candidates
    top_indices = similarities.topk(top_k - 1).indices.cpu().numpy()
    top_candidates = [all_classes[i] for i in top_indices]

    # Always append true class at the end
    return top_candidates + [true_class]


def _evaluate_queries(
    query_indices: List[int],
    test_dataset,
    retrieval_dataset,
    llava_model: Idefics2Wrapper,
    retrieval_fn,
    k: int,
    candidate_pool_size: int,
    use_generative: bool,
    prefilter_topk: Optional[int],
    clip_model,
    clip_preprocess,
    clip_text_features,
    device: str,
    return_predictions: bool = False,
    progress_desc: str = "Evaluating",
    candidate_batch_size: int = 8,
    cache_path: Optional[Path] = None,
    save_frequency: int = 50,
    all_classes_label_mapping: Optional[Dict] = None,
    force_recompute: bool = False,
) -> Dict:
    """
    Shared evaluation logic for both single-GPU and multi-GPU modes.

    Args:
        query_indices: List of query indices to evaluate
        test_dataset: Test dataset
        retrieval_dataset: Dataset to retrieve ICL examples from
        llava_model: Idefics2 model for classification
        retrieval_fn: Function(query_emb, dataset, k) -> List[indices]
        k: Number of ICL examples to include in prompt
        candidate_pool_size: Number of candidates to retrieve and rerank
        use_generative: Whether to use generative evaluation
        prefilter_topk: Number of candidates for CLIP pre-filtering (or None)
        clip_model: CLIP model for pre-filtering (or None)
        clip_preprocess: CLIP preprocessing (or None)
        clip_text_features: Precomputed CLIP text features (or None)
        device: Device string
        return_predictions: Whether to return detailed predictions
        progress_desc: Description for progress bar
        all_classes_label_mapping: Optional complete label_name -> label mapping for all classes
        force_recompute: If True, ignore existing cache and recompute from scratch

    Returns:
        Dictionary with evaluation results
    """
    # Load existing cache if available
    completed_queries = set()
    if not force_recompute and cache_path and cache_path.exists():
        cached = load_cached_results(cache_path)
        if cached:
            completed_queries = cached.get('completed_queries', set())
            predictions = cached.get('predictions', [])
            correct = cached.get('correct', 0)
            total = cached.get('total', 0)
            per_class_correct = cached.get('per_class_correct', {})
            per_class_total = cached.get('per_class_total', {})
            print(f"Resuming from {len(completed_queries)} completed queries")
        else:
            predictions = []
            correct = 0
            total = 0
            per_class_correct = {}
            per_class_total = {}
    else:
        predictions = []
        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}

    # Build label_name -> label mapping for O(1) lookups
    if all_classes_label_mapping is not None:
        # Use complete mapping across all splits
        label_name_to_label = all_classes_label_mapping
    else:
        # Use only classes from test_dataset
        label_name_to_label = {ex.label_name: ex.label for ex in test_dataset.examples}

    # Precompute candidate labels for discriminative evaluation
    discriminative_candidate_labels = None
    if not use_generative:
        if all_classes_label_mapping is not None:
            # Use all classes as candidates
            discriminative_candidate_labels = sorted([get_readable_name(label_name) for label_name in all_classes_label_mapping.keys()])
        else:
            # Use only classes from test_dataset
            discriminative_candidate_labels = [get_readable_name(ex.label_name) for ex in test_dataset.examples]
            seen = set()
            discriminative_candidate_labels = [x for x in discriminative_candidate_labels if not (x in seen or seen.add(x))]

    # Precompute all class names for generative evaluation
    all_class_names_sorted = sorted([name for name in IMAGENET_SYNSET_TO_NAME.values()]) if use_generative else []

    # Filter out already completed queries
    queries_to_process = [idx for idx in query_indices if idx not in completed_queries]
    print(f"Processing {len(queries_to_process)} queries ({len(completed_queries)} already cached)")

    for query_idx in tqdm(queries_to_process, desc=progress_desc):
        query_example, query_image = test_dataset[query_idx]
        true_label = query_example.label

        # Retrieve candidates from pool, then take top k for ICL prompt
        context_examples = []
        example_indices = []
        if k > 0:
            query_emb = test_dataset.clip_embeddings[query_idx]
            # Only exclude query if using same dataset for retrieval
            exclude_indices = [query_idx] if test_dataset is retrieval_dataset else None

            # Retrieve candidate_pool_size candidates and rerank them
            all_candidate_indices = retrieval_fn(query_emb, retrieval_dataset, candidate_pool_size, exclude_indices=exclude_indices, query_image=query_image)

            # Take only top k for the ICL prompt
            example_indices = all_candidate_indices[:k]

            for ex_idx in example_indices:
                ex_example, ex_image = retrieval_dataset[ex_idx]
                ex_label_text = get_readable_name(ex_example.label_name)
                context_examples.append((ex_image, ex_label_text))

        # Get candidate labels
        if use_generative:
            if prefilter_topk is not None:
                true_label_readable = get_readable_name(query_example.label_name)
                candidate_label_names = get_oracle_candidates(
                    image=query_image,
                    true_class=true_label_readable,
                    all_classes=all_class_names_sorted,
                    clip_text_features=clip_text_features,
                    clip_model=clip_model,
                    preprocess=clip_preprocess,
                    device=device,
                    top_k=prefilter_topk
                )
            else:
                candidate_label_names = all_class_names_sorted
        else:
            candidate_label_names = discriminative_candidate_labels

        # Query Idefics2
        if use_generative:
            predicted_label_text = llava_model.classify_with_context_generative(
                query_image=query_image,
                context_examples=context_examples,
                candidate_labels=candidate_label_names
            )
            predicted_label_text = get_synset_id(predicted_label_text)
        else:
            predicted_label_text = llava_model.classify_with_context(
                query_image=query_image,
                context_examples=context_examples,
                candidate_labels=candidate_label_names,
                batch_size=candidate_batch_size
            )
            predicted_label_text = get_synset_id(predicted_label_text)

        # Convert prediction to label index using O(1) dict lookup
        predicted_label = label_name_to_label.get(predicted_label_text, -1)

        # Track accuracy
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
        total += 1

        # Track per-class accuracy
        if true_label not in per_class_correct:
            per_class_correct[true_label] = 0
            per_class_total[true_label] = 0

        if is_correct:
            per_class_correct[true_label] += 1
        per_class_total[true_label] += 1

        # Store detailed prediction info
        if return_predictions:
            predictions.append({
                'query_idx': query_idx,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'is_correct': is_correct,
                'example_indices': example_indices,
                'predicted_label_text': predicted_label_text
            })

        # Mark as completed
        completed_queries.add(query_idx)

        # Free activation memory between queries
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Save incrementally every N queries
        if cache_path and len(completed_queries) % save_frequency == 0:
            intermediate_results = {
                'correct': correct,
                'total': total,
                'per_class_correct': per_class_correct,
                'per_class_total': per_class_total,
                'predictions': predictions if return_predictions else [],
                'completed_queries': completed_queries
            }
            save_cached_results(cache_path, intermediate_results)

    # Final save
    final_results = {
        'correct': correct,
        'total': total,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'predictions': predictions if return_predictions else [],
        'completed_queries': completed_queries
    }

    if cache_path:
        save_cached_results(cache_path, final_results)

    return final_results


def evaluate_icl_worker(
    gpu_id: int,
    query_indices: List[int],
    dataset_name: str,
    eval_split: str = "val+test",
    retrieval_split: str = None,
    reranker_checkpoint: str = None,
    kaggle_dataset: str = None,
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    load_in_8bit: bool = False,
    k: int = 1,
    candidate_pool_size: int = 50,
    seed: int = 42,
    return_predictions: bool = False,
    use_reranker: bool = False,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None,
    candidate_batch_size: int = 8,
    cache_path_base: Optional[str] = None,
    use_all_classes: bool = False,
    test_dataset=None,
    retrieval_dataset=None,
    all_classes_label_mapping: Optional[Dict] = None,
    do_image_splitting: bool = False,
    worker_id: Optional[int] = None,
    image_split_path: Optional[str] = None,
    force_recompute: bool = False,
    embeddings_kaggle_dataset: Optional[str] = None,
) -> Dict:
    """Worker function for multi-GPU evaluation. Runs on a single GPU and evaluates a subset of queries.

    test_dataset, retrieval_dataset, and all_classes_label_mapping can be pre-loaded by the
    main process and passed here to avoid HuggingFace Arrow cache deadlocks from concurrent reads.

    worker_id: stable identifier for logging and cache paths. When CUDA_VISIBLE_DEVICES isolates
               each process to a single GPU, gpu_id is always 0 but worker_id preserves the
               original worker index for unique cache filenames.
    """
    worker_id = worker_id if worker_id is not None else gpu_id
    device = f"cuda:{gpu_id}"

    # Determine retrieval split
    retrieval_split = determine_retrieval_split(eval_split, retrieval_split)
    print(f"[Worker {worker_id}] Queries from {eval_split}, candidates from {retrieval_split}")

    # Load datasets if not pre-loaded by main process (single-GPU path)
    if test_dataset is None or retrieval_dataset is None:
        test_dataset, retrieval_dataset = load_eval_datasets(
            dataset_name, eval_split, retrieval_split,
            image_split_path=image_split_path,
            embeddings_kaggle_dataset=embeddings_kaggle_dataset,
        )

        if use_all_classes and all_classes_label_mapping is None:
            print(f"[Worker {worker_id}] Building complete label mapping from all splits...")
            preloaded = {}
            if eval_split != "val+test":
                preloaded[eval_split] = test_dataset
            preloaded[retrieval_split] = retrieval_dataset
            all_classes_label_mapping = build_all_classes_label_mapping(
                dataset_name, preloaded, image_split_path=image_split_path,
                embeddings_kaggle_dataset=embeddings_kaggle_dataset,
            )
            print(f"[Worker {worker_id}] Using {len(all_classes_label_mapping)} classes as candidates")

    print(f"[Worker {worker_id}] {len(test_dataset)} eval examples, {len(retrieval_dataset)} retrieval examples")

    # Load reranker if needed
    reranker = None
    interaction_features = None
    if use_reranker and reranker_checkpoint and k > 0:
        reranker, interaction_features = load_reranker(
            checkpoint_path=reranker_checkpoint,
            device=device,
            kaggle_dataset=kaggle_dataset,
            force_refresh=False
        )

    # Initialize Idefics2
    llava_model = Idefics2Wrapper(
        model_name=llava_model_name,
        device=device,
        load_in_8bit=load_in_8bit,
        do_image_splitting=do_image_splitting
    )

    # Initialize CLIP for oracle pre-filtering if needed
    clip_model = None
    clip_preprocess = None
    clip_text_features = None
    if prefilter_topk is not None and use_generative:
        print(f"[Worker {worker_id}] Loading CLIP for oracle pre-filtering (top-{prefilter_topk})...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

        # Precompute text embeddings for all classes
        all_class_names = sorted([name for name in IMAGENET_SYNSET_TO_NAME.values()])
        text_tokens = clip.tokenize(all_class_names).to(device)
        with torch.no_grad():
            clip_text_features = clip_model.encode_text(text_tokens)
            clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

    # Pre-extract patch features for retrieval dataset if using patch model
    is_patch_model = isinstance(reranker, PatchCrossAttentionReranker) if reranker else False
    candidate_patch_features = None
    if is_patch_model:
        candidate_patch_features = extract_patch_features_for_dataset(
            retrieval_dataset, reranker, device, batch_size=64
        )

    # Create retrieval function
    def retrieval_fn(query_emb, dataset, k_examples, exclude_indices=None, query_image=None):
        if use_reranker and is_patch_model:
            return retrieve_by_patch_reranker(
                query_image=query_image,
                query_emb=query_emb,
                train_dataset=dataset,
                reranker=reranker,
                device=device,
                k=k_examples,
                exclude_indices=exclude_indices,
                prefilter_n=candidate_pool_size,
                candidate_patch_features=candidate_patch_features
            )
        elif use_reranker:
            return retrieve_by_reranker(query_emb, dataset, reranker, interaction_features, device, k_examples, exclude_indices)
        else:
            return retrieve_by_clip(query_emb, dataset, k_examples, exclude_indices)

    print(f"[Worker {worker_id}] Evaluating {len(query_indices)} queries...")

    # Create worker-specific cache path if base path provided
    cache_path = None
    if cache_path_base:
        cache_path = Path(f"{cache_path_base}.worker{worker_id}")

    # Use shared evaluation logic
    return _evaluate_queries(
        query_indices=query_indices,
        test_dataset=test_dataset,
        retrieval_dataset=retrieval_dataset,
        llava_model=llava_model,
        retrieval_fn=retrieval_fn,
        k=k,
        candidate_pool_size=candidate_pool_size,
        use_generative=use_generative,
        prefilter_topk=prefilter_topk,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_text_features=clip_text_features,
        device=device,
        return_predictions=return_predictions,
        progress_desc=f"GPU {gpu_id}",
        candidate_batch_size=candidate_batch_size,
        cache_path=cache_path,
        all_classes_label_mapping=all_classes_label_mapping,
        force_recompute=force_recompute,
    )


def evaluate_icl_multigpu(
    dataset_name: str,
    test_dataset_size: int,
    eval_split: str = "val+test",
    retrieval_split: str = None,
    reranker_checkpoint: Optional[str] = None,
    kaggle_dataset: Optional[str] = None,
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    load_in_8bit: bool = False,
    k: int = 1,
    candidate_pool_size: int = 50,
    num_queries: Optional[int] = None,
    seed: int = 42,
    return_predictions: bool = False,
    use_reranker: bool = False,
    num_gpus: int = 1,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None,
    candidate_batch_size: int = 8,
    cache_path: Optional[Path] = None,
    use_all_classes: bool = False,
    do_image_splitting: bool = True,
    image_split_path: Optional[str] = None,
    force_recompute: bool = False,
    embeddings_kaggle_dataset: Optional[str] = None,
) -> Dict:
    """
    Evaluate ICL performance using multiple GPUs in parallel.

    Datasets are pre-loaded in the main process and passed to workers to avoid
    HuggingFace Arrow cache deadlocks from concurrent reads.
    """
    random.seed(seed)

    # Sample queries
    query_indices = list(range(test_dataset_size))
    total_queries = num_queries if num_queries is not None else test_dataset_size
    if total_queries < test_dataset_size:
        query_indices = random.sample(query_indices, total_queries)

    # Pre-load datasets once in the main process to avoid HF Arrow cache deadlocks in workers
    retrieval_split = determine_retrieval_split(eval_split, retrieval_split)
    print("Pre-loading datasets in main process...")

    test_dataset, retrieval_dataset = load_eval_datasets(
        dataset_name, eval_split, retrieval_split,
        image_split_path=image_split_path,
        embeddings_kaggle_dataset=embeddings_kaggle_dataset,
    )

    all_classes_label_mapping = None
    if use_all_classes:
        print("Building complete label mapping from all splits...")
        preloaded = {}
        if eval_split != "val+test":
            preloaded[eval_split] = test_dataset
        preloaded[retrieval_split] = retrieval_dataset
        all_classes_label_mapping = build_all_classes_label_mapping(
            dataset_name, preloaded, image_split_path=image_split_path,
            embeddings_kaggle_dataset=embeddings_kaggle_dataset,
        )
        print(f"✓ Using {len(all_classes_label_mapping)} classes as candidates")

    print(f"✓ {len(test_dataset)} eval examples, {len(retrieval_dataset)} retrieval examples")

    # Use MultiGPUManager to handle parallel execution
    with MultiGPUManager(num_gpus=num_gpus, verbose=True) as mgr:
        # Run evaluation in parallel
        results = mgr.run_parallel(
            worker_fn=evaluate_icl_worker,
            work_items=query_indices,
            worker_kwargs={
                'dataset_name': dataset_name,
                'eval_split': eval_split,
                'retrieval_split': retrieval_split,
                'reranker_checkpoint': reranker_checkpoint,
                'kaggle_dataset': kaggle_dataset,
                'llava_model_name': llava_model_name,
                'load_in_8bit': load_in_8bit,
                'k': k,
                'candidate_pool_size': candidate_pool_size,
                'seed': seed,
                'return_predictions': return_predictions,
                'use_reranker': use_reranker,
                'use_generative': use_generative,
                'prefilter_topk': prefilter_topk,
                'candidate_batch_size': candidate_batch_size,
                'cache_path_base': str(cache_path) if cache_path else None,
                'use_all_classes': use_all_classes,
                'test_dataset': test_dataset,
                'retrieval_dataset': retrieval_dataset,
                'all_classes_label_mapping': all_classes_label_mapping,
                'do_image_splitting': do_image_splitting,
                'image_split_path': image_split_path,
                'force_recompute': force_recompute,
                'embeddings_kaggle_dataset': embeddings_kaggle_dataset,
            }
        )

    # Merge results using the utility function
    merged_results = merge_dict_results(
        results,
        sum_keys=['correct', 'total'],
        nested_dict_keys=['per_class_correct', 'per_class_total'],
        concat_keys=['predictions'] if return_predictions else []
    )

    # Sort predictions by query_idx if present
    if return_predictions and 'predictions' in merged_results:
        merged_results['predictions'].sort(key=lambda x: x['query_idx'])

    # Compute final metrics
    accuracy = (merged_results['correct'] / merged_results['total']
                if merged_results['total'] > 0 else 0.0)

    per_class_accuracy = {}
    for label in merged_results['per_class_total']:
        per_class_accuracy[label] = (
            merged_results['per_class_correct'][label] / merged_results['per_class_total'][label]
            if merged_results['per_class_total'][label] > 0 else 0.0
        )

    mean_per_class_accuracy = np.mean(list(per_class_accuracy.values())) if per_class_accuracy else 0.0

    final_results = {
        'accuracy': accuracy,
        'mean_per_class_accuracy': mean_per_class_accuracy,
        'correct': merged_results['correct'],
        'total': merged_results['total'],
        'per_class_accuracy': per_class_accuracy
    }

    if return_predictions:
        final_results['predictions'] = merged_results['predictions']

    return final_results


def evaluate_icl(
    test_dataset,
    retrieval_dataset,
    llava_model: Idefics2Wrapper,
    retrieval_fn,
    k: int,
    num_queries: int = None,
    seed: int = 42,
    return_predictions: bool = False,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None,
    device: str = "cuda",
    candidate_batch_size: int = 8,
    all_classes_label_mapping: Optional[Dict] = None,
    candidate_pool_size: int = 50
) -> Dict:
    """
    Evaluate ICL performance using a given retrieval method.

    Args:
        test_dataset: Test examples to query on
        retrieval_dataset: Dataset to retrieve ICL examples from
        llava_model: Idefics2 model for classification
        retrieval_fn: Function(query_emb, retrieval_dataset, k) -> List[example_indices]
        k: Number of in-context examples
        num_queries: Number of test queries to evaluate (None = all)
        seed: Random seed for reproducibility
        return_predictions: If True, return detailed predictions for each query

    Returns:
        Dictionary with accuracy and per-class results
    """
    random.seed(seed)

    # Sample queries if specified
    query_indices = list(range(len(test_dataset)))
    if num_queries is not None:
        query_indices = random.sample(query_indices, min(num_queries, len(test_dataset)))

    # Initialize CLIP for pre-filtering if needed
    clip_model = None
    clip_preprocess = None
    clip_text_features = None
    if prefilter_topk is not None and use_generative:
        print(f"\nLoading CLIP for pre-filtering (top-{prefilter_topk})...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

        # Precompute text embeddings for all classes
        all_class_names = sorted([name for name in IMAGENET_SYNSET_TO_NAME.values()])
        text_tokens = clip.tokenize(all_class_names).to(device)
        with torch.no_grad():
            clip_text_features = clip_model.encode_text(text_tokens)
            clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

    print(f"\nEvaluating on {len(query_indices)} queries with k={k}...")

    # Use shared evaluation logic
    eval_results = _evaluate_queries(
        query_indices=query_indices,
        test_dataset=test_dataset,
        retrieval_dataset=retrieval_dataset,
        llava_model=llava_model,
        retrieval_fn=retrieval_fn,
        k=k,
        candidate_pool_size=candidate_pool_size,
        use_generative=use_generative,
        prefilter_topk=prefilter_topk,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_text_features=clip_text_features,
        device=device,
        return_predictions=return_predictions,
        progress_desc="Querying Idefics2",
        candidate_batch_size=candidate_batch_size,
        cache_path=None,  # Single-GPU uses external caching
        all_classes_label_mapping=all_classes_label_mapping
    )

    # Compute final metrics
    accuracy = eval_results['correct'] / eval_results['total'] if eval_results['total'] > 0 else 0.0

    per_class_accuracy = {}
    for label in eval_results['per_class_total']:
        per_class_accuracy[label] = (
            eval_results['per_class_correct'][label] / eval_results['per_class_total'][label]
            if eval_results['per_class_total'][label] > 0 else 0.0
        )

    mean_per_class_accuracy = np.mean(list(per_class_accuracy.values())) if per_class_accuracy else 0.0

    results = {
        'accuracy': accuracy,
        'mean_per_class_accuracy': mean_per_class_accuracy,
        'correct': eval_results['correct'],
        'total': eval_results['total'],
        'per_class_accuracy': per_class_accuracy
    }

    if return_predictions:
        results['predictions'] = eval_results['predictions']

    return results


def _count_gpus_without_cuda_init() -> int:
    """Count available GPUs using nvidia-smi, avoiding CUDA runtime initialization.

    Calling torch.cuda.is_available() or device_count() in the parent process
    initializes CUDA, which is inherited by forked children and prevents
    CUDA_VISIBLE_DEVICES from taking effect in those children.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _setup_device(num_gpus_requested: Optional[int] = None):
    """Detect device and GPU count without initialising the CUDA runtime.

    Returns (device, num_gpus, use_multi_gpu).  Uses nvidia-smi so that
    CUDA_VISIBLE_DEVICES set on worker subprocesses still takes effect.
    """
    num_available = _count_gpus_without_cuda_init()
    if num_available > 0:
        device = "cuda"
        num_gpus = min(
            num_gpus_requested if num_gpus_requested is not None else num_available,
            num_available,
        )
        print(f"Using device: {device}, GPUs: {num_gpus}/{num_available}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        num_gpus = 1
        print(f"Using device: {device}")
    else:
        device = "cpu"
        num_gpus = 1
        print(f"Using device: {device}")
    return device, num_gpus, num_gpus > 1 and device == "cuda"


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL performance with a learned reranker")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["stanford_cars", "mini_imagenet"] + list(FINE_GRAINED_DATASETS.keys()),
                        help="Dataset to evaluate on")
    parser.add_argument("--eval-split", type=str, default="test",
                        choices=["train", "val", "test", "val+test"],
                        help="Which split(s) to evaluate on (default: test)")
    parser.add_argument("--retrieval-split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Which split to retrieve ICL examples from (default: auto)")
    parser.add_argument("--reranker-checkpoint", type=str, required=True,
                        help="Path to trained reranker checkpoint (local path or filename if using --kaggle-dataset)")
    parser.add_argument("--kaggle-dataset", type=str, default=None,
                        help="Kaggle dataset containing checkpoint (username/dataset-name)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download from Kaggle even if cached")
    parser.add_argument("--image-split-path", type=str, default=None,
                        help="Path to image-level split JSON (for within-distribution Stanford Cars eval)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of in-context examples to include in prompt")
    parser.add_argument("--candidate-pool-size", type=int, default=50,
                        help="Number of candidates to retrieve and rerank (default: 50)")
    parser.add_argument("--num-queries", type=int, default=None,
                        help="Number of test queries to evaluate (default: all)")
    parser.add_argument("--llava-model", type=str, default="HuggingFaceM4/idefics2-8b",
                        help="Idefics2 model to use")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load Idefics2 in 8-bit mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path to save results (.pkl)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect all available)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached predictions if available")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recompute even if cache exists")
    parser.add_argument("--use-generative", action="store_true",
                        help="Use generative evaluation instead of discriminative (probability-based)")
    parser.add_argument("--prefilter-topk", type=int, default=None,
                        help="For generative evaluation: CLIP pre-filter to top-K candidates")
    parser.add_argument("--candidate-batch-size", type=int, default=8,
                        help="Number of candidate labels to process in parallel (default: 8)")
    parser.add_argument("--use-all-classes", action="store_true",
                        help="Use all dataset classes as candidates (discriminative only)")

    args = parser.parse_args()

    device, num_gpus, use_multi_gpu = _setup_device(args.num_gpus)

    retrieval_split = determine_retrieval_split(args.eval_split, args.retrieval_split)
    eval_mode = "generative" if args.use_generative else "discriminative"
    print(f"\nEvaluation setup:")
    print(f"  Queries from: {args.eval_split}")
    print(f"  ICL candidates from: {retrieval_split}")
    print(f"  Evaluation mode: {eval_mode}")
    if args.image_split_path:
        print(f"  Image split: {args.image_split_path}")

    print(f"\nLoading {args.eval_split} split(s) for evaluation...")
    if args.eval_split == "val+test":
        temp_datasets = [
            load_dataset(args.dataset, split="val", image_split_path=args.image_split_path),
            load_dataset(args.dataset, split="test", image_split_path=args.image_split_path),
        ]
        test_dataset_size = sum(len(ds) for ds in temp_datasets)
        print(f"Combined val+test dataset size: {test_dataset_size} examples")
    else:
        temp_datasets = [load_dataset(args.dataset, split=args.eval_split, image_split_path=args.image_split_path)]
        test_dataset_size = len(temp_datasets[0])
        print(f"{args.eval_split} split size: {test_dataset_size} examples")

    if use_multi_gpu:
        del temp_datasets
        test_dataset = None
    else:
        if len(temp_datasets) > 1:
            test_dataset = CombinedDataset(temp_datasets)
            print(f"Combined dataset: {len(test_dataset)} examples, "
                  f"{len(set(ex.label for ex in test_dataset.examples))} classes")
        else:
            test_dataset = temp_datasets[0]
            print(f"{args.eval_split} split: {len(test_dataset)} examples, "
                  f"{len(set(ex.label for ex in test_dataset.examples))} classes")
        retrieval_dataset = test_dataset

        all_classes_label_mapping = None
        if args.use_all_classes and not args.use_generative:
            print("\nBuilding complete label mapping from all splits...")
            all_splits_datasets = [
                load_dataset(args.dataset, split=s, image_split_path=args.image_split_path)
                for s in ("train", "val", "test")
            ]
            all_classes_label_mapping = {}
            for ds in all_splits_datasets:
                for ex in ds.examples:
                    if ex.label_name not in all_classes_label_mapping:
                        all_classes_label_mapping[ex.label_name] = ex.label
            print(f"✓ Using {len(all_classes_label_mapping)} classes as candidates")

    # Reranker evaluation
    reranker_cache_path = get_cache_path(
        dataset_name=args.dataset,
        method="reranker",
        k=args.k,
        num_queries=args.num_queries or test_dataset_size,
        seed=args.seed,
        reranker_checkpoint=args.reranker_checkpoint,
        use_generative=args.use_generative,
        prefilter_topk=args.prefilter_topk,
        use_all_classes=args.use_all_classes,
        candidate_pool_size=args.candidate_pool_size,
    )

    reranker_results = None
    if args.use_cache and not args.force_recompute:
        reranker_results = load_cached_results(reranker_cache_path)

    if reranker_results is None:
        print("\n" + "="*70)
        print("EVALUATING: Learned Reranker")
        print("="*70)

        if use_multi_gpu:
            print(f"MULTI-GPU MODE: {num_gpus} GPUs")
            reranker_results = evaluate_icl_multigpu(
                dataset_name=args.dataset,
                test_dataset_size=test_dataset_size,
                eval_split=args.eval_split,
                retrieval_split=retrieval_split,
                reranker_checkpoint=args.reranker_checkpoint,
                kaggle_dataset=args.kaggle_dataset,
                llava_model_name=args.llava_model,
                load_in_8bit=args.load_in_8bit,
                k=args.k,
                candidate_pool_size=args.candidate_pool_size,
                num_queries=args.num_queries or test_dataset_size,
                seed=args.seed,
                return_predictions=True,
                use_reranker=True,
                num_gpus=num_gpus,
                use_generative=args.use_generative,
                prefilter_topk=args.prefilter_topk,
                candidate_batch_size=args.candidate_batch_size,
                cache_path=reranker_cache_path,
                use_all_classes=args.use_all_classes,
                image_split_path=args.image_split_path,
                force_recompute=args.force_recompute,
            )
        else:
            reranker, interaction_features = load_reranker(
                checkpoint_path=args.reranker_checkpoint,
                device=device,
                kaggle_dataset=args.kaggle_dataset,
                force_refresh=args.force_refresh
            )

            print(f"\nInitializing Idefics2: {args.llava_model}")
            llava_model = Idefics2Wrapper(
                model_name=args.llava_model,
                device=device,
                load_in_8bit=args.load_in_8bit
            )
            print("✓ Idefics2 model loaded")

            is_patch_model = isinstance(reranker, PatchCrossAttentionReranker)
            candidate_patch_features = None
            if is_patch_model:
                candidate_patch_features = extract_patch_features_for_dataset(
                    retrieval_dataset, reranker, device, batch_size=64
                )

            def reranker_retrieval_fn(query_emb, retr_ds, k, exclude_indices=None, query_image=None):
                if is_patch_model:
                    return retrieve_by_patch_reranker(
                        query_image=query_image,
                        query_emb=query_emb,
                        train_dataset=retr_ds,
                        reranker=reranker,
                        device=device,
                        k=k,
                        exclude_indices=exclude_indices,
                        prefilter_n=args.candidate_pool_size,
                        candidate_patch_features=candidate_patch_features
                    )
                else:
                    return retrieve_by_reranker(
                        query_emb, retr_ds, reranker, interaction_features, device, k, exclude_indices
                    )

            reranker_results = evaluate_icl(
                test_dataset=test_dataset,
                retrieval_dataset=retrieval_dataset,
                llava_model=llava_model,
                retrieval_fn=reranker_retrieval_fn,
                k=args.k,
                num_queries=args.num_queries,
                seed=args.seed,
                return_predictions=True,
                use_generative=args.use_generative,
                prefilter_topk=args.prefilter_topk,
                device=device,
                candidate_batch_size=args.candidate_batch_size,
                all_classes_label_mapping=all_classes_label_mapping,
                candidate_pool_size=args.candidate_pool_size
            )

        if args.use_cache:
            save_cached_results(reranker_cache_path, reranker_results)

    print(f"\nReranker Results:")
    print(f"  Accuracy:              {reranker_results['accuracy']:.2%}")
    print(f"  Mean per-class acc:    {reranker_results['mean_per_class_accuracy']:.2%}")
    print(f"  Correct:               {reranker_results['correct']}/{reranker_results['total']}")

    ckpt_stem = Path(args.reranker_checkpoint).stem
    save_eval_results(
        method=f"reranker_{ckpt_stem}",
        results=reranker_results,
        run_id_parts={
            "dataset": args.dataset,
            "k": args.k,
            "pool": args.candidate_pool_size,
            "n": args.num_queries or test_dataset_size,
            "seed": args.seed,
            "mode": eval_mode,
        },
        args=vars(args),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({'results': reranker_results, 'args': vars(args)}, f)
        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
