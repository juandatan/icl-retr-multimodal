"""
Evaluate In-Context Learning (ICL) performance using different retrieval methods.

This script compares:
1. CLIP similarity baseline: retrieve top-K examples by cosine similarity
2. Learned reranker: retrieve top-K examples by predicted marginal utility

For each method, we:
- Select K in-context examples for each test query
- Query LLaVA for classification
- Measure accuracy

Usage:
    # Mini-ImageNet with k=1
    python scripts/evaluate_icl_performance.py \
        --dataset mini_imagenet \
        --reranker-checkpoint outputs/reranker_checkpoints/reranker_mini_imagenet_v2/best_model.pt \
        --k 1 \
        --num-queries 100

    # Stanford Cars with k=1
    python scripts/evaluate_icl_performance.py \
        --dataset stanford_cars \
        --reranker-checkpoint outputs/reranker_checkpoints/reranker_stanford_cars/best_model.pt \
        --k 1 \
        --num-queries 100
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
from data.marginal_utility_dataset import InteractionFeaturesConfig
from data.base_dataset import ClassificationExample
from models.mlp_reranker import MLPReranker
from models.llava_wrapper import LLaVAWrapper
from utils.multigpu_utils import MultiGPUManager, merge_dict_results
from utils.imagenet_names import get_readable_name, get_synset_id, IMAGENET_SYNSET_TO_NAME


def get_cache_path(
    dataset_name: str,
    method: str,
    k: int,
    num_queries: int,
    seed: int,
    reranker_checkpoint: str = None,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None
) -> Path:
    """Generate cache path for evaluation results."""
    cache_dir = Path("outputs/icl_evaluation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique identifier for this evaluation configuration
    config_id = f"{dataset_name}_{method}_k{k}_n{num_queries}_seed{seed}"

    if reranker_checkpoint and method == "reranker":
        # Add checkpoint name to cache key
        ckpt_name = Path(reranker_checkpoint).stem
        config_id += f"_{ckpt_name}"

    # Add evaluation method to cache key
    if use_generative:
        config_id += "_generative"

    # Add prefilter setting to cache key
    if prefilter_topk is not None:
        config_id += f"_prefilter{prefilter_topk}"

    return cache_dir / f"{config_id}.pkl"


def load_cached_results(cache_path: Path) -> Optional[Dict]:
    """Load cached evaluation results if they exist."""
    if cache_path.exists():
        print(f"Found cached results at {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        print(f"✓ Loaded {cached['total']} cached predictions")
        return cached
    return None


def save_cached_results(cache_path: Path, results: Dict):
    """Save evaluation results to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Cached results saved to {cache_path}")


def load_dataset(dataset_name: str, split: str = "test"):
    """Load dataset for evaluation."""
    print(f"\nLoading {dataset_name} dataset ({split} split)...")

    if dataset_name == "stanford_cars":
        dataset = StanfordCarsDataset(
            split=split,
            data_dir="data/stanford_cars",
            class_split_seed=42
        )
    elif dataset_name == "mini_imagenet":
        dataset = MiniImageNetDataset(
            split=split,
            data_dir="data/mini_imagenet",
            class_split_seed=42
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load CLIP embeddings
    success = dataset.load_clip_embeddings()
    if not success:
        raise FileNotFoundError(
            f"CLIP embeddings not found. Please run: "
            f"python scripts/build_clip_embeddings.py --dataset {dataset_name} --splits {split}"
        )

    print(f"✓ Loaded {len(dataset)} examples")
    print(f"✓ Embeddings shape: {dataset.clip_embeddings.shape}")
    print(f"✓ Num classes: {dataset.num_classes}")

    return dataset


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


def load_reranker(checkpoint_path: str, device: str, kaggle_dataset: str = None, force_refresh: bool = False) -> MLPReranker:
    """
    Load trained reranker model from checkpoint.

    Args:
        checkpoint_path: Local path or filename of checkpoint
        device: Device to load model on
        kaggle_dataset: Optional Kaggle dataset (username/dataset-name) to download from
        force_refresh: If True, re-download from Kaggle even if cached

    Returns:
        Tuple of (model, interaction_features)
    """
    # Handle Kaggle dataset download
    if kaggle_dataset:
        cache_dir = "./cache/checkpoints"
        cache_path = Path(cache_dir) / kaggle_dataset.replace('/', '_') / Path(checkpoint_path).name

        if force_refresh and cache_path.exists():
            print(f"Force refresh: removing cached file {cache_path}")
            cache_path.unlink()

        checkpoint_path = download_from_kaggle_dataset(
            kaggle_dataset=kaggle_dataset,
            filename=Path(checkpoint_path).name,
            cache_dir=cache_dir
        )

    checkpoint_path = Path(checkpoint_path)
    print(f"\nLoading reranker from {checkpoint_path}...")

    # Load checkpoint (weights_only=False needed for full checkpoint with config/optimizer state)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    config = checkpoint['config']
    model_config = config['model']

    # Create interaction features config
    interaction_features = InteractionFeaturesConfig(
        use_product=model_config.get('use_product', False),
        use_difference=model_config.get('use_difference', False),
        use_l2_distance=model_config.get('use_l2_distance', False)
    )

    # Initialize model
    model = MLPReranker(
        embedding_dim=model_config['embedding_dim'],
        hidden_dims=model_config['hidden_dims'],
        dropout=model_config.get('dropout', 0.1),
        interaction_features=interaction_features,
        use_sigmoid=model_config.get('use_sigmoid', False)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model with {model.get_num_parameters():,} parameters")
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

    # Get top-K
    top_k_indices = np.argsort(similarities)[-k:][::-1]

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

    # Get top-K
    top_k_indices = np.argsort(utilities)[-k:][::-1]

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
    llava_model: LLaVAWrapper,
    retrieval_fn,
    k: int,
    use_generative: bool,
    prefilter_topk: Optional[int],
    clip_model,
    clip_preprocess,
    clip_text_features,
    device: str,
    return_predictions: bool = False,
    progress_desc: str = "Evaluating"
) -> Dict:
    """
    Shared evaluation logic for both single-GPU and multi-GPU modes.

    Args:
        query_indices: List of query indices to evaluate
        test_dataset: Test dataset
        retrieval_dataset: Dataset to retrieve ICL examples from
        llava_model: LLaVA model for classification
        retrieval_fn: Function(query_emb, dataset, k) -> List[indices]
        k: Number of ICL examples
        use_generative: Whether to use generative evaluation
        prefilter_topk: Number of candidates for CLIP pre-filtering (or None)
        clip_model: CLIP model for pre-filtering (or None)
        clip_preprocess: CLIP preprocessing (or None)
        clip_text_features: Precomputed CLIP text features (or None)
        device: Device string
        return_predictions: Whether to return detailed predictions
        progress_desc: Description for progress bar

    Returns:
        Dictionary with evaluation results
    """
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    predictions = []

    # Build label_name -> label mapping for O(1) lookups
    label_name_to_label = {ex.label_name: ex.label for ex in test_dataset.examples}

    # Precompute candidate labels for discriminative evaluation
    discriminative_candidate_labels = None
    if not use_generative:
        discriminative_candidate_labels = [get_readable_name(ex.label_name) for ex in test_dataset.examples]
        seen = set()
        discriminative_candidate_labels = [x for x in discriminative_candidate_labels if not (x in seen or seen.add(x))]

    # Precompute all class names for generative evaluation
    all_class_names_sorted = sorted([name for name in IMAGENET_SYNSET_TO_NAME.values()]) if use_generative else []

    for query_idx in tqdm(query_indices, desc=progress_desc):
        query_example, query_image = test_dataset[query_idx]
        true_label = query_example.label

        # Retrieve k examples (only if k > 0)
        context_examples = []
        example_indices = []
        if k > 0:
            query_emb = test_dataset.clip_embeddings[query_idx]
            # Exclude the query itself from retrieval
            example_indices = retrieval_fn(query_emb, retrieval_dataset, k, exclude_indices=[query_idx])

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

        # Query LLaVA
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
                candidate_labels=candidate_label_names
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

    return {
        'correct': correct,
        'total': total,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'predictions': predictions if return_predictions else []
    }


def evaluate_icl_worker(
    gpu_id: int,
    query_indices: List[int],
    dataset_name: str,
    eval_split: str = "val+test",
    reranker_checkpoint: str = None,
    kaggle_dataset: str = None,
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    load_in_8bit: bool = False,
    k: int = 1,
    seed: int = 42,
    return_predictions: bool = False,
    use_reranker: bool = False,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None
) -> Dict:
    """
    Worker function for multi-GPU evaluation.
    Runs on a single GPU and evaluates a subset of queries.
    """
    # Set device for this worker
    device = f"cuda:{gpu_id}"

    # Load datasets based on eval_split parameter
    print(f"GPU {gpu_id}: Loading {eval_split} split(s)...")

    if eval_split == "val+test":
        datasets_to_combine = [
            load_dataset(dataset_name, split="val"),
            load_dataset(dataset_name, split="test")
        ]
    else:
        datasets_to_combine = [load_dataset(dataset_name, split=eval_split)]

    # Combine datasets if multiple
    if len(datasets_to_combine) > 1:
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

        test_dataset = CombinedDataset(datasets_to_combine)
        print(f"GPU {gpu_id}: Combined dataset has {len(test_dataset)} examples from {len(set(ex.label for ex in test_dataset.examples))} classes")
    else:
        # Single split, no need to combine
        test_dataset = datasets_to_combine[0]
        print(f"GPU {gpu_id}: Using {eval_split} split with {len(test_dataset)} examples from {len(set(ex.label for ex in test_dataset.examples))} classes")

    # For retrieval, use the same dataset
    retrieval_dataset = test_dataset

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

    # Initialize LLaVA
    llava_model = LLaVAWrapper(
        model_name=llava_model_name,
        device=device,
        load_in_8bit=load_in_8bit
    )

    # Initialize CLIP for oracle pre-filtering if needed
    clip_model = None
    clip_preprocess = None
    clip_text_features = None
    if prefilter_topk is not None and use_generative:
        print(f"GPU {gpu_id}: Loading CLIP for oracle pre-filtering (top-{prefilter_topk})...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

        # Precompute text embeddings for all classes
        all_class_names = sorted([name for name in IMAGENET_SYNSET_TO_NAME.values()])
        text_tokens = clip.tokenize(all_class_names).to(device)
        with torch.no_grad():
            clip_text_features = clip_model.encode_text(text_tokens)
            clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

    # Create retrieval function
    def retrieval_fn(query_emb, dataset, k_examples, exclude_indices=None):
        if use_reranker:
            return retrieve_by_reranker(query_emb, dataset, reranker, interaction_features, device, k_examples, exclude_indices)
        else:
            return retrieve_by_clip(query_emb, dataset, k_examples, exclude_indices)

    print(f"GPU {gpu_id}: Evaluating {len(query_indices)} queries...")

    # Use shared evaluation logic
    return _evaluate_queries(
        query_indices=query_indices,
        test_dataset=test_dataset,
        retrieval_dataset=retrieval_dataset,
        llava_model=llava_model,
        retrieval_fn=retrieval_fn,
        k=k,
        use_generative=use_generative,
        prefilter_topk=prefilter_topk,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_text_features=clip_text_features,
        device=device,
        return_predictions=return_predictions,
        progress_desc=f"GPU {gpu_id}"
    )


def evaluate_icl_multigpu(
    dataset_name: str,
    test_dataset_size: int,
    eval_split: str = "val+test",
    reranker_checkpoint: Optional[str] = None,
    kaggle_dataset: Optional[str] = None,
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    load_in_8bit: bool = False,
    k: int = 1,
    num_queries: Optional[int] = None,
    seed: int = 42,
    return_predictions: bool = False,
    use_reranker: bool = False,
    num_gpus: int = 1,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None
) -> Dict:
    """
    Evaluate ICL performance using multiple GPUs in parallel.

    Note: Workers load their own dataset copies to avoid memory duplication in main process.
    """
    random.seed(seed)

    # Sample queries
    query_indices = list(range(test_dataset_size))
    total_queries = num_queries if num_queries is not None else test_dataset_size
    if total_queries < test_dataset_size:
        query_indices = random.sample(query_indices, total_queries)

    # Use MultiGPUManager to handle parallel execution
    with MultiGPUManager(num_gpus=num_gpus, verbose=True) as mgr:
        # Run evaluation in parallel
        results = mgr.run_parallel(
            worker_fn=evaluate_icl_worker,
            work_items=query_indices,
            worker_kwargs={
                'dataset_name': dataset_name,
                'eval_split': eval_split,
                'reranker_checkpoint': reranker_checkpoint,
                'kaggle_dataset': kaggle_dataset,
                'llava_model_name': llava_model_name,
                'load_in_8bit': load_in_8bit,
                'k': k,
                'seed': seed,
                'return_predictions': return_predictions,
                'use_reranker': use_reranker,
                'use_generative': use_generative,
                'prefilter_topk': prefilter_topk
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
    llava_model: LLaVAWrapper,
    retrieval_fn,
    k: int,
    num_queries: int = None,
    seed: int = 42,
    return_predictions: bool = False,
    use_generative: bool = False,
    prefilter_topk: Optional[int] = None,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate ICL performance using a given retrieval method.

    Args:
        test_dataset: Test examples to query on
        retrieval_dataset: Dataset to retrieve ICL examples from
        llava_model: LLaVA model for classification
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
        use_generative=use_generative,
        prefilter_topk=prefilter_topk,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_text_features=clip_text_features,
        device=device,
        return_predictions=return_predictions,
        progress_desc="Querying LLaVA"
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL performance with different retrieval methods")
    parser.add_argument("--dataset", type=str, required=True, choices=["stanford_cars", "mini_imagenet"],
                        help="Dataset to evaluate on")
    parser.add_argument("--eval-split", type=str, default="val+test",
                        choices=["train", "val", "test", "val+test"],
                        help="Which split(s) to evaluate on (default: val+test for 20 classes)")
    parser.add_argument("--reranker-checkpoint", type=str, default=None,
                        help="Path to trained reranker checkpoint (local path or filename if using --kaggle-dataset)")
    parser.add_argument("--kaggle-dataset", type=str, default=None,
                        help="Kaggle dataset containing checkpoint (username/dataset-name)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-download from Kaggle even if cached")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of in-context examples")
    parser.add_argument("--num-queries", type=int, default=None,
                        help="Number of test queries to evaluate (default: all)")
    parser.add_argument("--llava-model", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model to use")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load LLaVA in 8-bit mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path to save results")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect all available)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached predictions if available")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recompute even if cache exists")
    parser.add_argument("--use-generative", action="store_true",
                        help="Use generative evaluation (free-form generation + matching) instead of discriminative (probability-based)")
    parser.add_argument("--prefilter-topk", type=int, default=None,
                        help="For generative evaluation: Use CLIP to pre-filter to top-K candidates (with oracle guarantee that true label is included)")

    args = parser.parse_args()

    # Set device and detect GPUs
    if torch.cuda.is_available():
        device = "cuda"
        num_available_gpus = torch.cuda.device_count()
        num_gpus = args.num_gpus if args.num_gpus is not None else num_available_gpus
        num_gpus = min(num_gpus, num_available_gpus)  # Cap at available
        print(f"Using device: {device}")
        print(f"Available GPUs: {num_available_gpus}")
        print(f"Using GPUs: {num_gpus}")
    elif torch.backends.mps.is_available():
        device = "mps"
        num_gpus = 1
        print(f"Using device: {device}")
    else:
        device = "cpu"
        num_gpus = 1
        print(f"Using device: {device}")

    # Determine if we should use multi-GPU
    use_multi_gpu = num_gpus > 1 and device == "cuda"

    # Print evaluation mode
    eval_mode = "Generative" if args.use_generative else "Discriminative (probability-based)"
    print(f"\nEvaluation mode: {eval_mode}")

    # Get dataset size for num_queries calculation
    # Load the specified evaluation split(s)
    print(f"\nLoading {args.eval_split} split(s) for evaluation...")

    if args.eval_split == "val+test":
        temp_datasets = [
            load_dataset(args.dataset, split="val"),
            load_dataset(args.dataset, split="test")
        ]
        test_dataset_size = sum(len(ds) for ds in temp_datasets)
        print(f"Combined val+test dataset size: {test_dataset_size} examples")
    else:
        temp_datasets = [load_dataset(args.dataset, split=args.eval_split)]
        test_dataset_size = len(temp_datasets[0])
        print(f"{args.eval_split} split size: {test_dataset_size} examples")

    # For multi-GPU, free the datasets immediately to save memory
    # Workers will load their own copies
    if use_multi_gpu:
        del temp_datasets
        test_dataset = None
    else:
        # For single-GPU, combine and keep the loaded datasets
        if len(temp_datasets) > 1:
            print(f"\nCombining {args.eval_split} splits for single-GPU evaluation...")

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

            test_dataset = CombinedDataset(temp_datasets)
            print(f"Combined dataset has {len(test_dataset)} examples from {len(set(ex.label for ex in test_dataset.examples))} classes")
        else:
            # Single split, no need to combine
            test_dataset = temp_datasets[0]
            print(f"Using {args.eval_split} split with {len(test_dataset)} examples from {len(set(ex.label for ex in test_dataset.examples))} classes")

        # For retrieval, use the same dataset
        retrieval_dataset = test_dataset

    # Check cache for CLIP results
    clip_cache_path = get_cache_path(
        dataset_name=args.dataset,
        method="clip",
        k=args.k,
        num_queries=args.num_queries or test_dataset_size,
        seed=args.seed,
        use_generative=args.use_generative,
        prefilter_topk=args.prefilter_topk
    )

    clip_results = None
    if args.use_cache and not args.force_recompute:
        clip_results = load_cached_results(clip_cache_path)

    if clip_results is None:
        if use_multi_gpu:
            print(f"\n{'='*70}")
            print(f"MULTI-GPU MODE: Using {num_gpus} GPUs in parallel")
            print(f"{'='*70}")

            # Evaluate CLIP similarity baseline
            print("\n" + "="*70)
            print("EVALUATING: CLIP Similarity Baseline")
            print("="*70)

            clip_results = evaluate_icl_multigpu(
                dataset_name=args.dataset,
                test_dataset_size=test_dataset_size,
                eval_split=args.eval_split,
                reranker_checkpoint=None,
                kaggle_dataset=None,
                llava_model_name=args.llava_model,
                load_in_8bit=args.load_in_8bit,
                k=args.k,
                num_queries=args.num_queries or test_dataset_size,
                seed=args.seed,
                return_predictions=True,
                use_reranker=False,
                num_gpus=num_gpus,
                use_generative=args.use_generative,
                prefilter_topk=args.prefilter_topk
            )

            # Save to cache if requested
            if args.use_cache:
                save_cached_results(clip_cache_path, clip_results)
        else:
            # Single GPU/CPU mode
            # Load reranker if provided
            reranker = None
            interaction_features = None
            if args.reranker_checkpoint:
                reranker, interaction_features = load_reranker(
                    checkpoint_path=args.reranker_checkpoint,
                    device=device,
                    kaggle_dataset=args.kaggle_dataset,
                    force_refresh=args.force_refresh
                )

            # Initialize LLaVA
            print(f"\nInitializing LLaVA model: {args.llava_model}")
            llava_model = LLaVAWrapper(
                model_name=args.llava_model,
                device=device,
                load_in_8bit=args.load_in_8bit
            )
            print("✓ LLaVA model loaded")

            # Evaluate CLIP similarity baseline
            print("\n" + "="*70)
            print("EVALUATING: CLIP Similarity Baseline")
            print("="*70)

            def clip_retrieval_fn(query_emb, retr_ds, k, exclude_indices=None):
                return retrieve_by_clip(query_emb, retr_ds, k, exclude_indices)

            clip_results = evaluate_icl(
                test_dataset=test_dataset,
                retrieval_dataset=retrieval_dataset,
                llava_model=llava_model,
                retrieval_fn=clip_retrieval_fn,
                k=args.k,
                num_queries=args.num_queries,
                seed=args.seed,
                return_predictions=True,
                use_generative=args.use_generative,
                prefilter_topk=args.prefilter_topk,
                device=device
            )

            # Save to cache if requested
            if args.use_cache:
                save_cached_results(clip_cache_path, clip_results)

    print(f"\nCLIP Baseline Results:")
    print(f"  Accuracy: {clip_results['accuracy']:.2%}")
    print(f"  Mean per-class accuracy: {clip_results['mean_per_class_accuracy']:.2%}")
    print(f"  Correct: {clip_results['correct']}/{clip_results['total']}")

    # Evaluate reranker if provided
    reranker_results = None
    if args.reranker_checkpoint:
        # Check cache for reranker results
        reranker_cache_path = get_cache_path(
            dataset_name=args.dataset,
            method="reranker",
            k=args.k,
            num_queries=args.num_queries or test_dataset_size,
            seed=args.seed,
            reranker_checkpoint=args.reranker_checkpoint,
            use_generative=args.use_generative,
            prefilter_topk=args.prefilter_topk
        )

        if args.use_cache and not args.force_recompute:
            reranker_results = load_cached_results(reranker_cache_path)

        if reranker_results is None:
            if use_multi_gpu:
                # Multi-GPU mode for reranker
                print("\n" + "="*70)
                print("EVALUATING: Learned Reranker")
                print("="*70)

                reranker_results = evaluate_icl_multigpu(
                    dataset_name=args.dataset,
                    test_dataset_size=test_dataset_size,
                    eval_split=args.eval_split,
                    reranker_checkpoint=args.reranker_checkpoint,
                    kaggle_dataset=args.kaggle_dataset,
                    llava_model_name=args.llava_model,
                    load_in_8bit=args.load_in_8bit,
                    k=args.k,
                    num_queries=args.num_queries or test_dataset_size,
                    seed=args.seed,
                    return_predictions=True,
                    use_reranker=True,
                    num_gpus=num_gpus,
                    use_generative=args.use_generative,
                    prefilter_topk=args.prefilter_topk
                )
            else:
                # Single GPU mode for reranker
                print("\n" + "="*70)
                print("EVALUATING: Learned Reranker")
                print("="*70)

                def reranker_retrieval_fn(query_emb, retr_ds, k, exclude_indices=None):
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
                    device=device
                )

            # Save to cache if requested
            if args.use_cache:
                save_cached_results(reranker_cache_path, reranker_results)

    if reranker_results:
        print(f"\nReranker Results:")
        print(f"  Accuracy: {reranker_results['accuracy']:.2%}")
        print(f"  Mean per-class accuracy: {reranker_results['mean_per_class_accuracy']:.2%}")
        print(f"  Correct: {reranker_results['correct']}/{reranker_results['total']}")

        # Compute improvement
        improvement = reranker_results['accuracy'] - clip_results['accuracy']
        relative_improvement = (improvement / clip_results['accuracy']) * 100 if clip_results['accuracy'] > 0 else 0

        print(f"\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Absolute improvement: {improvement:+.2%}")
        print(f"Relative improvement: {relative_improvement:+.1f}%")

        # Detailed comparison analysis
        print(f"\n" + "="*70)
        print("DETAILED COMPARISON")
        print("="*70)

        clip_preds = clip_results['predictions']
        reranker_preds = reranker_results['predictions']

        # Track comparison categories
        reranker_wins = 0  # Reranker correct, CLIP wrong
        clip_wins = 0      # CLIP correct, reranker wrong
        both_correct = 0   # Both correct
        both_wrong = 0     # Both wrong

        # Track when both correct but selected different examples
        both_correct_diff_examples = []

        for clip_pred, reranker_pred in zip(clip_preds, reranker_preds):
            assert clip_pred['query_idx'] == reranker_pred['query_idx']

            clip_correct = clip_pred['is_correct']
            reranker_correct = reranker_pred['is_correct']

            if reranker_correct and not clip_correct:
                reranker_wins += 1
            elif clip_correct and not reranker_correct:
                clip_wins += 1
            elif clip_correct and reranker_correct:
                both_correct += 1
                # Check if they selected different examples
                if clip_pred['example_indices'] != reranker_pred['example_indices']:
                    both_correct_diff_examples.append({
                        'query_idx': clip_pred['query_idx'],
                        'clip_examples': clip_pred['example_indices'],
                        'reranker_examples': reranker_pred['example_indices']
                    })
            else:
                both_wrong += 1

        total_queries = len(clip_preds)

        print(f"\nOutcome Categories:")
        print(f"  Reranker wins (reranker ✓, CLIP ✗):     {reranker_wins:4d} ({reranker_wins/total_queries:6.2%})")
        print(f"  CLIP wins (CLIP ✓, reranker ✗):         {clip_wins:4d} ({clip_wins/total_queries:6.2%})")
        print(f"  Both correct:                            {both_correct:4d} ({both_correct/total_queries:6.2%})")
        print(f"  Both wrong:                              {both_wrong:4d} ({both_wrong/total_queries:6.2%})")
        print(f"  Total:                                   {total_queries:4d}")

        print(f"\nNet gain from reranker: {reranker_wins - clip_wins:+d} queries")

        print(f"\nWhen both methods are correct:")
        print(f"  Same examples selected:    {both_correct - len(both_correct_diff_examples):4d}")
        print(f"  Different examples:        {len(both_correct_diff_examples):4d}")

        if len(both_correct_diff_examples) > 0:
            print(f"\n  Note: Even when both are correct, reranker selects different")
            print(f"        examples in {len(both_correct_diff_examples)/both_correct:.1%} of cases")

        # Analyze example selection overlap
        if args.k == 1:
            same_examples = sum(
                1 for cp, rp in zip(clip_preds, reranker_preds)
                if cp['example_indices'] == rp['example_indices']
            )
            print(f"\nExample Selection Overlap:")
            print(f"  Same top-1 example selected: {same_examples}/{total_queries} ({same_examples/total_queries:.2%})")
            print(f"  Different top-1 selected:    {total_queries - same_examples}/{total_queries} ({(total_queries - same_examples)/total_queries:.2%})")

        # Store comparison details in results
        reranker_results['comparison'] = {
            'reranker_wins': reranker_wins,
            'clip_wins': clip_wins,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'both_correct_diff_examples': both_correct_diff_examples
        }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'dataset': args.dataset,
            'k': args.k,
            'num_queries': args.num_queries or len(test_dataset),
            'clip_results': clip_results,
            'reranker_results': reranker_results,
            'args': vars(args)
        }

        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
