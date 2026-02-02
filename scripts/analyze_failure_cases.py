"""
Analyze failure cases of the reranker model.

Identifies where the model performs poorly and provides insights into
what characteristics are associated with high prediction errors.

Usage:
    python scripts/analyze_failure_cases.py
    python scripts/analyze_failure_cases.py checkpoint_path=path/to/model.pt
"""

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.marginal_utility_dataset import MarginalUtilityDataset
from src.models.reranker import CLIPReranker
from src.utils.kaggle_utils import is_kaggle_environment, resolve_data_paths


def get_device(cfg: DictConfig) -> str:
    """Determine device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and cfg.get("use_mps", True):
        return "mps"
    else:
        return "cpu"


@torch.no_grad()
def collect_predictions(
    model: CLIPReranker,
    dataset: MarginalUtilityDataset,
    device: str,
    batch_size: int = 256
) -> pd.DataFrame:
    """Collect all predictions with metadata for analysis."""
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_data = []
    batch_idx = 0

    for query_emb, example_emb, similarity, utility in dataloader:
        # Move to device
        query_emb = query_emb.to(device)
        example_emb = example_emb.to(device)
        similarity_tensor = similarity.to(device)

        # Get predictions
        pred_utility = model(query_emb, example_emb, similarity_tensor)

        # Store results
        predictions = pred_utility.cpu().numpy().flatten()
        targets = utility.numpy().flatten()
        similarities = similarity.numpy().flatten()

        # Get indices in original dataset
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(predictions)

        for i, (pred, target, sim) in enumerate(zip(predictions, targets, similarities)):
            dataset_idx = start_idx + i
            result = dataset.results[dataset_idx]

            # Handle both dict and dataclass objects
            def get_attr(obj, key):
                if isinstance(obj, dict):
                    return obj[key]
                else:
                    return getattr(obj, key)

            all_data.append({
                'dataset_idx': dataset_idx,
                'query_idx': get_attr(result, 'query_idx'),
                'example_idx': get_attr(result, 'example_idx'),
                'query_label': get_attr(result, 'query_label'),
                'example_label': get_attr(result, 'example_label'),
                'same_class': get_attr(result, 'same_class'),
                'true_utility': target,
                'pred_utility': pred,
                'clip_similarity': sim,
                'prediction_error': abs(pred - target),
                'baseline_error': abs(sim - target),
                'error_reduction': abs(sim - target) - abs(pred - target)
            })

        batch_idx += 1

    return pd.DataFrame(all_data)


def analyze_failure_cases(df: pd.DataFrame, save_dir: Path):
    """Analyze and visualize failure cases."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Overall statistics
    print("\n" + "=" * 70)
    print("FAILURE CASE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal examples: {len(df)}")
    print(f"Mean prediction error: {df['prediction_error'].mean():.4f}")
    print(f"Mean baseline error: {df['baseline_error'].mean():.4f}")
    print(f"Mean error reduction: {df['error_reduction'].mean():.4f}")

    # Identify worst predictions
    worst_cases = df.nlargest(20, 'prediction_error')
    print(f"\n{'=' * 70}")
    print("TOP 20 WORST PREDICTIONS")
    print("=" * 70)
    print(worst_cases[['query_label', 'example_label', 'same_class',
                       'true_utility', 'pred_utility', 'clip_similarity',
                       'prediction_error']].to_string(index=False))

    # Identify best predictions
    best_improvements = df.nlargest(20, 'error_reduction')
    print(f"\n{'=' * 70}")
    print("TOP 20 BEST IMPROVEMENTS OVER BASELINE")
    print("=" * 70)
    print(best_improvements[['query_label', 'example_label', 'same_class',
                             'true_utility', 'pred_utility', 'clip_similarity',
                             'baseline_error', 'prediction_error', 'error_reduction']].to_string(index=False))

    # Analysis by same_class
    print(f"\n{'=' * 70}")
    print("ANALYSIS BY CLASS MATCH")
    print("=" * 70)
    same_class_stats = df.groupby('same_class').agg({
        'prediction_error': ['mean', 'std', 'min', 'max'],
        'baseline_error': ['mean', 'std'],
        'error_reduction': ['mean', 'std'],
        'true_utility': ['mean', 'std'],
        'clip_similarity': ['mean', 'std']
    })
    print(same_class_stats)

    # Analysis by utility ranges
    print(f"\n{'=' * 70}")
    print("ANALYSIS BY TRUE UTILITY RANGE")
    print("=" * 70)
    df['utility_bin'] = pd.cut(df['true_utility'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    utility_stats = df.groupby('utility_bin').agg({
        'prediction_error': ['mean', 'std', 'count'],
        'baseline_error': ['mean', 'std'],
        'error_reduction': ['mean']
    })
    print(utility_stats)

    # Correlation analysis
    print(f"\n{'=' * 70}")
    print("ERROR CORRELATION ANALYSIS")
    print("=" * 70)

    corr_true_utility = np.corrcoef(df['true_utility'], df['prediction_error'])[0, 1]
    corr_clip_sim = np.corrcoef(df['clip_similarity'], df['prediction_error'])[0, 1]
    corr_baseline_error = np.corrcoef(df['baseline_error'], df['prediction_error'])[0, 1]

    print(f"Correlation between true utility and prediction error: {corr_true_utility:.4f}")
    print(f"Correlation between CLIP similarity and prediction error: {corr_clip_sim:.4f}")
    print(f"Correlation between baseline error and prediction error: {corr_baseline_error:.4f}")

    # Generate visualizations
    create_failure_plots(df, save_dir)

    # Save detailed results to CSV
    csv_path = save_dir / 'failure_analysis_full.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved full analysis to {csv_path}")

    # Save summary statistics
    summary_path = save_dir / 'failure_analysis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FAILURE CASE ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total examples: {len(df)}\n")
        f.write(f"Mean prediction error: {df['prediction_error'].mean():.4f}\n")
        f.write(f"Mean baseline error: {df['baseline_error'].mean():.4f}\n")
        f.write(f"Mean error reduction: {df['error_reduction'].mean():.4f}\n\n")
        f.write("\nBy Class Match:\n")
        f.write(same_class_stats.to_string())
        f.write("\n\nBy Utility Range:\n")
        f.write(utility_stats.to_string())

    print(f"✓ Saved summary to {summary_path}")


def create_failure_plots(df: pd.DataFrame, save_dir: Path):
    """Create visualizations of failure cases."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Error vs True Utility
    axes[0, 0].scatter(df['true_utility'], df['prediction_error'],
                       alpha=0.3, s=5, c=df['same_class'], cmap='RdYlGn')
    axes[0, 0].set_xlabel('True Utility')
    axes[0, 0].set_ylabel('Prediction Error')
    axes[0, 0].set_title('Prediction Error vs True Utility')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Error vs CLIP Similarity
    axes[0, 1].scatter(df['clip_similarity'], df['prediction_error'],
                       alpha=0.3, s=5, c=df['same_class'], cmap='RdYlGn')
    axes[0, 1].set_xlabel('CLIP Similarity')
    axes[0, 1].set_ylabel('Prediction Error')
    axes[0, 1].set_title('Prediction Error vs CLIP Similarity')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Model Error vs Baseline Error
    axes[0, 2].scatter(df['baseline_error'], df['prediction_error'],
                       alpha=0.3, s=5)
    axes[0, 2].plot([0, df['baseline_error'].max()],
                    [0, df['baseline_error'].max()],
                    'r--', label='y=x (no improvement)')
    axes[0, 2].set_xlabel('Baseline Error (CLIP)')
    axes[0, 2].set_ylabel('Model Prediction Error')
    axes[0, 2].set_title('Model Error vs Baseline Error')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Error distribution by class match
    same_class_errors = df[df['same_class'] == True]['prediction_error']
    diff_class_errors = df[df['same_class'] == False]['prediction_error']

    axes[1, 0].hist(same_class_errors, bins=50, alpha=0.5,
                    label=f'Same Class (n={len(same_class_errors)})', density=True)
    axes[1, 0].hist(diff_class_errors, bins=50, alpha=0.5,
                    label=f'Diff Class (n={len(diff_class_errors)})', density=True)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error Distribution by Class Match')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Error by utility bin
    utility_bins = df.groupby('utility_bin')['prediction_error'].apply(list)
    axes[1, 1].boxplot(utility_bins, labels=utility_bins.index)
    axes[1, 1].set_xlabel('True Utility Range')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Error Distribution by Utility Range')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Plot 6: Improvement over baseline
    axes[1, 2].hist(df['error_reduction'], bins=50, edgecolor='black')
    axes[1, 2].axvline(x=0, color='red', linestyle='--',
                       label='No improvement (baseline = model)')
    axes[1, 2].set_xlabel('Error Reduction (Baseline - Model)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Improvement over Baseline Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / 'failure_analysis_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved failure analysis plots to {plot_path}")

    # Also save as PDF
    pdf_path = save_dir / 'failure_analysis_plots.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="train_reranker")
def main(cfg: DictConfig):
    """Main analysis function."""
    print("=" * 70)
    print("Reranker Failure Case Analysis")
    print("=" * 70)

    # Get checkpoint path
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        # Default: load best model from training
        checkpoint_path = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "best_model.pt"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Get device
    device = get_device(cfg)
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  Epoch: {checkpoint['epoch']}")

    # Resolve data paths (handles both local and Kaggle environments)
    print(f"\nResolving data paths...")
    if is_kaggle_environment():
        print("  Running in Kaggle environment")

    results_path = resolve_data_paths(
        local_path=cfg.data.results_path,
        kaggle_path=cfg.data.get('kaggle_results_path', '/kaggle/input/d/juandatan/marginal-utility-training-data/marginal_utilities_train.pkl'),
        dataset_name='juandatan/marginal-utility-training-data',
        required=True
    )

    embeddings_path = resolve_data_paths(
        local_path=cfg.data.embeddings_path,
        kaggle_path=cfg.data.get('kaggle_embeddings_path', '/kaggle/input/d/juandatan/stanford-cars-clip/clip_embeddings_train.pkl'),
        dataset_name='juandatan/stanford-cars-clip',
        required=True
    )

    print(f"  Results: {results_path}")
    print(f"  Embeddings: {embeddings_path}")

    # Load validation dataset (to analyze training performance)
    print(f"\nLoading validation dataset...")
    val_dataset = MarginalUtilityDataset(
        results_path=results_path,
        embeddings_path=embeddings_path,
        split='val',
        seed=cfg.experiment.seed
    )
    print(f"  Validation samples: {len(val_dataset)}")

    # Create and load model
    print(f"\nInitializing model...")
    model = CLIPReranker(
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {model.get_num_parameters():,}")

    # Collect predictions
    print(f"\nCollecting predictions on validation set...")
    df = collect_predictions(model, val_dataset, device, batch_size=cfg.training.batch_size)

    # Analyze failure cases
    save_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "failure_analysis"
    analyze_failure_cases(df, save_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
