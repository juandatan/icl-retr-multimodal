"""
Evaluation script for trained reranker model.

Loads a trained reranker checkpoint and evaluates it on the test set,
computing regression metrics and comparing to CLIP similarity baseline.

Usage:
    python scripts/evaluate_reranker.py
    python scripts/evaluate_reranker.py checkpoint_path=path/to/model.pt
"""

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.marginal_utility_dataset import MarginalUtilityDataset
from models.reranker import CLIPReranker


def get_device() -> str:
    """Determine device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@torch.no_grad()
def evaluate_model(
    model: CLIPReranker,
    dataloader: DataLoader,
    device: str
) -> dict:
    """Evaluate model and collect predictions."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_similarities = []

    for query_emb, example_emb, similarity, utility in dataloader:
        # Move to device
        query_emb = query_emb.to(device)
        example_emb = example_emb.to(device)
        similarity_tensor = similarity.to(device)

        # Forward pass
        pred_utility = model(query_emb, example_emb, similarity_tensor)

        # Store results
        all_predictions.extend(pred_utility.cpu().numpy().flatten())
        all_targets.extend(utility.numpy().flatten())
        all_similarities.extend(similarity.numpy().flatten())

    # Convert to arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    similarities = np.array(all_similarities)

    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(predictions, targets)

    # Baseline metrics (using CLIP similarity as predictor)
    baseline_mse = np.mean((similarities - targets) ** 2)
    baseline_mae = np.mean(np.abs(similarities - targets))
    baseline_spearman, _ = spearmanr(similarities, targets)

    metrics = {
        'predictions': predictions,
        'targets': targets,
        'similarities': similarities,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval,
        'baseline_mse': baseline_mse,
        'baseline_mae': baseline_mae,
        'baseline_spearman': baseline_spearman
    }

    return metrics


def plot_results(metrics: dict, save_path: Path):
    """Generate evaluation plots."""
    predictions = metrics['predictions']
    targets = metrics['targets']
    similarities = metrics['similarities']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Predicted vs True utilities
    axes[0].scatter(targets, predictions, alpha=0.3, s=1)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True Utility')
    axes[0].set_ylabel('Predicted Utility')
    axes[0].set_title(f'Reranker Predictions\n(R²={metrics["r2"]:.3f}, Spearman={metrics["spearman"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: CLIP similarity vs True utilities (baseline)
    axes[1].scatter(targets, similarities, alpha=0.3, s=1)
    axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('True Utility')
    axes[1].set_ylabel('CLIP Similarity')
    axes[1].set_title(f'CLIP Baseline\n(Spearman={metrics["baseline_spearman"]:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Error distributions
    reranker_errors = np.abs(predictions - targets)
    baseline_errors = np.abs(similarities - targets)

    axes[2].hist(reranker_errors, bins=50, alpha=0.5, label=f'Reranker (MAE={metrics["mae"]:.3f})', density=True)
    axes[2].hist(baseline_errors, bins=50, alpha=0.5, label=f'CLIP (MAE={metrics["baseline_mae"]:.3f})', density=True)
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plots to {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="train_reranker")
def main(cfg: DictConfig):
    """Main evaluation function."""
    print("=" * 70)
    print("Reranker Model Evaluation")
    print("=" * 70)

    # Get checkpoint path
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        # Default: load best model from training
        checkpoint_path = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "best_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Device: {device}")

    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = MarginalUtilityDataset(
        results_path=cfg.data.results_path,
        embeddings_path=cfg.data.embeddings_path,
        split='test',
        seed=cfg.experiment.seed
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0
    )

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

    # Evaluate
    print(f"\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)

    print("\nReranker Model:")
    print(f"  MSE:            {metrics['mse']:.4f}")
    print(f"  MAE:            {metrics['mae']:.4f}")
    print(f"  R²:             {metrics['r2']:.4f}")
    print(f"  Spearman:       {metrics['spearman']:.4f} (p={metrics['spearman_pval']:.2e})")

    print("\nCLIP Similarity Baseline:")
    print(f"  MSE:            {metrics['baseline_mse']:.4f}")
    print(f"  MAE:            {metrics['baseline_mae']:.4f}")
    print(f"  Spearman:       {metrics['baseline_spearman']:.4f}")

    print("\nImprovement over Baseline:")
    mse_improvement = (1 - metrics['mse'] / metrics['baseline_mse']) * 100
    mae_improvement = (1 - metrics['mae'] / metrics['baseline_mae']) * 100
    print(f"  MSE reduction:  {mse_improvement:+.1f}%")
    print(f"  MAE reduction:  {mae_improvement:+.1f}%")

    print("=" * 70)

    # Generate plots
    plot_path = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "evaluation_plots.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(metrics, plot_path)


if __name__ == "__main__":
    main()
