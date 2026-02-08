"""
Training script for CLIP-based reranker model.

Trains an MLP to predict marginal utility from CLIP embeddings,
using pre-computed marginal utilities as training labels.

Usage:
    python scripts/train_reranker.py
    python scripts/train_reranker.py training.batch_size=128 training.learning_rate=1e-3
"""

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path to enable src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.marginal_utility_dataset import InteractionFeaturesConfig, MarginalUtilityDataset
from src.models.reranker import CLIPReranker
from src.utils.kaggle_utils import is_kaggle_environment, resolve_data_paths


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: DictConfig) -> str:
    """Determine device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and cfg.get("use_mps", True):
        return "mps"
    else:
        return "cpu"


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    interaction_features: InteractionFeaturesConfig,
    grad_clip: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Unpack batch (handles variable length due to interaction features)
        # Always: query_emb, example_emb, similarity, ..., utility (last)
        *features, utility = batch

        # Move to device
        features = [f.to(device) for f in features]
        utility = utility.to(device)

        # Unpack features (at least 3: query_emb, example_emb, similarity)
        query_emb, example_emb, similarity = features[:3]
        interaction_feats = features[3:] if len(features) > 3 else []

        # Map interaction features to correct keyword arguments
        # Order must match dataset output order: product, difference, l2_distance
        product = None
        difference = None
        l2_distance = None

        feat_idx = 0
        if interaction_features.use_product and feat_idx < len(interaction_feats):
            product = interaction_feats[feat_idx]
            feat_idx += 1
        if interaction_features.use_difference and feat_idx < len(interaction_feats):
            difference = interaction_feats[feat_idx]
            feat_idx += 1
        if interaction_features.use_l2_distance and feat_idx < len(interaction_feats):
            l2_distance = interaction_feats[feat_idx]
            feat_idx += 1

        # Forward pass with properly mapped keyword arguments
        pred_utility = model(query_emb, example_emb, similarity,
                            product=product, difference=difference, l2_distance=l2_distance)

        # Compute loss
        loss = criterion(pred_utility, utility)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    interaction_features: InteractionFeaturesConfig
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        # Unpack batch (handles variable length due to interaction features)
        *features, utility = batch

        # Move to device
        features = [f.to(device) for f in features]
        utility = utility.to(device)

        # Unpack features
        query_emb, example_emb, similarity = features[:3]
        interaction_feats = features[3:] if len(features) > 3 else []

        # Map interaction features to correct keyword arguments
        product = None
        difference = None
        l2_distance = None

        feat_idx = 0
        if interaction_features.use_product and feat_idx < len(interaction_feats):
            product = interaction_feats[feat_idx]
            feat_idx += 1
        if interaction_features.use_difference and feat_idx < len(interaction_feats):
            difference = interaction_feats[feat_idx]
            feat_idx += 1
        if interaction_features.use_l2_distance and feat_idx < len(interaction_feats):
            l2_distance = interaction_feats[feat_idx]
            feat_idx += 1

        # Forward pass with properly mapped keyword arguments
        pred_utility = model(query_emb, example_emb, similarity,
                            product=product, difference=difference, l2_distance=l2_distance)

        # Compute loss
        loss = criterion(pred_utility, utility)
        total_loss += loss.item()
        num_batches += 1

        # Store predictions and targets
        all_predictions.extend(pred_utility.cpu().numpy().flatten())
        all_targets.extend(utility.cpu().numpy().flatten())

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))

    # R² score
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman correlation
    spearman_corr, _ = spearmanr(all_predictions, all_targets)

    metrics = {
        'loss': total_loss / num_batches,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman_corr
    }

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg: DictConfig,
    filepath: Path
):
    """Save model checkpoint."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': OmegaConf.to_container(cfg, resolve=True)
    }

    torch.save(checkpoint, filepath)


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_mses: list,
    val_maes: list,
    val_r2s: list,
    val_spearmans: list,
    learning_rates: list,
    save_dir: Path,
    cfg: DictConfig
):
    """Plot and save training curves with config annotations."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Create title with experiment info
    title = f"Training Curves: {cfg.experiment.name}"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Add config details as subtitle
    config_text = (
        f"Architecture: {cfg.model.hidden_dims} | "
        f"Dropout: {cfg.model.dropout} | "
        f"LR: {cfg.training.learning_rate} | "
        f"Batch: {cfg.training.batch_size} | "
        f"Seed: {cfg.experiment.seed}"
    )
    fig.text(0.5, 0.96, config_text, ha='center', fontsize=10, style='italic', color='gray')

    epochs = list(range(1, len(train_losses) + 1))

    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation MSE
    axes[0, 1].plot(epochs, val_mses, label='Val MSE', color='tab:orange', marker='o', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Validation MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    best_epoch = np.argmin(val_mses) + 1
    axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: Epoch {best_epoch}')
    axes[0, 1].legend()

    # Plot 3: Validation MAE
    axes[0, 2].plot(epochs, val_maes, label='Val MAE', color='tab:green', marker='o', markersize=3)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].set_title('Validation MAE')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Validation R²
    axes[1, 0].plot(epochs, val_r2s, label='Val R²', color='tab:purple', marker='o', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Validation R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Plot 5: Validation Spearman Correlation
    axes[1, 1].plot(epochs, val_spearmans, label='Val Spearman', color='tab:red', marker='o', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Spearman Correlation')
    axes[1, 1].set_title('Validation Spearman Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Learning Rate Schedule
    axes[1, 2].plot(epochs, learning_rates, label='Learning Rate', color='tab:brown', marker='o', markersize=3)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Add detailed config info as text annotation
    best_epoch = np.argmin(val_mses) + 1
    best_mse = min(val_mses)
    best_r2 = val_r2s[best_epoch - 1]
    best_spearman = val_spearmans[best_epoch - 1]

    config_info = (
        f"Configuration:\n"
        f"  Architecture: {cfg.model.hidden_dims}\n"
        f"  Dropout: {cfg.model.dropout}\n"
        f"  Learning Rate: {cfg.training.learning_rate}\n"
        f"  Weight Decay: {cfg.training.weight_decay}\n"
        f"  Batch Size: {cfg.training.batch_size}\n"
        f"  Seed: {cfg.experiment.seed}\n\n"
        f"Best Results (Epoch {best_epoch}):\n"
        f"  Val MSE: {best_mse:.4f}\n"
        f"  Val R²: {best_r2:.4f}\n"
        f"  Val Spearman: {best_spearman:.4f}\n"
        f"  Total Epochs: {len(epochs)}"
    )

    fig.text(0.99, 0.01, config_info,
             fontsize=8, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=(0, 0.03, 1, 0.98))

    # Save plot
    plot_path = save_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {plot_path}")

    # Also save as PDF for publications
    pdf_path = save_dir / 'training_curves.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')

    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="train_reranker")
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 70)
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")
    print("=" * 70)

    # Set seed
    set_seed(cfg.experiment.seed)

    # Get device
    device = get_device(cfg)
    print(f"\nDevice: {device}")

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

    # Create interaction features config from model config
    print(f"\nReading interaction features from config...")
    print(f"  use_product: {cfg.model.get('use_product', False)}")
    print(f"  use_difference: {cfg.model.get('use_difference', False)}")
    print(f"  use_l2_distance: {cfg.model.get('use_l2_distance', False)}")

    interaction_features = InteractionFeaturesConfig(
        use_product=cfg.model.get('use_product', False),
        use_difference=cfg.model.get('use_difference', False),
        use_l2_distance=cfg.model.get('use_l2_distance', False)
    )
    print(f"Interaction features config: {interaction_features}")
    print(f"Additional features: {interaction_features.num_features}")

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = MarginalUtilityDataset(
        results_path=results_path,
        embeddings_path=embeddings_path,
        split='train',
        seed=cfg.experiment.seed,
        interaction_features=interaction_features
    )

    val_dataset = MarginalUtilityDataset(
        results_path=results_path,
        embeddings_path=embeddings_path,
        split='val',
        seed=cfg.experiment.seed,
        interaction_features=interaction_features
    )

    # Compute baseline
    baseline_mse_train = train_dataset.compute_baseline_mse()
    baseline_mse_val = val_dataset.compute_baseline_mse()
    print(f"\nBaseline MSE (CLIP similarity):")
    print(f"  Train: {baseline_mse_train:.4f}")
    print(f"  Val:   {baseline_mse_val:.4f}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    # Create model
    print(f"\nInitializing model...")
    model = CLIPReranker(
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        interaction_features=interaction_features
    )
    model = model.to(device)
    print(f"  Parameters: {model.get_num_parameters():,}")
    print(f"  Input dimension: {2 * cfg.model.embedding_dim + 1 + interaction_features.num_features}")

    # Loss function
    criterion = nn.HuberLoss(delta=cfg.training.get("huber_delta", 1.0))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
        eta_min=cfg.training.learning_rate * 0.01
    )

    # Training loop
    print(f"\nStarting training for {cfg.training.num_epochs} epochs...")
    print("=" * 70)

    best_val_mse = float('inf')
    patience_counter = 0

    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_mses = []
    val_maes = []
    val_r2s = []
    val_spearmans = []
    learning_rates = []

    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            interaction_features=interaction_features,
            grad_clip=cfg.training.gradient_clip
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, interaction_features)

        # Update scheduler
        scheduler.step()

        # Track metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_mses.append(val_metrics['mse'])
        val_maes.append(val_metrics['mae'])
        val_r2s.append(val_metrics['r2'])
        val_spearmans.append(val_metrics['spearman'])
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val MSE:    {val_metrics['mse']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val R²:     {val_metrics['r2']:.4f}")
        print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0

            if cfg.checkpoint.enabled:
                checkpoint_path = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "best_model.pt"
                save_checkpoint(model, optimizer, epoch, val_metrics, cfg, checkpoint_path)
                print(f"  ✓ Saved best model (MSE: {best_val_mse:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= cfg.training.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        # Regular checkpoint
        if cfg.checkpoint.enabled and (epoch + 1) % cfg.checkpoint.get("save_interval", 10) == 0:
            checkpoint_path = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / f"epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch, val_metrics, cfg, checkpoint_path)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation MSE: {best_val_mse:.4f}")
    print(f"Baseline MSE: {baseline_mse_val:.4f}")
    print(f"Improvement: {(1 - best_val_mse / baseline_mse_val) * 100:.1f}%")
    print("=" * 70)

    # Plot training curves
    if cfg.checkpoint.enabled:
        plot_dir = Path(cfg.checkpoint.save_dir) / cfg.experiment.name
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            val_mses=val_mses,
            val_maes=val_maes,
            val_r2s=val_r2s,
            val_spearmans=val_spearmans,
            learning_rates=learning_rates,
            save_dir=plot_dir,
            cfg=cfg
        )


if __name__ == "__main__":
    main()
