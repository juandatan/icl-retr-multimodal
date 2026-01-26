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
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path to enable src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.marginal_utility_dataset import MarginalUtilityDataset
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
    grad_clip: float = 1.0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for query_emb, example_emb, similarity, utility in tqdm(dataloader, desc="Training", leave=False):
        # Move to device
        query_emb = query_emb.to(device)
        example_emb = example_emb.to(device)
        similarity = similarity.to(device)
        utility = utility.to(device)

        # Forward pass
        pred_utility = model(query_emb, example_emb, similarity)

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
    device: str
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    for query_emb, example_emb, similarity, utility in tqdm(dataloader, desc="Evaluating", leave=False):
        # Move to device
        query_emb = query_emb.to(device)
        example_emb = example_emb.to(device)
        similarity = similarity.to(device)
        utility = utility.to(device)

        # Forward pass
        pred_utility = model(query_emb, example_emb, similarity)

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

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = MarginalUtilityDataset(
        results_path=results_path,
        embeddings_path=embeddings_path,
        split='train',
        seed=cfg.experiment.seed
    )

    val_dataset = MarginalUtilityDataset(
        results_path=results_path,
        embeddings_path=embeddings_path,
        split='val',
        seed=cfg.experiment.seed
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
        dropout=cfg.model.dropout
    )
    model = model.to(device)
    print(f"  Parameters: {model.get_num_parameters():,}")

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

    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=cfg.training.gradient_clip
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

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


if __name__ == "__main__":
    main()
