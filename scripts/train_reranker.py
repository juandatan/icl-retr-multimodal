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

from src.data.marginal_utility_dataset import (
    InteractionFeaturesConfig,
    MarginalUtilityDataset,
    PairwiseMarginalUtilityDataset,
)
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

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Unpack batch (handles variable length due to interaction features)
        # Always: query_emb, example_emb, similarity, ..., utility (last)
        *features, utility = batch

        # Move to device
        features = [f.to(device) for f in features]
        utility = utility.to(device)

        # Forward pass
        pred_utility = _unpack_and_forward(model, features, interaction_features)

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


def _unpack_and_forward(
    model: nn.Module,
    features: list,
    interaction_features: InteractionFeaturesConfig
) -> torch.Tensor:
    """Helper to unpack features and run forward pass."""
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

    return model(query_emb, example_emb, similarity,
                product=product, difference=difference, l2_distance=l2_distance)


def train_epoch_ranking(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    interaction_features: InteractionFeaturesConfig,
    grad_clip: float = 1.0
) -> float:
    """Train for one epoch using pairwise ranking loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training (Ranking)", leave=False):
        # Move all tensors to device
        batch_tensors = [t.to(device) for t in batch]

        # Calculate features per example: 3 base + num interaction features
        num_base_features = 3
        num_interaction_feats = 0
        if interaction_features.use_product:
            num_interaction_feats += 1
        if interaction_features.use_difference:
            num_interaction_feats += 1
        if interaction_features.use_l2_distance:
            num_interaction_feats += 1
        features_per_example = num_base_features + num_interaction_feats

        # Split into better and worse examples
        better_features = batch_tensors[:features_per_example]
        worse_features = batch_tensors[features_per_example:2 * features_per_example]

        # Forward pass for both examples
        pred_utility_better = _unpack_and_forward(model, better_features, interaction_features)
        pred_utility_worse = _unpack_and_forward(model, worse_features, interaction_features)

        # Compute ranking loss
        # MarginRankingLoss expects (input1, input2, target) where target=1 means input1 > input2
        target = torch.ones_like(pred_utility_better)
        loss = criterion(pred_utility_better, pred_utility_worse, target)

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
    interaction_features: InteractionFeaturesConfig,
    use_bce: bool = False
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    # For ranking loss, use MSE for validation metrics
    use_ranking_loss = isinstance(criterion, nn.MarginRankingLoss)
    eval_criterion = nn.MSELoss() if use_ranking_loss else criterion

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        # Unpack batch (handles variable length due to interaction features)
        *features, utility = batch

        # Move to device
        features = [f.to(device) for f in features]
        utility = utility.to(device)

        # Forward pass
        pred_utility = _unpack_and_forward(model, features, interaction_features)

        # Compute loss (use MSE for ranking loss validation)
        loss = eval_criterion(pred_utility, utility)
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

    # Spearman correlation (most important for ranking quality)
    spearman_corr, _ = spearmanr(all_predictions, all_targets)

    metrics = {
        'loss': total_loss / num_batches,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman_corr
    }

    # Add BCE-specific metrics
    if use_bce:
        # Calibration: How close are predicted probabilities to true utilities?
        # For BCE, both predictions and targets are in [0,1]
        calibration_error = np.mean(np.abs(all_predictions - all_targets))

        # Top-K agreement: Do we rank the best examples correctly?
        # Check if top 10% of predictions overlap with top 10% of true utilities
        k = max(1, int(0.1 * len(all_predictions)))
        top_k_pred_indices = set(np.argsort(all_predictions)[-k:])
        top_k_true_indices = set(np.argsort(all_targets)[-k:])
        top_k_overlap = len(top_k_pred_indices & top_k_true_indices) / k

        metrics['calibration_error'] = calibration_error  # Same as MAE but more interpretable name
        metrics['top10_overlap'] = top_k_overlap

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

    # Determine loss type and select appropriate dataset
    loss_type = cfg.training.get('loss_type', 'huber')
    print(f"\nLoss type: {loss_type}")

    # Map loss types to dataset classes
    dataset_classes = {
        'mse': MarginalUtilityDataset,
        'huber': MarginalUtilityDataset,
        'ranking': PairwiseMarginalUtilityDataset
    }

    train_dataset_class = dataset_classes.get(loss_type, MarginalUtilityDataset)

    # Load datasets
    print(f"\nLoading datasets...")
    # Check if utilities should be normalized (can be set explicitly in config)
    normalize_utilities = loss_type == 'bce' or cfg.training.get('normalize_utilities', False)
    print(f"  Normalize utilities: {normalize_utilities}")

    train_dataset_kwargs = {
        'results_path': results_path,
        'embeddings_path': embeddings_path,
        'split': 'train',
        'seed': cfg.experiment.seed,
        'interaction_features': interaction_features,
        'normalize_utilities': normalize_utilities
    }

    # Add pairs_per_query for pairwise ranking dataset
    if loss_type == 'ranking':
        train_dataset_kwargs['pairs_per_query'] = cfg.training.get('pairs_per_query', 10)

    train_dataset = train_dataset_class(**train_dataset_kwargs)

    # Validation always uses regular dataset (for MSE/Spearman metrics)
    val_dataset = MarginalUtilityDataset(
        results_path=results_path,
        embeddings_path=embeddings_path,
        split='val',
        seed=cfg.experiment.seed,
        interaction_features=interaction_features,
        normalize_utilities=normalize_utilities
    )

    # Compute baseline
    baseline_mse_train = train_dataset.compute_baseline_mse()
    baseline_mse_val = val_dataset.compute_baseline_mse()
    baseline_spearman_train = train_dataset.compute_baseline_spearman()
    baseline_spearman_val = val_dataset.compute_baseline_spearman()

    print(f"\nBaseline (CLIP similarity):")
    print(f"  Train MSE:      {baseline_mse_train:.4f}")
    print(f"  Val MSE:        {baseline_mse_val:.4f}")
    print(f"  Train Spearman: {baseline_spearman_train:.4f}")
    print(f"  Val Spearman:   {baseline_spearman_val:.4f}")

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
    # Check if sigmoid should be used (can be set explicitly in config)
    use_sigmoid = loss_type == 'bce' or cfg.model.get('use_sigmoid', False)
    model = CLIPReranker(
        embedding_dim=cfg.model.embedding_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        interaction_features=interaction_features,
        use_sigmoid=use_sigmoid
    )
    model = model.to(device)
    print(f"  Parameters: {model.get_num_parameters():,}")
    print(f"  Input dimension: {2 * cfg.model.embedding_dim + 1 + interaction_features.num_features}")
    print(f"  Use sigmoid output: {use_sigmoid}")

    # Loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'huber':
        criterion = nn.HuberLoss(delta=cfg.training.get("huber_delta", 1.0))
    elif loss_type == 'ranking':
        criterion = nn.MarginRankingLoss(margin=cfg.training.get("ranking_margin", 0.1))
    elif loss_type == 'bce':
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

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

    # Choose early stopping metric based on loss type
    if loss_type in ['bce', 'ranking']:
        # For BCE and ranking: use Spearman correlation (higher is better)
        best_val_mse = float('inf')
        best_val_spearman = float('-inf')
        metric_name = 'spearman'
        metric_mode = 'max'
        print(f"Early stopping based on: Spearman correlation (maximize)\n")
    else:
        # For regression losses: use MSE first, then Spearman as tiebreaker
        best_val_mse = float('inf')
        best_val_spearman = float('-inf')
        metric_name = 'mse'
        metric_mode = 'min'
        print(f"Early stopping based on: MSE (minimize), then Spearman (maximize)\n")

    patience_counter = 0

    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_mses = []
    val_maes = []
    val_r2s = []
    val_spearmans = []
    learning_rates = []
    val_calibration_errors = []
    val_top10_overlaps = []

    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        # Train (use appropriate training function based on loss type)
        if loss_type == 'ranking':
            train_loss = train_epoch_ranking(
                model, train_loader, criterion, optimizer, device,
                interaction_features=interaction_features,
                grad_clip=cfg.training.gradient_clip
            )
        else:
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                interaction_features=interaction_features,
                grad_clip=cfg.training.gradient_clip
            )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, interaction_features, use_bce=(loss_type == 'bce'))

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

        # Track BCE-specific metrics if available
        if 'calibration_error' in val_metrics:
            val_calibration_errors.append(val_metrics['calibration_error'])
        if 'top10_overlap' in val_metrics:
            val_top10_overlaps.append(val_metrics['top10_overlap'])

        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")

        # Show different metrics based on loss type
        if loss_type == 'bce':
            # For BCE: focus on ranking and calibration
            print(f"  Val Spearman:    {val_metrics['spearman']:.4f}  (ranking quality)")
            print(f"  Val Calibration: {val_metrics['calibration_error']:.4f}  (prediction accuracy)")
            print(f"  Val Top-10% Hit: {val_metrics['top10_overlap']:.2%}  (best examples ranked correctly)")
            print(f"  Val MSE:         {val_metrics['mse']:.4f}  (on [0,1] scale)")
        else:
            # For regression losses: show traditional metrics
            print(f"  Val MSE:    {val_metrics['mse']:.4f}")
            print(f"  Val MAE:    {val_metrics['mae']:.4f}")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")
            print(f"  Val Spearman: {val_metrics['spearman']:.4f}")

        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model based on two-stage criteria: MSE first, then Spearman
        current_mse = val_metrics['mse']
        current_spearman = val_metrics['spearman']

        # Check if this is an improvement
        is_improvement = False
        if metric_mode == 'min':
            # For regression: prioritize MSE, use Spearman as tiebreaker
            if current_mse < best_val_mse:
                is_improvement = True
            elif abs(current_mse - best_val_mse) < 1e-6 and current_spearman > best_val_spearman:
                # If MSE is essentially the same, prefer higher Spearman
                is_improvement = True
        else:
            # For ranking-based losses: prioritize Spearman
            if current_spearman > best_val_spearman:
                is_improvement = True
            elif abs(current_spearman - best_val_spearman) < 1e-6 and current_mse < best_val_mse:
                # If Spearman is essentially the same, prefer lower MSE
                is_improvement = True

        if is_improvement:
            best_val_mse = current_mse
            best_val_spearman = current_spearman
            patience_counter = 0

            if cfg.checkpoint.enabled:
                checkpoint_path = Path(cfg.checkpoint.save_dir) / cfg.experiment.name / "best_model.pt"
                save_checkpoint(model, optimizer, epoch, val_metrics, cfg, checkpoint_path)
                print(f"  ✓ Saved best model (MSE: {best_val_mse:.4f}, Spearman: {best_val_spearman:.4f})")
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
    print(f"Best validation Spearman: {best_val_spearman:.4f}")
    if metric_name == 'mse':
        print(f"Baseline MSE: {baseline_mse_val:.4f}")
        print(f"MSE Improvement: {(1 - best_val_mse / baseline_mse_val) * 100:.1f}%")
    print(f"Baseline Spearman: {baseline_spearman_val:.4f}")
    print(f"Spearman Improvement: {(best_val_spearman - baseline_spearman_val) / baseline_spearman_val * 100:.1f}%")
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
