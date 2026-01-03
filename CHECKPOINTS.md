# Checkpoint System for Marginal Utility Computation

## Overview

The marginal utility computation script supports automatic checkpointing and resumption to handle long-running experiments safely.

## How It Works

### Automatic Checkpointing

The script automatically saves progress every N queries (default: 100) to:
```
outputs/marginal_utilities/{experiment_name}/checkpoints/checkpoint_{query_idx:06d}.pkl
```

Each checkpoint contains:
- All results computed so far
- Last completed query index
- Full configuration
- Statistics (num_queries, num_pairs)

### Automatic Resumption

When you restart the script, it automatically:
1. Looks for existing checkpoints in the checkpoint directory
2. Loads the latest checkpoint
3. Resumes from the next query
4. Continues saving checkpoints at the configured interval

### Configuration

In `configs/marginal_utility.yaml`:

```yaml
checkpoint:
  enabled: true  # Enable/disable checkpointing
  save_dir: "outputs/marginal_utilities"
  save_interval: 100  # Save every N queries
  resume_from: null  # Not used (auto-detects latest)
```

## Usage Examples

### Standard Run (with checkpointing)
```bash
python scripts/compute_marginal_utilities.py
```

### Override Checkpoint Interval
```bash
# Save checkpoint every 50 queries
python scripts/compute_marginal_utilities.py checkpoint.save_interval=50
```

### Disable Checkpointing
```bash
python scripts/compute_marginal_utilities.py checkpoint.enabled=false
```

### Resume After Crash
Just run the same command again - it will automatically detect and resume:
```bash
python scripts/compute_marginal_utilities.py
```

Output will show:
```
Found existing checkpoint: outputs/.../checkpoint_000300.pkl
✓ Loaded checkpoint with 6000 results
  Last completed query: 300
  Resuming from query 301
```

## Inspecting Checkpoints

Use the checkpoint inspection script:

```bash
# Inspect latest checkpoint
python scripts/inspect_checkpoint.py

# Inspect specific checkpoint
python scripts/inspect_checkpoint.py outputs/.../checkpoint_000100.pkl
```

## Important Notes

1. **Checkpoint files accumulate** - each checkpoint is a separate file
   - Checkpoint at query 100: ~10MB (2000 results)
   - Checkpoint at query 1000: ~100MB (20,000 results)
   - Checkpoint at query 6504: ~650MB (130,080 results for top_k=20)

2. **Final results file** - saved separately at the end:
   - Location: `outputs/marginal_utilities/{experiment}/marginal_utilities_train.pkl`
   - This is the file you'll use for training the reranker

3. **Checkpoint cleanup** - After successful completion, you can delete old checkpoints:
   ```bash
   rm -rf outputs/marginal_utilities/{experiment}/checkpoints/
   ```

4. **Starting fresh** - To ignore existing checkpoints and start over:
   ```bash
   # Option 1: Rename the experiment
   python scripts/compute_marginal_utilities.py experiment.name="marginal_utility_v2"

   # Option 2: Delete checkpoints
   rm -rf outputs/marginal_utilities/marginal_utility_stanford_cars/checkpoints/
   ```

## Recovery Scenarios

### Script Crashed Mid-Run
```bash
# Just restart - will resume automatically
python scripts/compute_marginal_utilities.py
```

### Machine Rebooted
```bash
# Same - checkpoints are on disk
python scripts/compute_marginal_utilities.py
```

### Want to Process More Queries
```bash
# If you ran with limits.max_queries=100, then want to do all:
python scripts/compute_marginal_utilities.py limits.max_queries=null
# Will start from query 101
```

### Checkpoint Interval Too Large/Small
```bash
# Change for future checkpoints (doesn't affect already-saved ones)
python scripts/compute_marginal_utilities.py checkpoint.save_interval=50
```

## Performance Considerations

For a full run with 6,504 queries:
- With `save_interval=100`: ~65 checkpoints, ~15GB total
- With `save_interval=50`: ~130 checkpoints, ~30GB total
- With `save_interval=200`: ~33 checkpoints, ~7.5GB total

**Recommendation**: Use `save_interval=100` (default) for good balance between safety and disk usage.
