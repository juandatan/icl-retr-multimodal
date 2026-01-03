# ICL Retrieval for Multimodal Models

Learning to predict the utility of in-context examples for multimodal VQA tasks.

## Project Structure

```
├── data/                      # Data storage
│   ├── vqa_v2/               # VQAv2 dataset
│   └── utilities/            # Generated utility datasets
├── src/
│   ├── data/
│   │   ├── vqa_loader.py     # VQAv2 dataset handling
│   │   └── utility_dataset.py # Utility dataset generation
│   ├── models/
│   │   ├── llava_wrapper.py  # LLaVA-1.5-7B interface
│   │   └── clip_reranker.py  # CLIP + MLP reranker
│   └── utils/
│       ├── compute_utility.py # Utility computation logic
│       └── retrieval.py       # Example retrieval strategies
├── scripts/
│   ├── generate_utilities.py # Main data generation script
│   └── train_reranker.py     # Training script
└── configs/                   # Hydra configurations
    └── generate_utilities.yaml
```

## Setup

### 1. Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Download

The Stanford Cars dataset will be automatically downloaded from HuggingFace on first use and cached locally in `data/stanford_cars/hf_cache/`. This is a one-time download (~800MB).

To manually verify the dataset:

```bash
pytest src/tests/test_stanford_cars.py -v
```

### 3. Generate CLIP Embeddings

Pre-compute CLIP embeddings for fast semantic similarity-based retrieval:

```bash
# Generate embeddings for all splits (takes ~5-10 minutes on CPU)
python scripts/build_clip_embeddings.py

# Or generate for specific splits
python scripts/build_clip_embeddings.py --splits train val

# Use a larger/better CLIP model (slower but better)
python scripts/build_clip_embeddings.py --model ViT-L/14

# Use GPU if available
python scripts/build_clip_embeddings.py --device cuda
```

Test the semantic similarity retrieval:

```bash
# Test similarity retrieval
python scripts/test_clip_similarity.py --query_idx 0 --top_k 10

# Visualize similar images (saves PNG)
python scripts/test_clip_similarity.py --query_idx 0 --top_k 5 --visualize
```

## Data Generation

Generate utility training data:

```bash
python scripts/generate_utilities.py
```
