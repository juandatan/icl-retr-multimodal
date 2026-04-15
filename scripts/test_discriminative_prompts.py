"""Test discriminative evaluation with different prompt configurations."""

import sys
sys.path.insert(0, "src")

from data.mini_imagenet import MiniImageNetDataset
from models.llava_wrapper import LLaVAWrapper
from models.mlp_reranker import MLPReranker
from utils.imagenet_names import IMAGENET_SYNSET_TO_NAME, get_readable_name
import torch
import numpy as np

def test_discriminative_with_config(
    llava: LLaVAWrapper,
    query_image,
    context_examples,
    candidate_labels,
    show_classes_in_prompt: bool,
    k: int,
    test_name: str
):
    """Test discriminative evaluation with given configuration."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print(f"  k={k}, show_classes_in_prompt={show_classes_in_prompt}")
    print("=" * 80)

    # Format prompt
    example_labels = [label for _, label in context_examples] if context_examples else None

    if show_classes_in_prompt:
        # Modified: show candidate list in prompt
        prompt = llava.format_prompt(example_labels=example_labels, candidate_labels=candidate_labels)
    else:
        # Standard discriminative: no candidate list in prompt
        prompt = llava.format_prompt(example_labels=example_labels, candidate_labels=None)

    print("\n--- PROMPT ---")
    if len(prompt) > 800:
        print(prompt[:400])
        print("\n... [middle truncated] ...\n")
        print(prompt[-400:])
    else:
        print(prompt)

    # Prepare images for probability computation
    images = [img for img, _ in context_examples] + [query_image]

    # Compute log probabilities for ALL candidates, one at a time to avoid OOM
    print(f"\n--- COMPUTING LOG PROBABILITIES ---")
    print(f"Computing for all {len(candidate_labels)} candidates (one at a time to avoid OOM)...")

    log_probs = []
    for i, label in enumerate(candidate_labels):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(candidate_labels)}...")

        # Process one candidate at a time
        batch_log_probs = llava._compute_label_probabilities_batch(
            images=[images],
            prompts=[prompt],
            labels=[label]
        )
        log_probs.extend(batch_log_probs)

    # Sort candidates by probability (descending)
    sorted_pairs = sorted(zip(candidate_labels, log_probs), key=lambda x: x[1], reverse=True)

    print(f"\n--- ALL PREDICTIONS (sorted by log probability) ---")
    for rank, (label, log_prob) in enumerate(sorted_pairs, 1):
        print(f"  {rank:3d}. {label:30s} log_prob={log_prob:.4f}")

    return sorted_pairs[0][0]


def retrieve_clip_example(query_emb: np.ndarray, train_dataset, k: int = 1):
    """Retrieve top-k most similar examples using CLIP embeddings."""
    # Compute cosine similarity with all train examples
    # Normalize embeddings
    query_emb_norm = query_emb / np.linalg.norm(query_emb)
    train_embs_norm = train_dataset.clip_embeddings / np.linalg.norm(train_dataset.clip_embeddings, axis=1, keepdims=True)

    # Compute similarities
    similarities = train_embs_norm @ query_emb_norm

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:k]

    return top_indices, similarities[top_indices]


def retrieve_reranker_example(query_emb: np.ndarray, train_dataset, reranker, device, k: int = 1, clip_topk: int = 50):
    """Retrieve top-k examples using reranker predictions."""
    # First, get top-50 CLIP candidates
    clip_indices, _ = retrieve_clip_example(query_emb, train_dataset, k=clip_topk)

    # Get query embedding as torch tensor
    query_emb_torch = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)

    # Compute utilities for all CLIP candidates
    utilities = []
    for idx in clip_indices:
        example_emb = torch.from_numpy(train_dataset.clip_embeddings[idx]).float().unsqueeze(0).to(device)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(query_emb_torch, example_emb)

        # Predict utility
        with torch.no_grad():
            utility = reranker(query_emb_torch, example_emb, similarity.unsqueeze(0))

        utilities.append(utility.item())

    # Get top-k by utility
    top_k_indices = np.argsort(utilities)[::-1][:k]
    selected_indices = [clip_indices[i] for i in top_k_indices]
    selected_utilities = [utilities[i] for i in top_k_indices]

    return selected_indices, selected_utilities


def verify_multi_image_support(llava, test_dataset):
    """
    Verify that LLaVA correctly processes multiple images.

    Test: Show example from class A, query from class B.
    Expected: Model should predict class B (query), not class A (example).
    If it predicts A, it's copying the example label (multi-image broken).
    """
    print("\n" + "=" * 80)
    print("VERIFICATION TEST: Multi-Image Support")
    print("=" * 80)

    # Find two examples from DIFFERENT classes
    # Example: golden retriever (class 14)
    example_idx = 0
    example_ex, example_img = test_dataset[example_idx]
    example_label = get_readable_name(example_ex.label_name)

    # Query: Find an image from a DIFFERENT class (e.g., house finch)
    query_idx = None
    for i in range(len(test_dataset)):
        query_ex, _ = test_dataset[i]
        query_label = get_readable_name(query_ex.label_name)
        if query_label != example_label:
            query_idx = i
            break

    query_ex, query_img = test_dataset[query_idx]
    query_label = get_readable_name(query_ex.label_name)

    print(f"\nTest setup:")
    print(f"  Example: '{example_label}' (index {example_idx})")
    print(f"  Query:   '{query_label}' (index {query_idx})")
    print(f"\nExpected behavior:")
    print(f"  ✓ Correct:   Model predicts '{query_label}' (uses query image)")
    print(f"  ✗ Incorrect: Model predicts '{example_label}' (copying example label)")

    # Test with discriminative evaluation on just these two classes
    candidate_labels = sorted(set([example_label, query_label]))

    # Prepare prompt and images
    prompt = llava.format_prompt(example_labels=[example_label], candidate_labels=None)
    images = [example_img, query_img]

    print(f"\n--- PROMPT STRUCTURE ---")
    if len(prompt) > 800:
        print(prompt[:400])
        print("\n... [middle truncated] ...\n")
        print(prompt[-400:])
    else:
        print(prompt)
    print(f"--- END PROMPT ---")
    print(f"\n--- IMAGE ORDER ---")
    print(f"  images[0]: Example ({example_label})")
    print(f"  images[1]: Query ({query_label})")
    num_image_tokens = prompt.count('<image>')
    print(f"  Number of <image> tokens in prompt: {num_image_tokens}")
    print(f"  Number of images provided: {len(images)}")
    if num_image_tokens != len(images):
        print(f"  ⚠️  MISMATCH: {num_image_tokens} <image> tokens but {len(images)} images!")
    else:
        print(f"  ✓ Match!")
    print(f"\nComputing probabilities for: {candidate_labels}")

    # Compute log probabilities
    log_probs = []
    for label in candidate_labels:
        batch_log_probs = llava._compute_label_probabilities_batch(
            images=[images],
            prompts=[prompt],
            labels=[label]
        )
        log_probs.append(batch_log_probs[0])

    # Show results
    print("\nResults (image order: [example, query]):")
    for label, log_prob in zip(candidate_labels, log_probs):
        print(f"  {label}: log_prob={log_prob:.4f}")

    predicted_forward = candidate_labels[np.argmax(log_probs)]

    print(f"\nPredicted: '{predicted_forward}'")
    print(f"True label: '{query_label}'")

    # CRITICAL: Now test with SWAPPED image order
    print("\n" + "-" * 60)
    print("SWAP TEST: Reversing image order to verify correct mapping")
    print("-" * 60)
    print("\n--- NEW IMAGE ORDER ---")
    print(f"  images[0]: Query ({query_label}) [SWAPPED]")
    print(f"  images[1]: Example ({example_label}) [SWAPPED]")

    # Swap the images but keep the same prompt (should give different result!)
    images_swapped = [query_img, example_img]

    print(f"\nComputing probabilities with swapped images...")
    log_probs_swapped = []
    for label in candidate_labels:
        batch_log_probs = llava._compute_label_probabilities_batch(
            images=[images_swapped],
            prompts=[prompt],  # Same prompt!
            labels=[label]
        )
        log_probs_swapped.append(batch_log_probs[0])

    print("\nResults (image order: [query, example]):")
    for label, log_prob in zip(candidate_labels, log_probs_swapped):
        print(f"  {label}: log_prob={log_prob:.4f}")

    predicted_swapped = candidate_labels[np.argmax(log_probs_swapped)]
    print(f"\nPredicted after swap: '{predicted_swapped}'")

    # Analysis
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    if predicted_forward == example_label and predicted_swapped == example_label:
        print("\n❌ FAIL: Model predicts example label in BOTH cases")
        print("   → Model is copying the example label from the prompt text")
        print("   → Multi-image mapping is BROKEN")
        return False
    elif predicted_forward == query_label and predicted_swapped == query_label:
        print("\n❌ FAIL: Model predicts query label in BOTH cases")
        print("   → Model is ignoring the example and always predicting based on images[1]")
        print("   → Multi-image mapping may be reversed or broken")
        return False
    elif predicted_forward == example_label and predicted_swapped == query_label:
        print("\n❌ FAIL: Model always predicts images[0] regardless of which image it is")
        print("   → Model is only looking at the first image")
        print("   → Multi-image mapping is BROKEN")
        return False
    elif predicted_forward == query_label and predicted_swapped == example_label:
        print("\n✅ PASS: Predictions changed when images were swapped!")
        print(f"   → Forward:  predicted '{predicted_forward}' (query at position 1)")
        print(f"   → Swapped:  predicted '{predicted_swapped}' (query at position 0)")
        print("   → Model correctly maps each <image> token to its image")
        return True
    else:
        print(f"\n⚠️  UNEXPECTED: forward='{predicted_forward}', swapped='{predicted_swapped}'")
        print("   → Results are inconsistent, further investigation needed")
        return False


def main():
    print("=" * 80)
    print("Loading Mini-ImageNet dataset...")
    print("=" * 80)

    # Load test dataset
    test_dataset = MiniImageNetDataset(split="test")

    # Load train dataset with CLIP embeddings for retrieval
    train_dataset = MiniImageNetDataset(split="train")

    print(f"Loaded {len(test_dataset)} test examples, {len(train_dataset)} train examples")

    # Get a test image (golden retriever)
    query_idx = 0
    query_example, query_image = test_dataset[query_idx]
    true_label = get_readable_name(query_example.label_name)

    print(f"\nQuery image: index={query_idx}, true_label='{true_label}'")
    print(f"Image size: {query_image.size}")

    # Get example from different class (malamute) - for baseline comparison
    random_example_idx = 500
    random_example, random_image = train_dataset[random_example_idx]
    random_label = get_readable_name(random_example.label_name)
    print(f"Random example: index={random_example_idx}, label='{random_label}'")

    # Get all 100 candidate labels
    candidate_labels = list(IMAGENET_SYNSET_TO_NAME.values())

    print("\n" + "=" * 80)
    print("Loading LLaVA model...")
    print("=" * 80)

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    load_in_8bit = torch.cuda.is_available()

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available, loading full precision model on CPU")

    llava = LLaVAWrapper(device=device, load_in_8bit=load_in_8bit)
    print(f"\nModel loaded on device: {device}, 8-bit: {load_in_8bit}")

    # CRITICAL: Verify multi-image support first
    multi_image_works = verify_multi_image_support(llava, test_dataset)

    if not multi_image_works:
        print("\n⚠️  WARNING: Multi-image support appears broken!")
        print("   ICL results may be unreliable. Consider using k=0 only.")
        print("   Or try a different model with proper multi-image support.")

    # Get query CLIP embedding (skip if multi-image is broken)
    if not multi_image_works:
        print("\nSkipping retrieval tests since multi-image is broken.")
        return

    query_clip_emb = test_dataset.clip_embeddings[query_idx]

    # Get CLIP-retrieved example
    print("\n" + "=" * 80)
    print("Retrieving examples with CLIP...")
    print("=" * 80)

    clip_indices, clip_sims = retrieve_clip_example(query_clip_emb, train_dataset, k=1)
    clip_example, clip_image = train_dataset[clip_indices[0]]
    clip_label = get_readable_name(clip_example.label_name)
    print(f"CLIP top-1: index={clip_indices[0]}, label='{clip_label}', similarity={clip_sims[0]:.4f}")

    # Get Reranker-retrieved example
    print("\n" + "=" * 80)
    print("Loading reranker and retrieving examples...")
    print("=" * 80)

    # Find the latest reranker checkpoint
    import glob
    checkpoints = glob.glob("/Users/jd.tan/Projects/icl-retr-multimodal/outputs/reranker_checkpoints/*.pt")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: x)
        print(f"Loading reranker from: {latest_checkpoint}")

        reranker = MLPReranker(embedding_dim=512)
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        reranker.load_state_dict(checkpoint['model_state_dict'])
        reranker.to(device)
        reranker.eval()

        reranker_indices, reranker_utilities = retrieve_reranker_example(
            query_clip_emb, train_dataset, reranker, device, k=1, clip_topk=50
        )
        reranker_example, reranker_image = train_dataset[reranker_indices[0]]
        reranker_label = get_readable_name(reranker_example.label_name)
        print(f"Reranker top-1: index={reranker_indices[0]}, label='{reranker_label}', utility={reranker_utilities[0]:.4f}")
        has_reranker = True
    else:
        print("No reranker checkpoint found, skipping reranker test")
        has_reranker = False

    # Test 1: k=1 with random example (baseline)
    pred1 = test_discriminative_with_config(
        llava=llava,
        query_image=query_image,
        context_examples=[(random_image, random_label)],
        candidate_labels=candidate_labels,
        show_classes_in_prompt=True,
        k=1,
        test_name="k=1, random example (malamute)"
    )

    # Test 2: k=1 with CLIP-retrieved example
    pred2 = test_discriminative_with_config(
        llava=llava,
        query_image=query_image,
        context_examples=[(clip_image, clip_label)],
        candidate_labels=candidate_labels,
        show_classes_in_prompt=True,
        k=1,
        test_name="k=1, CLIP-retrieved example"
    )

    # Test 3: k=1 with Reranker-retrieved example (if available)
    if has_reranker:
        pred3 = test_discriminative_with_config(
            llava=llava,
            query_image=query_image,
            context_examples=[(reranker_image, reranker_label)],
            candidate_labels=candidate_labels,
            show_classes_in_prompt=True,
            k=1,
            test_name="k=1, Reranker-retrieved example"
        )

    # Test 4: k=0 baseline for comparison
    pred4 = test_discriminative_with_config(
        llava=llava,
        query_image=query_image,
        context_examples=[],
        candidate_labels=candidate_labels,
        show_classes_in_prompt=True,
        k=0,
        test_name="k=0 (0-shot baseline)"
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"True label: {true_label}")
    print(f"\nRetrieval methods:")
    print(f"  Random:   example={random_label}")
    print(f"  CLIP:     example={clip_label}, similarity={clip_sims[0]:.4f}")
    if has_reranker:
        print(f"  Reranker: example={reranker_label}, utility={reranker_utilities[0]:.4f}")

    print(f"\nPredictions:")
    print(f"  1. k=1, random:    {pred1} {'✓' if pred1 == true_label else '✗'}")
    print(f"  2. k=1, CLIP:      {pred2} {'✓' if pred2 == true_label else '✗'}")
    if has_reranker:
        print(f"  3. k=1, Reranker:  {pred3} {'✓' if pred3 == true_label else '✗'}")
    print(f"  4. k=0 (baseline): {pred4} {'✓' if pred4 == true_label else '✗'}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
