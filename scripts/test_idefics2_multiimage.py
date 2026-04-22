"""Test Idefics2 multi-image support and ICL capability."""

import sys
sys.path.insert(0, "src")

from data.mini_imagenet import MiniImageNetDataset
from models.idefics2_wrapper import Idefics2Wrapper
from utils.imagenet_names import get_readable_name
import torch
import numpy as np

def verify_multi_image_support(idefics2, test_dataset):
    """
    Verify that Idefics2 correctly processes multiple images with ICL structure.

    Test: Show example from class A, query from class B.
    Expected: Model should predict class B (query), not class A (example).
    """
    print("\n" + "=" * 80)
    print("VERIFICATION TEST: Idefics2 Multi-Image ICL Support")
    print("=" * 80)

    # Find two examples from DIFFERENT classes
    example_idx = 0
    example_ex, example_img = test_dataset[example_idx]
    example_label = get_readable_name(example_ex.label_name)

    # Query: Find an image from a DIFFERENT class
    query_idx = 500
    query_ex, query_img = test_dataset[query_idx]
    query_label = get_readable_name(query_ex.label_name)

    print(f"\nTest setup:")
    print(f"  Example: '{example_label}' (index {example_idx})")
    print(f"  Query:   '{query_label}' (index {query_idx})")

    # Verify images are different
    example_array = np.array(example_img)
    query_array = np.array(query_img)

    print(f"\n--- IMAGE VERIFICATION ---")
    print(f"  Example image shape: {example_array.shape}")
    print(f"  Query image shape: {query_array.shape}")
    print(f"  Example mean pixel: {example_array.mean():.2f}")
    print(f"  Query mean pixel: {query_array.mean():.2f}")

    # Resize to same shape if needed for comparison
    if example_array.shape != query_array.shape:
        print(f"  Images have different shapes - they are definitely different!")
        images_are_same = False
    else:
        images_are_same = np.array_equal(example_array, query_array)
        if images_are_same:
            print(f"  ❌ WARNING: Images are IDENTICAL!")
        else:
            diff = np.abs(example_array.astype(float) - query_array.astype(float)).mean()
            print(f"  Mean absolute difference: {diff:.2f}")
            print(f"  ✓ Images are DIFFERENT")

    print(f"\nExpected behavior:")
    print(f"  ✓ Correct:   Model predicts '{query_label}' (uses query image)")
    print(f"  ✗ Incorrect: Model predicts '{example_label}' (copying example label)")

    # Test with discriminative evaluation on just these two classes
    candidate_labels = sorted(set([example_label, query_label]))

    # Prepare prompt and images
    prompt = idefics2.format_prompt(example_labels=[example_label], candidate_labels=None)
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
        batch_log_probs = idefics2._compute_label_probabilities_batch(
            images=[[example_img, query_img]],
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
    print(f"\nComputing probabilities with swapped images...")
    log_probs_swapped = []
    for label in candidate_labels:
        batch_log_probs = idefics2._compute_label_probabilities_batch(
            images=[[query_img, example_img]],  # SWAPPED order
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
        print("   → Model correctly maps each image to its position")
        return True
    else:
        print(f"\n⚠️  UNEXPECTED: forward='{predicted_forward}', swapped='{predicted_swapped}'")
        print("   → Results are inconsistent, further investigation needed")
        return False


def main():
    print("=" * 80)
    print("Testing Idefics2 Multi-Image ICL Capability")
    print("=" * 80)

    # Load test dataset
    test_dataset = MiniImageNetDataset(split="test")
    print(f"Loaded {len(test_dataset)} test examples")

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    load_in_8bit = torch.cuda.is_available()

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available, loading full precision model on CPU")

    idefics2 = Idefics2Wrapper(device=device, load_in_8bit=load_in_8bit)
    print(f"\nModel loaded on device: {device}, 8-bit: {load_in_8bit}")

    # Run verification test
    multi_image_works = verify_multi_image_support(idefics2, test_dataset)

    if multi_image_works:
        print("\n✅ SUCCESS: Idefics2 properly supports multi-image ICL!")
        print("   Ready to proceed with full evaluation.")
    else:
        print("\n❌ FAILURE: Idefics2 multi-image ICL is not working correctly.")
        print("   Need to investigate further.")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
