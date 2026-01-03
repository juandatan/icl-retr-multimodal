"""
Pytest tests for Stanford Cars dataset loading and class splitting.

Run with:
    pytest src/tests/test_stanford_cars.py -v
    pytest src/tests/test_stanford_cars.py -v -s  # with print output
"""

import pytest
from collections import Counter

from src.data.stanford_cars import StanfordCarsDataset


@pytest.fixture(scope="module")
def datasets():
    """Load all three dataset splits once for all tests."""
    print("\nLoading datasets...")
    dataset_train = StanfordCarsDataset(split='train', data_dir='./data/stanford_cars')
    dataset_val = StanfordCarsDataset(split='val', data_dir='./data/stanford_cars')
    dataset_test = StanfordCarsDataset(split='test', data_dir='./data/stanford_cars')
    return {
        'train': dataset_train,
        'val': dataset_val,
        'test': dataset_test
    }


class TestDatasetLoading:
    """Test basic dataset loading functionality."""

    def test_train_split_loads(self, datasets):
        """Test that train split loads successfully."""
        assert len(datasets['train']) > 0, "Train split should not be empty"
        print(f"\nTrain split: {len(datasets['train'])} examples")

    def test_val_split_loads(self, datasets):
        """Test that val split loads successfully."""
        assert len(datasets['val']) > 0, "Val split should not be empty"
        print(f"Val split: {len(datasets['val'])} examples")

    def test_test_split_loads(self, datasets):
        """Test that test split loads successfully."""
        assert len(datasets['test']) > 0, "Test split should not be empty"
        print(f"Test split: {len(datasets['test'])} examples")

    def test_total_classes(self, datasets):
        """Test that total number of classes is 196."""
        train_classes = set(datasets['train'].train_classes)
        val_classes = set(datasets['val'].val_classes)
        test_classes = set(datasets['test'].test_classes)

        total_classes = train_classes | val_classes | test_classes
        assert len(total_classes) == 196, f"Expected 196 total classes, got {len(total_classes)}"


class TestClassSplits:
    """Test class split disjointness and correctness."""

    def test_class_splits_are_disjoint(self, datasets):
        """Test that train/val/test class splits have no overlap."""
        train_classes = set(datasets['train'].train_classes)
        val_classes = set(datasets['val'].val_classes)
        test_classes = set(datasets['test'].test_classes)

        # Check pairwise disjointness
        train_val_overlap = train_classes & val_classes
        train_test_overlap = train_classes & test_classes
        val_test_overlap = val_classes & test_classes

        assert len(train_val_overlap) == 0, f"Train/Val overlap: {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"Train/Test overlap: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"Val/Test overlap: {val_test_overlap}"

    def test_class_split_sizes(self, datasets):
        """Test that class splits have expected sizes."""
        n_train = len(datasets['train'].train_classes)
        n_val = len(datasets['val'].val_classes)
        n_test = len(datasets['test'].test_classes)

        print(f"\nClass counts: train={n_train}, val={n_val}, test={n_test}")

        # Check ratios are approximately correct (80/10/10)
        total = n_train + n_val + n_test
        assert total == 196, f"Total classes should be 196, got {total}"

        # Allow some flexibility due to rounding
        assert 150 <= n_train <= 160, f"Train classes should be ~156, got {n_train}"
        assert 15 <= n_val <= 25, f"Val classes should be ~20, got {n_val}"
        assert 15 <= n_test <= 25, f"Test classes should be ~20, got {n_test}"

    def test_train_examples_only_train_classes(self, datasets):
        """Test that train examples only contain train classes."""
        train_classes = set(datasets['train'].train_classes)
        train_labels = set(ex.label for ex in datasets['train'].examples)

        assert train_labels.issubset(train_classes), \
            "Train examples contain labels not in train classes"

    def test_val_examples_only_val_classes(self, datasets):
        """Test that val examples only contain val classes."""
        val_classes = set(datasets['val'].val_classes)
        val_labels = set(ex.label for ex in datasets['val'].examples)

        assert val_labels.issubset(val_classes), \
            "Val examples contain labels not in val classes"

    def test_test_examples_only_test_classes(self, datasets):
        """Test that test examples only contain test classes."""
        test_classes = set(datasets['test'].test_classes)
        test_labels = set(ex.label for ex in datasets['test'].examples)

        assert test_labels.issubset(test_classes), \
            "Test examples contain labels not in test classes"


class TestImageLoading:
    """Test image loading functionality."""

    def test_load_first_image(self, datasets):
        """Test loading the first image from train split."""
        example, image = datasets['train'][0]

        # Check example properties
        assert example.index == 0
        assert example.label >= 0
        assert example.split == 'train'
        assert len(example.image_path) > 0

        # Check image properties
        assert image.mode == 'RGB', f"Expected RGB image, got {image.mode}"
        assert image.size[0] > 0 and image.size[1] > 0, "Invalid image size"

    def test_load_multiple_images(self, datasets):
        """Test loading multiple images from train split."""
        indices = [0, 10, 50, 100]

        for idx in indices:
            if idx < len(datasets['train']):
                example, image = datasets['train'][idx]
                assert image.mode == 'RGB', f"Image {idx} should be RGB"
                assert image.size[0] > 0 and image.size[1] > 0

    def test_example_properties(self, datasets):
        """Test that example dataclass has correct properties."""
        example, _ = datasets['train'][0]

        # Check all required attributes exist
        assert hasattr(example, 'index')
        assert hasattr(example, 'image_path')
        assert hasattr(example, 'label')
        assert hasattr(example, 'label_name')
        assert hasattr(example, 'split')

        # Check types
        assert isinstance(example.index, int)
        assert isinstance(example.label, int)
        assert isinstance(example.label_name, str)
        assert isinstance(example.split, str)


class TestClassDistribution:
    """Test class distribution statistics."""

    def test_train_class_distribution(self, datasets):
        """Analyze class distribution in train split."""
        label_counts = Counter(ex.label for ex in datasets['train'].examples)

        assert len(label_counts) > 0, "Train split should have examples"

        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        avg_count = sum(label_counts.values()) / len(label_counts)

        print(f"\nTrain distribution:")
        print(f"  Unique classes: {len(label_counts)}")
        print(f"  Min/Max/Avg per class: {min_count}/{max_count}/{avg_count:.1f}")

        # Sanity check: should have multiple examples per class
        assert min_count > 0, "All classes should have at least one example"

    def test_val_class_distribution(self, datasets):
        """Analyze class distribution in val split."""
        label_counts = Counter(ex.label for ex in datasets['val'].examples)

        assert len(label_counts) > 0, "Val split should have examples"

        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        avg_count = sum(label_counts.values()) / len(label_counts)

        print(f"\nVal distribution:")
        print(f"  Unique classes: {len(label_counts)}")
        print(f"  Min/Max/Avg per class: {min_count}/{max_count}/{avg_count:.1f}")

    def test_test_class_distribution(self, datasets):
        """Analyze class distribution in test split."""
        label_counts = Counter(ex.label for ex in datasets['test'].examples)

        assert len(label_counts) > 0, "Test split should have examples"

        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        avg_count = sum(label_counts.values()) / len(label_counts)

        print(f"\nTest distribution:")
        print(f"  Unique classes: {len(label_counts)}")
        print(f"  Min/Max/Avg per class: {min_count}/{max_count}/{avg_count:.1f}")


class TestReproducibility:
    """Test that class splits are reproducible with same seed."""

    def test_same_seed_same_splits(self):
        """Test that same seed produces same class splits."""
        dataset1 = StanfordCarsDataset(split='train', class_split_seed=42)
        dataset2 = StanfordCarsDataset(split='train', class_split_seed=42)

        assert list(dataset1.train_classes) == list(dataset2.train_classes), \
            "Same seed should produce same train classes"
        assert list(dataset1.val_classes) == list(dataset2.val_classes), \
            "Same seed should produce same val classes"
        assert list(dataset1.test_classes) == list(dataset2.test_classes), \
            "Same seed should produce same test classes"

    def test_different_seed_different_splits(self):
        """Test that different seed produces different class splits."""
        dataset1 = StanfordCarsDataset(split='train', class_split_seed=42)
        dataset2 = StanfordCarsDataset(split='train', class_split_seed=123)

        # Class splits should be different (with high probability)
        assert list(dataset1.train_classes) != list(dataset2.train_classes), \
            "Different seeds should produce different splits"
