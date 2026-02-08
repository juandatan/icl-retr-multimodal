"""
Mini-ImageNet Dataset for ICL Utility Learning

Mini-ImageNet is the standard benchmark for few-shot learning with 100 diverse
classes sampled from ImageNet. Unlike fine-grained datasets like Stanford Cars,
it provides strong semantic diversity which should yield clearer marginal utility
signals for reranker training.

Dataset: 100 classes, 600 images per class (500 train, 50 val, 50 test in original split)
For our purposes, we use the full training set and split by classes (80/10/10).
"""

from typing import Any, Optional, Tuple

from PIL import Image
from datasets import load_dataset, load_from_disk

from .base_dataset import BaseUtilityDataset, ClassificationExample


class MiniImageNetDataset(BaseUtilityDataset):
    """
    Mini-ImageNet dataset for ICL utility learning.

    100 diverse classes from ImageNet, specifically curated for few-shot learning.
    Provides much stronger semantic diversity than fine-grained datasets,
    which should produce clearer marginal utility signals.

    Supports:
    - Disjoint class splits for train/val/test
    - CLIP embedding pre-computation
    - Semantic similarity-based candidate retrieval
    """

    def __init__(
        self,
        split: str = 'train',
        data_dir: str = './data/mini_imagenet',
        class_split_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        build_embeddings: bool = False,
        clip_model=None,
        clip_preprocess=None,
        embedding_batch_size: int = 32,
        device: str = 'cpu',
    ):
        """
        Args:
            split: One of 'train', 'val', 'test'
            data_dir: Directory to store dataset
            class_split_seed: Random seed for reproducible class splits
            train_ratio: Proportion of classes for training (default 0.8 = 80 classes)
            val_ratio: Proportion of classes for validation (default 0.1 = 10 classes)
            build_embeddings: Whether to build CLIP embeddings in constructor
            clip_model: CLIP model (required if build_embeddings=True)
            clip_preprocess: CLIP preprocessing (required if build_embeddings=True)
            embedding_batch_size: Batch size for embedding computation
            device: Device for embedding computation
        """
        # Store HF dataset reference
        self.hf_dataset: Any = None

        # Call parent init (which calls load_data())
        super().__init__(
            split=split,
            data_dir=data_dir,
            class_split_seed=class_split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        # Optionally build embeddings in constructor
        if build_embeddings:
            if clip_model is None or clip_preprocess is None:
                raise ValueError("clip_model and clip_preprocess required when build_embeddings=True")

            self.build_clip_embeddings(
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                batch_size=embedding_batch_size,
                device=device,
            )

    def load_data(self):
        """
        Load Mini-ImageNet dataset from HuggingFace.

        We use the 'train' split from the original Mini-ImageNet which contains
        all training images. Our BaseUtilityDataset will then split by classes.

        Populates:
        - self.hf_dataset: HuggingFace dataset
        - self.examples: List of all examples (before split filtering)
        - self.num_classes: 100 classes
        - self.class_names: List of class names (ImageNet synsets)
        """
        print(f"Loading Mini-ImageNet dataset...")

        # Check if dataset is already saved locally
        local_dataset_path = self.data_dir / "hf_cache" / "mini_imagenet_train"

        if local_dataset_path.exists():
            print(f"Loading cached dataset from {local_dataset_path}...")
            self.hf_dataset = load_from_disk(str(local_dataset_path))
            print("✓ Loaded from cache")
        else:
            print("Downloading Mini-ImageNet from HuggingFace (this may take a few minutes)...")
            # Using timm/mini-imagenet - verified working dataset
            self.hf_dataset = load_dataset(
                "timm/mini-imagenet",
                split="train",
                cache_dir=str(self.data_dir / "hf_cache")
            )
            print("✓ Successfully downloaded")

            # Save to disk for future use
            print(f"Saving dataset to {local_dataset_path}...")
            self.hf_dataset.save_to_disk(str(local_dataset_path))
            print("✓ Dataset cached locally")

        # Set class information
        self.num_classes = 100
        self.class_names = self._load_class_names()

        # Create examples from HF dataset (before split filtering)
        self.examples = []
        for idx, item in enumerate(self.hf_dataset):
            # Mini-ImageNet typically has 'label' field
            label = item['label']

            example = ClassificationExample(
                index=len(self.examples),
                image_path=f"hf_dataset_idx_{idx}",  # Placeholder
                label=label,
                label_name=self.class_names[label] if label < len(self.class_names) else f"class_{label}",
                split='',  # Will be set by parent class
                _hf_index=idx,  # Store original HF index for later retrieval
            )

            self.examples.append(example)

        print(f"✓ Loaded {len(self.examples)} total examples")
        print(f"✓ Images per class: ~{len(self.examples) // self.num_classes}")

    def _load_class_names(self):
        """Load human-readable class names from HF dataset."""
        # Try to get class names from HuggingFace dataset features
        if self.hf_dataset is not None:
            if hasattr(self.hf_dataset, 'features') and 'label' in self.hf_dataset.features:
                feature = self.hf_dataset.features['label']
                if hasattr(feature, 'names'):
                    return feature.names

        # Fallback: Mini-ImageNet canonical class names (ImageNet synsets)
        # These are the 100 classes commonly used in Mini-ImageNet
        return [
            'n01532829',  # house_finch
            'n01558993',  # robin
            'n01704323',  # triceratops
            'n01749939',  # green_mamba
            'n01770081',  # harvestman
            'n01843383',  # toucan
            'n01910747',  # jellyfish
            'n01930112',  # nematode
            'n01981276',  # king_crab
            'n02074367',  # dugong
            'n02089867',  # Walker_hound
            'n02091831',  # Saluki
            'n02101006',  # Gordon_setter
            'n02105505',  # komondor
            'n02108089',  # boxer
            'n02108551',  # Tibetan_mastiff
            'n02108915',  # French_bulldog
            'n02110063',  # malamute
            'n02110341',  # dalmatian
            'n02111277',  # Newfoundland
            'n02113712',  # miniature_poodle
            'n02114548',  # white_wolf
            'n02116738',  # African_hunting_dog
            'n02120079',  # Arctic_fox
            'n02129165',  # lion
            'n02138441',  # meerkat
            'n02165456',  # ladybug
            'n02174001',  # rhinoceros_beetle
            'n02219486',  # ant
            'n02443484',  # black-footed_ferret
            'n02606052',  # rock_beauty
            'n02687172',  # aircraft_carrier
            'n02747177',  # ashcan
            'n02795169',  # barrel
            'n02823428',  # beer_bottle
            'n02871525',  # bookshop
            'n02950826',  # cannon
            'n02971356',  # carton
            'n02981792',  # catamaran
            'n03017168',  # chiffonier
            'n03075370',  # combination_lock
            'n03146219',  # cuirass
            'n03207743',  # dishrag
            'n03272010',  # electric_guitar
            'n03337140',  # file
            'n03347037',  # fire_screen
            'n03400231',  # frying_pan
            'n03476684',  # go-kart
            'n03527444',  # hourglass
            'n03535780',  # horizontal_bar
            'n03544143',  # hourglass
            'n03584254',  # iPod
            'n03633091',  # ladle
            'n03676483',  # lipstick
            'n03770439',  # miniskirt
            'n03773504',  # missile
            'n03775546',  # mixing_bowl
            'n03838899',  # oboe
            'n03854065',  # organ
            'n03888257',  # parachute
            'n03908618',  # pencil_box
            'n03930313',  # pick
            'n03977966',  # police_van
            'n04067472',  # reel
            'n04146614',  # school_bus
            'n04149813',  # scoreboard
            'n04243546',  # slot
            'n04251144',  # snorkel
            'n04258138',  # solar_dish
            'n04398044',  # teapot
            'n04435653',  # tile_roof
            'n04443257',  # tobacco_shop
            'n04509417',  # unicycle
            'n04515003',  # upright
            'n04522168',  # vase
            'n04596742',  # wok
            'n04604644',  # worm_fence
            'n06794110',  # street_sign
            'n07584110',  # consomme
            'n07613480',  # trifle
            'n07697537',  # hotdog
            'n07747607',  # orange
            'n09246464',  # cliff
            'n09256479',  # coral_reef
            'n09332890',  # lakeside
            'n09428293',  # seashore
            'n12267677',  # acorn
            'n12620546',  # hip
            'n13133613',  # ear
            'n01440764',  # tench
            'n02110185',  # Siberian_husky
            'n02110627',  # affenpinscher
            'n02111129',  # Leonberg
            'n02112137',  # chow
            'n02113186',  # Cardigan
            'n02113799',  # standard_poodle
            'n02123045',  # tabby
            'n02123394',  # Persian_cat
            'n02129604',  # tiger
            'n02981792',  # catamaran
        ]

    def __getitem__(self, idx: int) -> Tuple[ClassificationExample, Image.Image]:
        """
        Get example and image.

        Returns:
            (ClassificationExample, PIL.Image)
        """
        example = self.examples[idx]

        # Load image from HuggingFace dataset using stored index
        if self.hf_dataset is None or example._hf_index is None:
            raise ValueError("HuggingFace dataset not loaded")

        hf_item = self.hf_dataset[example._hf_index]

        # Mini-ImageNet typically uses 'image' or 'img' key
        if 'image' in hf_item:
            image = hf_item['image']
        elif 'img' in hf_item:
            image = hf_item['img']
        else:
            raise ValueError(f"Could not find image in dataset item. Available keys: {list(hf_item.keys())}")

        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return example, image
