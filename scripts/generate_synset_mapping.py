"""Generate mapping from ImageNet synset IDs to human-readable names."""

import sys
from pathlib import Path
sys.path.insert(0, "src")

from data.mini_imagenet import MiniImageNetDataset
from nltk.corpus import wordnet as wn

def get_readable_name(synset_id: str) -> str:
    """Get human-readable name from ImageNet synset ID."""
    try:
        # ImageNet format is 'n########' where ######## is the WordNet offset
        offset = int(synset_id[1:])
        synset = wn.synset_from_pos_and_offset('n', offset)

        # Get the first lemma (most common name)
        lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
        return lemmas[0] if lemmas else synset_id
    except:
        return synset_id


def main():
    # Load all splits to get all synsets
    all_synsets = set()
    for split in ["train", "val", "test"]:
        dataset = MiniImageNetDataset(split=split)
        all_synsets.update(ex.label_name for ex in dataset.examples)

    all_synsets = sorted(all_synsets)

    print(f"Found {len(all_synsets)} unique synsets in Mini-ImageNet")
    print("\nGenerating mapping...")

    mapping = {}
    for synset_id in all_synsets:
        readable_name = get_readable_name(synset_id)
        mapping[synset_id] = readable_name
        print(f"  {synset_id} -> {readable_name}")

    # Write to Python file
    output_path = Path("src/utils/imagenet_names.py")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('"""\n')
        f.write('Mapping of ImageNet synset IDs to human-readable class names.\n')
        f.write('Auto-generated from WordNet.\n')
        f.write('"""\n\n')
        f.write('# Mapping from ImageNet synset ID to readable name\n')
        f.write('IMAGENET_SYNSET_TO_NAME = {\n')
        for synset_id, name in sorted(mapping.items()):
            f.write(f'    {repr(synset_id)}: {repr(name)},\n')
        f.write('}\n\n')
        f.write('# Reverse mapping\n')
        f.write('IMAGENET_NAME_TO_SYNSET = {v: k for k, v in IMAGENET_SYNSET_TO_NAME.items()}\n\n')
        f.write('def get_readable_name(synset_id: str) -> str:\n')
        f.write('    """Get human-readable name for synset ID."""\n')
        f.write('    return IMAGENET_SYNSET_TO_NAME.get(synset_id, synset_id)\n\n')
        f.write('def get_synset_id(readable_name: str) -> str:\n')
        f.write('    """Get synset ID from readable name."""\n')
        f.write('    return IMAGENET_NAME_TO_SYNSET.get(readable_name.lower(), readable_name)\n')

    print(f"\n✓ Wrote mapping to {output_path}")
    print(f"✓ {len(mapping)} synsets mapped")


if __name__ == "__main__":
    main()
