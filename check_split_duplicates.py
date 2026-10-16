"""
Check for byte-identical image files across train/validation/test splits.

Run from the repository root:
    python check_split_duplicates.py

This only detects exact byte-for-byte duplicates. It does not detect near-duplicates,
re-encoded copies, or images derived from the same original concrete surface.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

DATA_DIR = Path("ai_model/dataset_split_deduplicated")
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def digest_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Dataset split folder not found: {DATA_DIR.resolve()}")

    digest_locations = defaultdict(list)
    counts = defaultdict(int)

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"WARNING: missing split folder: {split_dir}")
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                counts[split] += 1
                digest_locations[digest_file(path)].append((split, path))

    print("Image counts:")
    for split in SPLITS:
        print(f"  {split}: {counts[split]}")

    cross_split_groups = []
    for digest, locations in digest_locations.items():
        splits_present = {split for split, _ in locations}
        if len(splits_present) > 1:
            cross_split_groups.append((digest, locations))

    if not cross_split_groups:
        print("\nNo byte-identical image files were found across different splits.")
        print("Note: near-duplicates/source-surface leakage are NOT ruled out by this check.")
        return

    print(f"\nFOUND {len(cross_split_groups)} exact-duplicate hash group(s) across splits:")
    for index, (digest, locations) in enumerate(cross_split_groups, start=1):
        print(f"\nGroup {index} SHA256={digest}")
        for split, path in locations:
            print(f"  [{split}] {path}")

    print("\nRecommendation: rebuild splits so identical/group-related images stay in one split.")


if __name__ == "__main__":
    main()
