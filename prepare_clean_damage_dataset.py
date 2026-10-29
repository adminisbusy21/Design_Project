import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SOURCE = Path(
    os.getenv(
        "SOURCE_DATASET_DIR",
        str(ROOT / "Structural_Damage_Dataset" / "combined_dataset"),
    )
).expanduser().resolve()

CLEANED_DIR = ROOT / "cleaned_damage_dataset"
SPLIT_DIR = ROOT / "damage_dataset_split_clean_v2"
REPORT_PATH = ROOT / "clean_dataset_report.json"

CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_images(class_dir: Path) -> list[Path]:
    """Find supported image files recursively in a class directory."""
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_configuration() -> None:
    """Validate the dataset location, class folders, and split ratios."""
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Source dataset directory was not found:\n{SOURCE}\n\n"
            "Place the dataset at that location or set SOURCE_DATASET_DIR "
            "to the correct combined_dataset folder."
        )

    missing_classes = [
        class_name
        for class_name in CLASSES
        if not (SOURCE / class_name).is_dir()
    ]

    if missing_classes:
        raise FileNotFoundError(
            "The source dataset is missing these class folders:\n"
            + "\n".join(f"- {name}" for name in missing_classes)
            + f"\n\nExpected source directory:\n{SOURCE}"
        )

    if not (
        0 < TRAIN_RATIO < 1
        and 0 < VAL_RATIO < 1
        and 0 < TEST_RATIO < 1
        and abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
    ):
        raise ValueError(
            "TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must be positive "
            "and sum to 1."
        )

    # Prevent accidental destruction of the source dataset.
    protected_paths = {SOURCE, ROOT, Path.home().resolve()}

    for output_path in (CLEANED_DIR, SPLIT_DIR):
        if output_path.resolve() in protected_paths:
            raise ValueError(
                f"Unsafe output path; refusing to overwrite: {output_path}"
            )


def clean_dataset() -> dict:
    """
    Copy valid unique images into class folders.

    Exact duplicate images with conflicting labels are excluded from every
    class to avoid training on contradictory labels. Exact duplicates within
    the same class are retained only once.
    """
    print(f"Source dataset: {SOURCE}")
    print("Scanning images and calculating SHA-256 hashes...")

    records = []
    hash_to_labels = defaultdict(set)
    hash_to_paths = defaultdict(list)
    unreadable_files = []

    for class_name in CLASSES:
        class_dir = SOURCE / class_name
        images = find_images(class_dir)

        print(f"  {class_name}: found {len(images)} candidate image(s)")

        for image_path in images:
            try:
                image_hash = sha256_file(image_path)
            except (OSError, PermissionError) as exc:
                unreadable_files.append(
                    {
                        "path": str(image_path.relative_to(SOURCE)),
                        "error": str(exc),
                    }
                )
                continue

            record = {
                "path": image_path,
                "class_name": class_name,
                "sha256": image_hash,
            }
            records.append(record)
            hash_to_labels[image_hash].add(class_name)
            hash_to_paths[image_hash].append(record)

    if not records:
        raise RuntimeError(
            f"No readable supported images were found in:\n{SOURCE}"
        )

    conflicting_hashes = {
        image_hash
        for image_hash, labels in hash_to_labels.items()
        if len(labels) > 1
    }

    kept_hashes = set()
    cleaned_counts = Counter()
    duplicate_same_class_count = 0
    conflicting_image_count = sum(
        len(hash_to_paths[image_hash])
        for image_hash in conflicting_hashes
    )

    # Start fresh only for generated output directories.
    # Never delete or modify SOURCE.
    for output_dir in (CLEANED_DIR, SPLIT_DIR):
        if output_dir.exists():
            shutil.rmtree(output_dir)

    for class_name in CLASSES:
        (CLEANED_DIR / class_name).mkdir(parents=True, exist_ok=True)

    # Keep the first occurrence of each hash within its original class.
    # If a hash occurs under multiple labels, exclude it entirely.
    for record in records:
        image_hash = record["sha256"]
        class_name = record["class_name"]

        if image_hash in conflicting_hashes:
            continue

        if image_hash in kept_hashes:
            duplicate_same_class_count += 1
            continue

        kept_hashes.add(image_hash)

        source_path = record["path"]
        destination_dir = CLEANED_DIR / class_name
        destination_path = destination_dir / source_path.name

        # Avoid overwriting distinct files that happen to share a filename.
        if destination_path.exists():
            destination_path = destination_dir / (
                f"{source_path.stem}_{image_hash[:12]}"
                f"{source_path.suffix.lower()}"
            )

        shutil.copy2(source_path, destination_path)
        cleaned_counts[class_name] += 1

    empty_classes = [
        class_name
        for class_name in CLASSES
        if cleaned_counts[class_name] == 0
    ]

    if empty_classes:
        raise RuntimeError(
            "No clean images remain for these classes: "
            + ", ".join(empty_classes)
            + ". Check the source data and duplicate-label conflicts."
        )

    report = {
        "source_directory": str(SOURCE),
        "cleaned_directory": str(CLEANED_DIR.relative_to(ROOT)),
        "split_directory": str(SPLIT_DIR.relative_to(ROOT)),
        "classes": CLASSES,
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "candidate_images_scanned": len(records) + len(unreadable_files),
        "readable_images_hashed": len(records),
        "unique_cleaned_images": sum(cleaned_counts.values()),
        "same_class_duplicate_images_removed": duplicate_same_class_count,
        "conflicting_label_images_excluded": conflicting_image_count,
        "conflicting_hash_count": len(conflicting_hashes),
        "unreadable_files": unreadable_files,
        "class_counts": dict(cleaned_counts),
        "splits": {},
    }

    print("\nCleaning summary:")
    print(f"  Readable images: {len(records)}")
    print(f"  Cleaned images: {sum(cleaned_counts.values())}")
    print(f"  Same-class duplicates removed: {duplicate_same_class_count}")
    print(f"  Conflicting-label images excluded: {conflicting_image_count}")
    print(f"  Unreadable files: {len(unreadable_files)}")

    return report


def create_splits(report: dict) -> None:
    """
    Create class-organized train, validation, and test splits.

    Splits are deterministic for the same cleaned files and seed. The split
    is image-level, not source-object-level; related photos of the same
    physical object could therefore appear in different splits.
    """
    rng = random.Random(RANDOM_SEED)

    split_names = ("train", "val", "test")

    for split_name in split_names:
        for class_name in CLASSES:
            (SPLIT_DIR / split_name / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )

    split_counts = {
        split_name: Counter()
        for split_name in split_names
    }

    for class_name in CLASSES:
        image_paths = find_images(CLEANED_DIR / class_name)
        rng.shuffle(image_paths)

        total = len(image_paths)

        if total < 3:
            raise ValueError(
                f"Class '{class_name}' has only {total} cleaned image(s). "
                "At least three are required to place images in train, "
                "validation, and test."
            )

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        # Ensure every split gets at least one image for each class.
        train_end = max(1, min(train_end, total - 2))
        val_end = max(train_end + 1, min(val_end, total - 1))

        split_images = {
            "train": image_paths[:train_end],
            "val": image_paths[train_end:val_end],
            "test": image_paths[val_end:],
        }

        for split_name, files in split_images.items():
            target_dir = SPLIT_DIR / split_name / class_name

            for source_path in files:
                destination_path = target_dir / source_path.name

                if destination_path.exists():
                    raise FileExistsError(
                        f"Duplicate destination filename: {destination_path}"
                    )

                shutil.copy2(source_path, destination_path)
                split_counts[split_name][class_name] += 1

        print(
            f"  {class_name}: "
            f"train={len(split_images['train'])}, "
            f"val={len(split_images['val'])}, "
            f"test={len(split_images['test'])}"
        )

    report["splits"] = {
        split_name: {
            "total": sum(counts.values()),
            "class_counts": dict(counts),
        }
        for split_name, counts in split_counts.items()
    }


def main() -> None:
    validate_configuration()

    report = clean_dataset()

    print("\nCreating train/validation/test splits:")
    create_splits(report)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nCompleted successfully.")
    print(f"Cleaned dataset: {CLEANED_DIR}")
    print(f"Dataset splits:  {SPLIT_DIR}")
    print(f"Report:          {REPORT_PATH}")
    print("\nImportant: this script creates image-level splits. If multiple")
    print("images show the same physical object or scene, group them before")
    print("splitting to avoid data leakage.")


if __name__ == "__main__":
    main()
