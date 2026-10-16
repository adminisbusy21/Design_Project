import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42

SOURCES = {
    "Negative": Path(
        r"C:\Users\Gauri\Downloads\5y9wdsg2zt-2\Negative"
    ),
    "Positive": Path(
        r"C:\Users\Gauri\Downloads\5y9wdsg2zt-2\Positive"
    ),
}

OUTPUT = Path(__file__).parent / "dataset_split_deduplicated"

SPLITS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename it before rerunning."
        )

    random.seed(SEED)

    for label, source in SOURCES.items():
        if not source.is_dir():
            raise FileNotFoundError(
                f"Source folder not found: {source}"
            )

        images = sorted(
            path for path in source.iterdir()
            if path.is_file()
            and path.suffix.lower() in EXTENSIONS
        )

        if not images:
            raise ValueError(f"No images found in {source}")

        # Group byte-identical images so they cannot cross splits.
        groups_by_hash = defaultdict(list)

        for image in images:
            groups_by_hash[sha256_file(image)].append(image)

        groups = list(groups_by_hash.values())
        random.shuffle(groups)

        total = len(images)
        targets = {
            "train": round(total * SPLITS["train"]),
            "validation": round(total * SPLITS["validation"]),
        }

        assigned = {
            "train": [],
            "validation": [],
            "test": [],
        }

        counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        for group in groups:
            # Assign whole groups, never individual duplicates.
            if counts["train"] < targets["train"]:
                split = "train"
            elif counts["validation"] < targets["validation"]:
                split = "validation"
            else:
                split = "test"

            assigned[split].extend(group)
            counts[split] += len(group)

        for split, files in assigned.items():
            destination = OUTPUT / split / label
            destination.mkdir(parents=True, exist_ok=True)

            for image in files:
                shutil.copy2(image, destination / image.name)

            print(f"{split}/{label}: {len(files)} images")

        print(
            f"{label}: {len(images)} files, "
            f"{len(groups)} unique exact-hash groups"
        )

    print(f"\nDeduplicated split created at: {OUTPUT}")
    print("Original dataset and previous split were not modified.")


if __name__ == "__main__":
    main()