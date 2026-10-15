import random
import shutil
from pathlib import Path

random.seed(42)

sources = {
    "Negative": Path(r"C:\Users\Gauri\Downloads\5y9wdsg2zt-2\Negative"),
    "Positive": Path(r"C:\Users\Gauri\Downloads\5y9wdsg2zt-2\Positive"),
}

output = Path(__file__).parent / "dataset_split"
splits = {"train": 0.70, "validation": 0.15, "test": 0.15}
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

if output.exists():
    raise FileExistsError(
        f"{output} already exists. Rename or remove it before creating a new split."
    )

for label, source in sources.items():
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    images = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise ValueError(f"No images found in {source}")

    random.shuffle(images)

    train_end = int(len(images) * 0.70)
    val_end = train_end + int(len(images) * 0.15)

    groups = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, files in groups.items():
        destination = output / split / label
        destination.mkdir(parents=True, exist_ok=True)

        for image in files:
            shutil.copy2(image, destination / image.name)

        print(f"{split}/{label}: {len(files)} images")

print(f"\nDataset split complete: {output}")