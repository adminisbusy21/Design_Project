
from pathlib import Path

DATASET_DIR = Path(
    r"C:\Users\Gauri\Downloads\Structural_Damage_Dataset\combined_dataset"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.exists():
    print("ERROR: Dataset folder not found!")
    print("Expected location:", DATASET_DIR)
else:
    total = 0

    for folder in sorted(DATASET_DIR.iterdir()):
        if folder.is_dir():
            count = sum(
                1 for file in folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in IMAGE_EXTENSIONS
            )
            print(f"{folder.name}: {count} images")
            total += count

    print(f"\nTOTAL IMAGES: {total}")
