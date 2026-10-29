"""
Safely prepare a cleaned five-class damage dataset.
- Never modifies the original source dataset or existing split folders.
- Removes exact duplicates within the same class (keeps one copy).
- Excludes exact image files that occur under conflicting class labels.
- Creates a fresh grouped 70/15/15 split with no exact duplicate crossing splits.
Run from the project root:
    python ai_model\prepare_clean_damage_dataset.py
"""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import random
import shutil
import json

SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE = Path(r"C:\Users\Gauri\Downloads\Structural_Damage_Dataset\combined_dataset")
OUT = Path(__file__).resolve().parent / "cleaned_damage_dataset"
SPLIT_OUT = Path(__file__).resolve().parent / "damage_dataset_split_clean"
REPORT = Path(__file__).resolve().parent / "clean_dataset_report.json"
CLASSES = ["Crack", "Non-Crack", "Road_Pothole", "Road_Surface_Erosion", "Rust_Corrosion"]

def image_files(folder):
    return sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSIONS)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"Source dataset not found: {SOURCE}")
    missing = [c for c in CLASSES if not (SOURCE / c).is_dir()]
    if missing:
        raise SystemExit(f"Missing expected class folders: {missing}")

    # Scan source dataset, not the previously generated split folders.
    groups = defaultdict(list)
    for cls in CLASSES:
        for path in image_files(SOURCE / cls):
            groups[digest(path)].append((cls, path))

    conflicting = {h: items for h, items in groups.items()
                   if len({cls for cls, _ in items}) > 1}
    # Keep only one exact copy per class; omit any hash with conflicting labels.
    chosen = []
    duplicate_copies_removed = 0
    for h, items in groups.items():
        if h in conflicting:
            continue
        cls = items[0][0]
        same_class_items = sorted(path for label, path in items if label == cls)
        chosen.append((cls, same_class_items[0]))
        duplicate_copies_removed += max(0, len(same_class_items) - 1)

    if OUT.exists() or SPLIT_OUT.exists():
        raise SystemExit(
            "Output already exists; nothing was changed.\n"
            f"Please rename or back up these folders first if you want to rerun:\n{OUT}\n{SPLIT_OUT}"
        )

    OUT.mkdir(parents=True)
    for cls in CLASSES:
        (OUT / cls).mkdir()
    # Copy unique, non-conflicting images with stable names.
    counts = Counter()
    for cls, src in chosen:
        dst = OUT / cls / src.name
        if dst.exists():
            dst = OUT / cls / f"{src.stem}_{digest(src)[:10]}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        counts[cls] += 1

    # Split independently per class after exact deduplication, using fixed seed.
    rng = random.Random(SEED)
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            (SPLIT_OUT / split / cls).mkdir(parents=True, exist_ok=True)

    split_counts = {split: Counter() for split in ("train", "validation", "test")}
    for cls in CLASSES:
        files = image_files(OUT / cls)
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        partitions = {
            "train": files[:n_train],
            "validation": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:],
        }
        for split, paths in partitions.items():
            for src in paths:
                shutil.copy2(src, SPLIT_OUT / split / src.name)
                split_counts[split][cls] += 1

    report = {
        "source": str(SOURCE),
        "cleaned_dataset": str(OUT),
        "clean_split": str(SPLIT_OUT),
        "classes": CLASSES,
        "source_exact_duplicate_hash_groups": sum(len(v) > 1 for v in groups.values()),
        "source_conflicting_label_hash_groups_excluded": len(conflicting),
        "source_files_in_conflicting_groups_excluded": sum(len(v) for v in conflicting.values()),
        "same_class_duplicate_copies_removed": duplicate_copies_removed,
        "cleaned_unique_images_per_class": dict(counts),
        "split_counts": {s: dict(c) for s, c in split_counts.items()},
        "seed": SEED,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Preparation completed. Original data and existing splits were not modified.")
    print(f"Clean dataset: {OUT}")
    print(f"New split:     {SPLIT_OUT}")
    print(f"Report:        {REPORT}")
    print(f"Conflicting duplicate groups excluded: {len(conflicting)}")
    print(f"Same-class duplicate copies removed: {duplicate_copies_removed}")
    print("Images per class in cleaned dataset:")
    for cls in CLASSES:
        print(f"  {cls}: {counts[cls]}")
    print("Split totals:")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {sum(split_counts[split].values())}")

if __name__ == "__main__":
    main()
