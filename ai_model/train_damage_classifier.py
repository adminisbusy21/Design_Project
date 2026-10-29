
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).resolve().parent

# Change this only if your actual dataset is elsewhere.
DATASET_DIR = Path(
    r"C:\Users\Gauri\Downloads\Structural_Damage_Dataset\combined_dataset"
)

# This script rebuilds this generated folder when executed.
SPLIT_DIR = BASE_DIR / "damage_dataset_split"

MODEL_PATH = BASE_DIR / "resnet18_damage_classifier.pth"

METRICS_PATH = BASE_DIR / "damage_evaluation_metrics.json"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ==================================================
# 2. VALIDATE THE SOURCE DATASET
# ==================================================

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}\n"
        "Check DATASET_DIR before running training."
    )

# ImageFolder uses alphabetical folder order.
# Keep inference labels consistent with this mapping.
EXPECTED_CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

actual_classes = sorted(
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

if actual_classes != EXPECTED_CLASSES:
    raise ValueError(
        "\nDataset class folders do not match the expected classes.\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found:    {actual_classes}\n"
        "Check the folder names before training."
    )

class_files = {}

print("\nSource dataset counts:")

for class_name in EXPECTED_CLASSES:
    class_dir = DATASET_DIR / class_name

    files = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    # Verify that the images can actually be opened.
    valid_files = []

    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()

            valid_files.append(path)

        except Exception:
            print(f"WARNING: Unreadable image: {path}")

    class_files[class_name] = valid_files

    print(f"{class_name}: {len(valid_files)} valid images")

    if len(valid_files) < 10:
        raise ValueError(
            f"{class_name} has fewer than 10 valid images. "
            "Check the dataset before training."
        )


# ==================================================
# 3. REBUILD TRAIN / VALIDATION / TEST SPLITS
# ==================================================

print("\nPreparing fresh dataset splits...")

# Only the generated damage_dataset_split directory
# is removed. The source dataset is never deleted.
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

for class_name in EXPECTED_CLASSES:

    files = class_files[class_name]

    # 70% training, 15% validation, 15% testing.
    train_files, temporary_files = train_test_split(
        files,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
    )

    validation_files, test_files = train_test_split(
        temporary_files,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    split_groups = {
        "train": train_files,
        "validation": validation_files,
        "test": test_files,
    }

    for split_name, selected_files in split_groups.items():

        destination_dir = (
            SPLIT_DIR / split_name / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, source_file in enumerate(selected_files):

            # Unique names prevent filename collisions.
            destination_file = (
                destination_dir
                / f"{index:05d}_{source_file.name}"
            )

            shutil.copy2(
                source_file,
                destination_file,
            )

        print(
            f"{split_name}/{class_name}: "
            f"{len(selected_files)} images"
        )

print(f"\nFresh splits created at: {SPLIT_DIR}")


# ==================================================
# 4. IMAGE TRANSFORMS
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================================================
# 5. LOAD DATASETS
# ==================================================

train_dataset = datasets.ImageFolder(
    SPLIT_DIR / "train",
    transform=train_transform,
)

validation_dataset = datasets.ImageFolder(
    SPLIT_DIR / "validation",
    transform=evaluation_transform,
)

test_dataset = datasets.ImageFolder(
    SPLIT_DIR / "test",
    transform=evaluation_transform,
)

print("\nActual class mapping:")

for class_name, index in train_dataset.class_to_idx.items():
    print(f"{index}: {class_name}")

# Check that all three splits use exactly the same mapping.
if not (
    train_dataset.class_to_idx
    == validation_dataset.class_to_idx
    == test_dataset.class_to_idx
):
    raise ValueError(
        "Class mappings differ between dataset splits."
    )

if train_dataset.classes != EXPECTED_CLASSES:
    raise ValueError(
        "ImageFolder class order does not match "
        "the expected inference class order."
    )

print(
    "\nDataset sizes:"
    f"\nTraining:   {len(train_dataset)}"
    f"\nValidation: {len(validation_dataset)}"
    f"\nTesting:    {len(test_dataset)}"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==================================================
# 6. INITIALIZE RESNET18
# ==================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    len(EXPECTED_CLASSES),
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ==================================================
# 7. TRAIN AND VALIDATE
# ==================================================

best_validation_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    training_loss_sum = 0.0
    training_correct = 0
    training_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        training_loss_sum += (
            loss.item() * images.size(0)
        )

        training_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        training_total += labels.size(0)

    training_loss = (
        training_loss_sum / training_total
    )

    training_accuracy = (
        training_correct / training_total
    )

    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    validation_loss_sum = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            validation_loss_sum += (
                loss.item() * images.size(0)
            )

            validation_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            validation_total += labels.size(0)

    validation_loss = (
        validation_loss_sum / validation_total
    )

    validation_accuracy = (
        validation_correct / validation_total
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f"\nTraining loss:      {training_loss:.4f}"
        f"\nTraining accuracy:  {training_accuracy:.4f}"
        f"\nValidation loss:    {validation_loss:.4f}"
        f"\nValidation accuracy:{validation_accuracy:.4f}"
    )

    # Save the state with the lowest validation loss.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


# ==================================================
# 8. RESTORE BEST MODEL AND SAVE WEIGHTS
# ==================================================

if best_state is None:
    raise RuntimeError(
        "Training did not produce a valid model."
    )

model.load_state_dict(best_state)

model = model.to(DEVICE)
model.eval()

torch.save(
    model.state_dict(),
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")


# ==================================================
# 9. EVALUATE ON THE HELD-OUT TEST SET
# ==================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.tolist()
        )

test_accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

test_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

test_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(EXPECTED_CLASSES))),
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(EXPECTED_CLASSES))),
    target_names=EXPECTED_CLASSES,
    zero_division=0,
    output_dict=True,
)

metrics = {
    "classes": EXPECTED_CLASSES,
    "class_to_idx": train_dataset.class_to_idx,
    "accuracy": test_accuracy,
    "precision_macro": test_precision,
    "recall_macro": test_recall,
    "f1_macro": test_f1,
    "confusion_matrix": matrix.tolist(),
    "classification_report": report,
    "training_epochs": EPOCHS,
    "best_validation_loss": best_validation_loss,
    "test_images": len(test_dataset),
}

METRICS_PATH.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("\n========== TEST RESULTS ==========")

print(f"Accuracy:        {test_accuracy:.4f}")
print(f"Macro precision: {test_precision:.4f}")
print(f"Macro recall:    {test_recall:.4f}")
print(f"Macro F1-score:  {test_f1:.4f}")

print("\nClassification report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(EXPECTED_CLASSES))),
        target_names=EXPECTED_CLASSES,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(matrix)

print(f"\nMetrics saved to: {METRICS_PATH}")

