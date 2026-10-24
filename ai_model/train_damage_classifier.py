
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).parent
DATASET_DIR = Path(
    r"C:\Users\Gauri\Downloads\Structural_Damage_Dataset\combined_dataset"
)

SPLIT_DIR = BASE_DIR / "damage_dataset_split"
MODEL_PATH = BASE_DIR / "resnet18_damage_classifier.pth"
METRICS_PATH = BASE_DIR / "damage_evaluation_metrics.json"

CLASSES = [
    "Crack",
    "Non-Crack",
    "Rust_Corrosion",
    "Road_Pothole",
    "Road_Surface_Erosion",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --------------------------------------------------
# Validate dataset
# --------------------------------------------------
if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

for class_name in CLASSES:
    folder = DATASET_DIR / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")

    count = sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"{class_name}: {count} images")

    if count < 10:
        raise ValueError(
            f"{class_name} has only {count} images. "
            "At least 10 are required for this initial experiment."
        )

# --------------------------------------------------
# Create reproducible train/validation/test splits
# --------------------------------------------------
# Existing split folder is reused to prevent accidental overwrites.
# Delete it manually only if you intentionally want to rebuild the split.
if not SPLIT_DIR.exists():
    print("\nCreating dataset splits...")

    for class_name in CLASSES:
        source_dir = DATASET_DIR / class_name
        files = sorted(
            p for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        train_files, temp_files = train_test_split(
            files,
            test_size=0.30,
            random_state=SEED,
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.50,
            random_state=SEED,
        )

        for split_name, split_files in [
            ("train", train_files),
            ("validation", val_files),
            ("test", test_files),
        ]:
            destination = SPLIT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for index, source in enumerate(split_files):
                # Unique names avoid collisions across source folders.
                destination_file = destination / f"{index:05d}_{source.name}"
                shutil.copy2(source, destination_file)

    print(f"Dataset splits created at: {SPLIT_DIR}")
else:
    print(f"\nUsing existing split folder: {SPLIT_DIR}")

# --------------------------------------------------
# Image transformations
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = datasets.ImageFolder(
    SPLIT_DIR / "train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    SPLIT_DIR / "validation", transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    SPLIT_DIR / "test", transform=eval_transform
)

print("\nClass mapping:", train_dataset.class_to_idx)
print(
    f"Images: train={len(train_dataset)}, "
    f"validation={len(val_dataset)}, test={len(test_dataset)}"
)

if not (
    train_dataset.class_to_idx
    == val_dataset.class_to_idx
    == test_dataset.class_to_idx
):
    raise ValueError("Class mappings differ between splits.")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# --------------------------------------------------
# Load pretrained ResNet18
# --------------------------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
best_state = None

# --------------------------------------------------
# Train and validate
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_accuracy:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

# --------------------------------------------------
# Save best model and evaluate test set
# --------------------------------------------------
if best_state is None:
    raise RuntimeError("Training did not produce a valid model.")

model.load_state_dict(best_state)
model.eval()
torch.save(model.state_dict(), MODEL_PATH)

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        all_predictions.extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

metrics = {
    "classes": test_dataset.classes,
    "accuracy": accuracy_score(all_labels, all_predictions),
    "precision_macro": precision_score(
        all_labels, all_predictions, average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        all_labels, all_predictions, average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        all_labels, all_predictions
    ).tolist(),
    "classification_report": classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        zero_division=0,
        output_dict=True,
    ),
}

METRICS_PATH.write_text(json.dumps(metrics, indent=2))

print("\nTest results:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    zero_division=0,
))
print("Confusion matrix:")
print(metrics["confusion_matrix"])
print(f"Model saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")
