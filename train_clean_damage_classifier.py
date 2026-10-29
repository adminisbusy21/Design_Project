
"""
Train and evaluate the five-class structural damage classifier.

Expected dataset structure:
    cleaned_damage_dataset/
        <class_name>/

    damage_dataset_split_clean_v2/
        train/<class_name>/
        val/<class_name>/
        test/<class_name>/

Run from the project root:
    python train_clean_damage_classifier.py

By default, an existing model checkpoint will not be overwritten.
To explicitly permit replacement in PowerShell:
    $env:ALLOW_MODEL_OVERWRITE = "1"
    python train_clean_damage_classifier.py
    Remove-Item Env:ALLOW_MODEL_OVERWRITE
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

ROOT = Path(__file__).resolve().parent

SPLIT_DIR = ROOT / "damage_dataset_split_clean_v2"
AI_DIR = ROOT / "ai_model"

MODEL_PATH = AI_DIR / "resnet18_damage_classifier_clean.pth"
METRICS_PATH = AI_DIR / "damage_evaluation_metrics_clean.json"

CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================================================
# REPRODUCIBILITY
# =====================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================
# DATASET VALIDATION
# =====================================================

def validate_split() -> None:
    """Check the prepared dataset split before loading it."""
    if not SPLIT_DIR.is_dir():
        raise SystemExit(
            f"Prepared dataset split not found:\n{SPLIT_DIR}\n\n"
            "Prepare the dataset first. Training has not started."
        )

    for split_name, folder_name in SPLITS.items():
        split_path = SPLIT_DIR / folder_name

        if not split_path.is_dir():
            raise SystemExit(
                f"Missing {split_name} split folder:\n{split_path}"
            )

        for class_name in CLASSES:
            class_path = split_path / class_name

            if not class_path.is_dir():
                raise SystemExit(
                    f"Missing class folder:\n{class_path}"
                )

            image_count = sum(
                1
                for path in class_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in EXTENSIONS
            )

            if image_count == 0:
                raise SystemExit(
                    f"No supported images found in:\n{class_path}"
                )


def load_dataset(
    directory: Path,
    transform: transforms.Compose,
    expected_classes: list[str],
) -> datasets.ImageFolder:
    """Load an ImageFolder dataset and verify class mapping."""
    dataset = datasets.ImageFolder(
        str(directory),
        transform=transform,
    )

    if dataset.classes != expected_classes:
        raise RuntimeError(
            f"Unexpected class mapping in {directory}.\n"
            f"Expected: {expected_classes}\n"
            f"Found:    {dataset.classes}"
        )

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {directory}")

    return dataset


# =====================================================
# TRAINING
# =====================================================

def main() -> None:
    set_seed(SEED)
    validate_split()

    allow_overwrite = (
        __import__("os").environ.get("ALLOW_MODEL_OVERWRITE", "").strip()
        == "1"
    )

    if MODEL_PATH.exists() and not allow_overwrite:
        raise SystemExit(
            f"Existing checkpoint protected:\n{MODEL_PATH}\n\n"
            "Training has not started, and the checkpoint was not changed.\n"
            "If you intentionally want to replace it, set "
            "ALLOW_MODEL_OVERWRITE=1 in your environment."
        )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = load_dataset(
        SPLIT_DIR / SPLITS["train"],
        train_transform,
        CLASSES,
    )

    validation_dataset = load_dataset(
        SPLIT_DIR / SPLITS["validation"],
        evaluation_transform,
        CLASSES,
    )

    test_dataset = load_dataset(
        SPLIT_DIR / SPLITS["test"],
        evaluation_transform,
        CLASSES,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Classes: {CLASSES}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

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

    # Do not silently fall back to randomly initialized weights.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as error:
        raise RuntimeError(
            "Could not load pretrained ResNet18 weights. "
            "Check internet access or the local pretrained-weight cache. "
            "Training was stopped to avoid silently changing the "
            "training configuration."
        ) from error

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_training_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_training_loss += loss.item() * images.size(0)

        average_training_loss = (
            total_training_loss / len(train_dataset)
        )

        model.eval()
        total_validation_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_validation_loss += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(dim=1)

                correct_predictions += (
                    predictions == labels
                ).sum().item()

        average_validation_loss = (
            total_validation_loss / len(validation_dataset)
        )

        validation_accuracy = (
            correct_predictions / len(validation_dataset)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {average_training_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}",
            flush=True,
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # =================================================
    # SAVE CHECKPOINT
    # =================================================

    model.load_state_dict(best_model_state)
    model.eval()

    AI_DIR.mkdir(parents=True, exist_ok=True)

    # Save to a temporary file first, then replace the destination.
    temporary_model_path = MODEL_PATH.with_suffix(".tmp")

    try:
        torch.save(model.state_dict(), temporary_model_path)
        temporary_model_path.replace(MODEL_PATH)
    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    print(f"\nModel saved to: {MODEL_PATH}")

    # =================================================
    # FINAL TEST EVALUATION
    # =================================================

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            predicted_labels.extend(predictions.cpu().tolist())
            true_labels.extend(labels.tolist())

    label_indices = list(range(len(CLASSES)))

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_indices,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "classes": CLASSES,
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "test_images": len(test_dataset),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "split_path": str(SPLIT_DIR.relative_to(ROOT)),
        "random_seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "pretrained_weights": "ResNet18_Weights.DEFAULT",
        "device_used": str(device),
    }

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n========== FINAL RESULTS ==========")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"\nMetrics saved to: {METRICS_PATH}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
