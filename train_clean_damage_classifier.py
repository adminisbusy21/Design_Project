from pathlib import Path
import json
import random
import copy
import shutil

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

CLEANED_DIR = ROOT / "cleaned_damage_dataset"
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

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================================================
# HELPER FUNCTION
# =====================================================

def get_images(folder):
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )


# =====================================================
# CREATE A NEW TRAIN / VALIDATION / TEST SPLIT
# =====================================================

def prepare_split():

    if not CLEANED_DIR.is_dir():
        raise SystemExit(
            f"Cleaned dataset not found: {CLEANED_DIR}"
        )

    # Verify that every class contains images.
    for class_name in CLASSES:

        class_folder = CLEANED_DIR / class_name

        if not class_folder.is_dir():
            raise SystemExit(
                f"Missing class folder: {class_folder}"
            )

        if not get_images(class_folder):
            raise SystemExit(
                f"No images found in: {class_folder}"
            )

    # Never overwrite an existing split.
    if SPLIT_DIR.exists():
        raise SystemExit(
            f"Split folder already exists: {SPLIT_DIR}\n"
            "Nothing was changed. Rename or back up that folder "
            "before running again."
        )

    rng = random.Random(SEED)

    for split_name in ["train", "validation", "test"]:

        for class_name in CLASSES:

            folder = SPLIT_DIR / split_name / class_name
            folder.mkdir(parents=True, exist_ok=True)

    totals = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for class_name in CLASSES:

        images = get_images(CLEANED_DIR / class_name)
        rng.shuffle(images)

        total = len(images)

        train_end = int(total * 0.70)
        val_end = train_end + int(total * 0.15)

        split_images = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, selected_images in split_images.items():

            for source in selected_images:

                destination = (
                    SPLIT_DIR
                    / split_name
                    / class_name
                    / source.name
                )

                # Avoid overwriting files if names collide.
                if destination.exists():

                    destination = destination.with_name(
                        f"{source.stem}_{len(list(destination.parent.iterdir()))}"
                        f"{source.suffix.lower()}"
                    )

                shutil.copy2(source, destination)
                totals[split_name] += 1

    print("\nNew dataset split created successfully.")
    print("Training images:", totals["train"])
    print("Validation images:", totals["validation"])
    print("Testing images:", totals["test"])


# =====================================================
# TRAIN MODEL
# =====================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Create a fresh, correctly structured split.
    prepare_split()

    # Training transformations.
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

    # Validation and testing transformations.
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

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

    # Verify class ordering.
    if train_dataset.classes != CLASSES:
        raise SystemExit(
            f"Unexpected class order: {train_dataset.classes}"
        )

    if (
        validation_dataset.classes != CLASSES
        or test_dataset.classes != CLASSES
    ):
        raise SystemExit(
            "Class mappings differ between dataset splits."
        )

    # Select GPU if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Classes:", CLASSES)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Testing images:", len(test_dataset))

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

    # Load pretrained ResNet18.
    try:

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

    except Exception as error:

        print("\nCould not load pretrained weights:", error)
        print("Using randomly initialized ResNet18.")

        model = models.resnet18(weights=None)

    # Replace final layer for five classes.
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

    # =================================================
    # TRAINING LOOP
    # =================================================

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

            total_training_loss += (
                loss.item() * images.size(0)
            )

        average_training_loss = (
            total_training_loss / len(train_dataset)
        )

        # Validate model.
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

        # Keep the model with the lowest validation loss.
        if average_validation_loss < best_validation_loss:

            best_validation_loss = average_validation_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    # =================================================
    # SAVE NEW MODEL
    # =================================================

    model.load_state_dict(best_model_state)

    AI_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print("\nNew model saved to:")
    print(MODEL_PATH)

    # =================================================
    # FINAL TEST EVALUATION
    # =================================================

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

            true_labels.extend(
                labels.tolist()
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASSES))),
    )

    report = classification_report(
        true_labels,
        predicted_labels,
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
        "model_path": str(MODEL_PATH),
        "split_path": str(SPLIT_DIR),
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

    print("\nMetrics saved to:")
    print(METRICS_PATH)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()