import json
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
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

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths and settings
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset_split"
MODEL_PATH = BASE_DIR / "resnet18_crack_classifier.pth"
METRICS_PATH = BASE_DIR / "evaluation_metrics.json"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Image preprocessing
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

# Load datasets
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation", transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    DATA_DIR / "test", transform=eval_transform
)

print("Class mapping:", train_dataset.class_to_idx)
print(
    f"Images: train={len(train_dataset)}, "
    f"validation={len(val_dataset)}, test={len(test_dataset)}"
)

if not (
    train_dataset.class_to_idx
    == val_dataset.class_to_idx
    == test_dataset.class_to_idx
):
    raise ValueError("Class mappings differ between dataset splits.")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Load pretrained ResNet18
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    print("Loaded pretrained ResNet18 weights.")
except Exception as error:
    raise RuntimeError(
        "Could not download pretrained ResNet18 weights. "
        "Check your internet connection and try again."
    ) from error

# Replace final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

best_val_loss = float("inf")

# Training and validation
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    model.eval()
    val_loss_sum = 0.0
    val_total = 0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Validation loss: {val_loss:.4f} | "
        f"Validation accuracy: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved best model.")

# Load best model for final test evaluation
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

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

print("\nFinal test results:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    zero_division=0,
))
print("Confusion matrix:")
print(metrics["confusion_matrix"])
print(f"Saved model to: {MODEL_PATH}")
print(f"Saved metrics to: {METRICS_PATH}")