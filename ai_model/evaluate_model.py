import json
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

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset_split"
MODEL_PATH = BASE_DIR / "resnet18_crack_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("Using device:", DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))

state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model = model.to(DEVICE)
model.eval()

actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.cpu().tolist())

metrics = {
    "classes": test_dataset.classes,
    "accuracy": accuracy_score(actual, predicted),
    "precision_macro": precision_score(
        actual, predicted, average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        actual, predicted, average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        actual, predicted, average="macro", zero_division=0
    ),
    "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
}

report = classification_report(
    actual,
    predicted,
    target_names=test_dataset.classes,
    zero_division=0,
)

print("\nTEST RESULTS")
print(report)
print("Confusion matrix:", metrics["confusion_matrix"])

metrics_path = BASE_DIR / "evaluation_metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))

print("\nMetrics saved to:", metrics_path)