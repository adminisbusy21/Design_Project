
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "resnet18_damage_classifier.pth"

CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# This is a warning threshold, not a guarantee of correctness.
LOW_CONFIDENCE_THRESHOLD = 0.70


# ==================================================
# 2. CHECK MODEL FILE
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# ==================================================
# 3. IMAGE PREPROCESSING
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES),
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(
    state_dict,
    strict=True,
)

model = model.to(DEVICE)
model.eval()

print(f"Model loaded: {MODEL_PATH.name}")
print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")


# ==================================================
# 5. GET IMAGE PATH
# ==================================================

image_path_text = input(
    "\nEnter the full path of an image: "
).strip().strip('"')

image_path = Path(image_path_text)

if not image_path.is_file():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


# ==================================================
# 6. LOAD AND PREPROCESS IMAGE
# ==================================================

try:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

except (UnidentifiedImageError, OSError) as exc:
    raise ValueError(
        f"The selected file is not a valid readable image: "
        f"{image_path}"
    ) from exc

image_tensor = transform(image).unsqueeze(0).to(DEVICE)


# ==================================================
# 7. PREDICT DAMAGE CLASS
# ==================================================

with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

confidence, predicted_index = probabilities.max(dim=0)

predicted_class = CLASSES[predicted_index.item()]


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n========== DAMAGE CLASSIFICATION ==========")

print(f"Image: {image_path.name}")
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence.item() * 100:.2f}%")

if confidence.item() < LOW_CONFIDENCE_THRESHOLD:
    print(
        "\nWARNING: The prediction confidence is low. "
        "Manual review is recommended."
    )

print("\nClass probability breakdown:")

for index in probabilities.argsort(
    descending=True
).tolist():

    class_name = CLASSES[index]
    score = probabilities[index].item() * 100

    print(f"{class_name}: {score:.2f}%")

print(
    "\nNote: Confidence scores are model outputs, "
    "not guarantees that the classification is correct."
)

