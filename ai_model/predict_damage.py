
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "resnet18_damage_classifier.pth"

CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

image_path = input("Enter the full path of an image: ").strip().strip('"')
image_path = Path(image_path)

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    probabilities = torch.softmax(model(image_tensor), dim=1)[0]

confidence, predicted_index = probabilities.max(dim=0)

print("\nPrediction:", CLASSES[predicted_index.item()])
print(f"Confidence: {confidence.item() * 100:.2f}%")

print("\nAll class probabilities:")
for index in probabilities.argsort(descending=True).tolist():
    print(f"{CLASSES[index]}: {probabilities[index].item() * 100:.2f}%")
