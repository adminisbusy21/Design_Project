
"""Run five-class structural damage classification on a local image."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ai_model" / "resnet18_damage_classifier_clean.pth"

CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

CONFIDENCE_THRESHOLD = 0.80

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(device: torch.device) -> nn.Module:
    """Load the trained five-class ResNet18 checkpoint."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Obtain the trained checkpoint before running prediction."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def predict_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """Return predicted class, confidence, and class probabilities."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image: {image_path}") from exc

    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_index].item())

    all_probabilities = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(CLASSES)
    }

    return CLASSES[predicted_index], confidence, all_probabilities


def main() -> None:
    image_input = input("Enter image path: ").strip().strip('"')

    if not image_input:
        print("Error: no image path provided.")
        return

    image_path = Path(image_input).expanduser()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        model = load_model(device)
        prediction, confidence, probabilities = predict_image(
            model,
            image_path,
            device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return

    print(f"\nDevice: {device}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")

    print("\nClass probabilities:")
    for class_name, probability in probabilities.items():
        print(f"  {class_name}: {probability:.2%}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(
            "\nWarning: confidence is below "
            f"{CONFIDENCE_THRESHOLD:.0%}. Manual review is recommended."
        )


if __name__ == "__main__":
    main()
