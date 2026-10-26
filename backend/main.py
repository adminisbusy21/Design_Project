from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms

from backend.database import SessionLocal
from backend.models import InspectionResult
from backend.priority import calculate_priority


app = FastAPI(
    title="Distributed AI Damage Classification Service",
    description=(
        "API for concrete crack detection and multi-class damage classification."
    ),
    version="2.0.0",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
MAX_IMAGE_SIDE = 6000
AI_DIR = Path(__file__).resolve().parent.parent / "ai_model"

DAMAGE_CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]
DAMAGE_MODEL_PATH = AI_DIR / "resnet18_damage_classifier.pth"

image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_resnet18(model_path: Path, number_of_classes: int):
    """Load a trained ResNet18 classifier."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model file not found: {model_path}")

    network = models.resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, number_of_classes)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    network.load_state_dict(state_dict)
    network = network.to(DEVICE)
    network.eval()
    return network

damage_model = load_resnet18(DAMAGE_MODEL_PATH, len(DAMAGE_CLASSES))


def classify_image(image: Image.Image, model, class_names: list[str]) -> dict:
    """Return predicted class, confidence, and per-class probabilities."""
    tensor = image.convert("RGB")
    tensor = image_transform(tensor).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    return {
        "prediction": class_names[predicted_index],
        "confidence": round(float(probabilities[predicted_index].item()), 4),
        "class_probabilities": {
            class_names[index]: round(float(probabilities[index].item()), 4)
            for index in range(len(class_names))
        },
    }


async def read_uploaded_image(file: UploadFile) -> dict:
    """Validate uploaded image and return decoded image plus metadata."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPEG, PNG, or WEBP.",
        )

    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image size must not exceed 10 MB.")

    try:
        with Image.open(BytesIO(image_data)) as check_image:
            check_image.verify()
        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format
            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=400, detail="Unsupported image format.")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail="Invalid image dimensions.")
            if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image dimensions must not exceed {MAX_IMAGE_SIDE} pixels per side.",
                )
            decoded_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    return {
        "image": decoded_image,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(image_data),
    }


@app.get("/")
async def root():
    return {
        "message": "FastAPI is working",
        "project": "Distributed AI Damage Classification Service",
        "available_endpoints": [
            "/",
            "/health",
            "/upload",
            "/predict_damage",
            "/inspections",
            "/inspections/{inspection_id}/review",
            "/docs",
        ],
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "damage-classification-api",
        "device": str(DEVICE),
        "damage_model_loaded": damage_model is not None,
    }


class ReviewRequest(BaseModel):
    review_status: str
    review_notes: str | None = None


@app.get("/inspections")
async def get_inspections():
    db = SessionLocal()
    try:
        results = (
            db.query(InspectionResult)
            .order_by(InspectionResult.created_at.desc())
            .all()
        )
        return [
            {
                "id": result.id,
                "filename": result.filename,
                "format": result.image_format,
                "width": result.width,
                "height": result.height,
                "size_bytes": result.size_bytes,
                "prediction": result.prediction,
                "confidence": result.confidence,
                "priority": result.priority,
                "priority_reason": result.priority_reason,
                "review_required": result.review_required,
                "review_status": result.review_status,
                "review_notes": result.review_notes,
                "created_at": result.created_at,
            }
            for result in results
        ]
    finally:
        db.close()


@app.put("/inspections/{inspection_id}/review")
async def review_inspection(inspection_id: int, review: ReviewRequest):
    if review.review_status not in {"APPROVED", "REJECTED"}:
        raise HTTPException(
            status_code=400,
            detail="review_status must be APPROVED or REJECTED.",
        )

    db = SessionLocal()
    try:
        inspection = (
            db.query(InspectionResult)
            .filter(InspectionResult.id == inspection_id)
            .first()
        )
        if inspection is None:
            raise HTTPException(status_code=404, detail="Inspection result not found.")

        inspection.review_status = review.review_status
        inspection.review_notes = review.review_notes
        db.commit()
        db.refresh(inspection)
        return {
            "message": "Inspection review updated successfully",
            "id": inspection.id,
            "review_status": inspection.review_status,
            "review_notes": inspection.review_notes,
        }
    finally:
        db.close()


@app.post("/upload")
async def upload_images(file: UploadFile = File(...)):
    """Classify a damage image, calculate its priority, and persist the result."""
    metadata = await read_uploaded_image(file)
    prediction_result = classify_image(metadata["image"], damage_model, DAMAGE_CLASSES)
    priority_result = calculate_priority(
        prediction_result["prediction"], prediction_result["confidence"]
    )

    db = SessionLocal()
    try:
        inspection_result = InspectionResult(
            filename=file.filename or "unknown",
            image_format=metadata["format"],
            width=metadata["width"],
            height=metadata["height"],
            size_bytes=metadata["size_bytes"],
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            priority=priority_result["priority"],
            priority_reason=priority_result["reason"],
            review_required=priority_result["review_required"],
            review_status="PENDING",
            review_notes=None,
        )
        db.add(inspection_result)
        db.commit()
        db.refresh(inspection_result)
        inspection_id = inspection_result.id
    finally:
        db.close()

    return {
        "message": "Image classified successfully",
        "id": inspection_id,
        "filename": file.filename,
        "format": metadata["format"],
        "width": metadata["width"],
        "height": metadata["height"],
        "size_bytes": metadata["size_bytes"],
        **prediction_result,
        "priority": priority_result["priority"],
        "priority_reason": priority_result["reason"],
        "review_required": priority_result["review_required"],
        "review_status": "PENDING",
        "model_version": "resnet18-damage-v1",
    }


@app.post("/predict_damage")
async def predict_damage_endpoint(file: UploadFile = File(...)):
    """Classify Crack, Non-Crack, Road_Pothole, Road_Surface_Erosion, or Rust_Corrosion."""
    metadata = await read_uploaded_image(file)
    result = classify_image(metadata["image"], damage_model, DAMAGE_CLASSES)
    return {
        "message": "Damage classification completed",
        "filename": file.filename,
        "format": metadata["format"],
        "width": metadata["width"],
        "height": metadata["height"],
        "size_bytes": metadata["size_bytes"],
        **result,
        "model_version": "resnet18-damage-v1",
    }
