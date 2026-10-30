
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms

from fastapi.middleware.cors import CORSMiddleware

from backend.database import SessionLocal
from backend.models import InspectionResult
from backend.priority import calculate_priority


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

app = FastAPI(
    title="Distributed AI Damage Classification Service",
    description=(
        "API for five-class structural damage classification, "
        "risk-aware inspection prioritization, and review management."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_VERSION = "resnet18-damage-clean-v2"

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 6000

AI_DIR = Path(__file__).resolve().parent.parent / "ai_model"

DAMAGE_CLASSES = [
    "Crack",
    "Non-Crack",
    "Road_Pothole",
    "Road_Surface_Erosion",
    "Rust_Corrosion",
]

DAMAGE_MODEL_PATH = AI_DIR / "resnet18_damage_classifier_clean.pth"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================================
# MODEL LOADING
# =====================================================

def load_resnet18(
    model_path: Path,
    number_of_classes: int,
) -> nn.Module:
    """Load a trained ResNet18 classifier from a state dictionary."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model file not found: {model_path}"
        )

    network = models.resnet18(weights=None)
    network.fc = nn.Linear(
        network.fc.in_features,
        number_of_classes,
    )

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    network.load_state_dict(state_dict, strict=True)
    network.to(DEVICE)
    network.eval()

    return network


# Fail at startup if the required checkpoint is missing or incompatible.
damage_model = load_resnet18(
    DAMAGE_MODEL_PATH,
    len(DAMAGE_CLASSES),
)


# =====================================================
# IMAGE CLASSIFICATION
# =====================================================

def classify_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
) -> dict:
    """Return the predicted class, confidence, and class probabilities."""
    tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return {
        "prediction": class_names[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index].item()),
            4,
        ),
        "class_probabilities": {
            class_names[index]: round(
                float(probabilities[index].item()),
                4,
            )
            for index in range(len(class_names))
        },
    }


# =====================================================
# IMAGE VALIDATION
# =====================================================

async def read_uploaded_image(file: UploadFile) -> dict:
    """Validate an uploaded image and return its image and metadata."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPEG, PNG, or WEBP.",
        )

    image_data = await file.read()

    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MiB.",
        )

    try:
        with Image.open(BytesIO(image_data)) as check_image:
            check_image.verify()

        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format

            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format.",
                )

            width, height = image.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image dimensions.",
                )

            if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Image dimensions must not exceed "
                        f"{MAX_IMAGE_SIDE} pixels per side."
                    ),
                )

            decoded_image = image.convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    return {
        "image": decoded_image,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(image_data),
    }


# =====================================================
# RESPONSE HELPERS
# =====================================================

def inspection_to_dict(result: InspectionResult) -> dict:
    """Convert a database inspection record into an API response."""
    return {
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


# =====================================================
# ROOT AND HEALTH ENDPOINTS
# =====================================================

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
        "model_version": MODEL_VERSION,
        "classes": DAMAGE_CLASSES,
    }


# =====================================================
# INSPECTION REVIEW
# =====================================================

class ReviewRequest(BaseModel):
    review_status: Literal["APPROVED", "REJECTED"]
    review_notes: str | None = None


@app.get("/inspections")
async def get_inspections():
    """Return persisted inspections, newest first."""
    db = SessionLocal()

    try:
        results = (
            db.query(InspectionResult)
            .order_by(InspectionResult.created_at.desc())
            .all()
        )

        return [inspection_to_dict(result) for result in results]

    finally:
        db.close()


@app.put("/inspections/{inspection_id}/review")
async def review_inspection(
    inspection_id: int,
    review: ReviewRequest,
):
    """Approve or reject a persisted inspection."""
    db = SessionLocal()

    try:
        inspection = (
            db.query(InspectionResult)
            .filter(InspectionResult.id == inspection_id)
            .first()
        )

        if inspection is None:
            raise HTTPException(
                status_code=404,
                detail="Inspection result not found.",
            )

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

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# =====================================================
# UPLOAD, CLASSIFY, PRIORITIZE, AND PERSIST
# =====================================================

@app.post("/upload")
async def upload_images(file: UploadFile = File(...)):
    """Classify an image, calculate risk priority, and save its result."""
    metadata = await read_uploaded_image(file)

    prediction_result = classify_image(
        metadata["image"],
        damage_model,
        DAMAGE_CLASSES,
    )

    priority_result = calculate_priority(
        prediction_result["prediction"],
        prediction_result["confidence"],
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

        response = {
            "message": "Image classified successfully",
            **inspection_to_dict(inspection_result),
            "class_probabilities": prediction_result[
                "class_probabilities"
            ],
            "model_version": MODEL_VERSION,
        }

        return response

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# =====================================================
# PREDICTION WITHOUT DATABASE PERSISTENCE
# =====================================================

@app.post("/predict_damage")
async def predict_damage_endpoint(file: UploadFile = File(...)):
    """Classify an image without creating a database inspection record."""
    metadata = await read_uploaded_image(file)

    result = classify_image(
        metadata["image"],
        damage_model,
        DAMAGE_CLASSES,
    )

    return {
        "message": "Damage classification completed",
        "filename": file.filename or "unknown",
        "format": metadata["format"],
        "width": metadata["width"],
        "height": metadata["height"],
        "size_bytes": metadata["size_bytes"],
        **result,
        "model_version": MODEL_VERSION,
    }
