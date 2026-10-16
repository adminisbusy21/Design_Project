from io import BytesIO
from pathlib import Path
from backend.priority import calculate_priority
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from backend.models import InspectionResult
from backend.database import SessionLocal
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Distributed AI Damage Classification Service",
    description="API for structural damage classification",
    version="1.0.0",
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

CLASS_NAMES = ["Negative", "Positive"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_model"
    / "resnet18_crack_classifier.pth"
)


# --------------------------------------------------
# Load trained AI model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES),
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model = model.to(DEVICE)
model.eval()


# --------------------------------------------------
# Image preprocessing
# Must match the training preprocessing
# --------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# AI prediction function
# --------------------------------------------------

def predict_damage(image: Image.Image) -> dict:
    """
    Classify a concrete image as cracked or uncracked.
    """

    image = image.convert("RGB")

    image_tensor = image_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = int(
            torch.argmax(probabilities, dim=1).item()
        )

        confidence = float(
            probabilities[0, predicted_index].item()
        )

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 4),
    }


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "FastAPI is working",
        "project": "Distributed AI Damage Classification Service",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "damage-classification-api",
        "device": str(DEVICE),
        "model_loaded": True,
    }


# --------------------------------------------------
# Image upload and classification endpoint
# --------------------------------------------------
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
class ReviewRequest(BaseModel):
    review_status: str
    review_notes: str | None = None


@app.put("/inspections/{inspection_id}/review")
async def review_inspection(
    inspection_id: int,
    review: ReviewRequest,
):
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
    finally:
        db.close()
@app.post("/upload")
async def upload_images(file: UploadFile = File(...)):

    # Check uploaded content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPEG, PNG, or WEBP.",
        )

    # Read image data
    image_data = await file.read()

    # Check empty file
    if not image_data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    # Check file size
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB",
        )

    # Validate image
    try:
        with Image.open(BytesIO(image_data)) as image:
            image.verify()

        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format
            width, height = image.size

            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format",
                )

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image dimensions",
                )

            if width > 6000 or height > 6000:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions must not exceed 6000 pixels per side.",
                )

            # Run AI classification
            prediction_result = predict_damage(image.copy())
            priority_result = calculate_priority(
                 prediction_result["prediction"],
                 prediction_result["confidence"],
            )
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image",
        )
    db = SessionLocal()
    try:
        inspection_result = InspectionResult(
            filename=file.filename or "unknown",
            image_format=image_format,
            width=width,
            height=height,
            size_bytes=len(image_data),
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
    finally:
        db.close()    
    # Return classification result
    return {
        "message": "Image classified successfully",
        "filename": file.filename,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(image_data),
        "prediction": prediction_result["prediction"],
        "priority": priority_result["priority"],
        "priority_reason": priority_result["reason"],
        "review_required": priority_result["review_required"],
        "confidence": prediction_result["confidence"],
        "model_version": "resnet18-v1",
    }