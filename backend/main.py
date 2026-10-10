from io import BytesIO
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI,File, HTTPException, UploadFile
app = FastAPI(
    title="Distributed AI Damage Classification Service",
    description="API for structural damage classification",
    version="1.0.0",
)
ALLOWED_CONTENT_TYPES = {"image/jpeg","image/webp", "image/png"}
MAX_FILE_SIZE=10*1024*1024 
@app.get("/")
async def root():
    return {
        "message": "FastAPI is working",
        "project": "Distributed AI Damage Classification Service",
    }
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "damage-classification-api",
    }
@app.post("/upload")
async def upload_images(file:UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type so upload in JPEG, PNG, or WEBP format",)
    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty",)
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,detail="IMage size should not exceed 10MB",)
    try:
        with Image.open(BytesIO(image_data)) as image:
            image.verify()
        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(status_code=400, detail="Unsupported image format",)
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail="Invalid image dimensions",)
    except (UnidentifiedImageError, OSError,ValueError):
        raise HTTPException(status_code=400,detail="Uploaded file is not a valid image",)
    return {
        "message":"Image received successfully",
        "filename":file.filename,
        "format": image_format,
        "width": width,
        "height":height,
        # "content_type":file.content_type,
        "size_bytes":len(image_data),
    }