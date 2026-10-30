
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install CPU-only PyTorch and torchvision.
RUN pip install --no-cache-dir \
    --retries 10 \
    --timeout 120 \
    torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend dependencies.
RUN grep -v -E '^(torch|torchvision)==' requirements.txt > /tmp/requirements-other.txt \
    && pip install --no-cache-dir \
    --retries 10 \
    --timeout 120 \
    -r /tmp/requirements-other.txt

COPY backend /app/backend
COPY ai_model /app/ai_model

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
