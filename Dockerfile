# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcribe.py upload-server.py ./

EXPOSE 8080
CMD ["uvicorn", "upload-server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
