FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Complete Media Processing API with 6 AI services"
LABEL version="3.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    rubberband-cli \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY audio_controller.py .
COPY text_controller.py .
COPY target_language.py .
COPY voice_cloning.py .
COPY lip_sync.py .
COPY video_rendering.py .

# Create directories
RUN mkdir -p temp models && chmod 777 temp models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "900", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]