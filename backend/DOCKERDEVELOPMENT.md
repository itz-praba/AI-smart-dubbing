# Docker Deployment Guide

Complete guide for deploying the Media Processing API with Docker.

## Quick Start

### 1. Build and Run

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### 2. Stop and Clean

```bash
# Stop services
docker-compose down

# Remove volumes (clears temp files and models)
docker-compose down -v
```

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Required
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
HF_TOKEN=your_huggingface_token

# Optional
FFMPEG_PATH=ffmpeg
FFPROBE_PATH=ffprobe
MODEL_CACHE_DIR=/app/models
```

### GPU Support

Uncomment GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Prerequisites:**
- NVIDIA GPU
- NVIDIA Docker runtime installed
- Docker Compose v1.28+

Install NVIDIA Docker:
```bash
# Ubuntu/Debian
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
# For CPU-only servers
mem_limit: 16g

# For GPU servers
mem_limit: 32g

# CPU cores (optional)
cpus: 4
```

## Production Deployment

### 1. Build Production Image

```bash
# Remove development mounts
# Edit docker-compose.yml and comment out volume mounts for .py files

# Build
docker-compose build --no-cache

# Tag for registry
docker tag media-processing-api:latest your-registry.com/media-api:v3.0.0

# Push to registry
docker push your-registry.com/media-api:v3.0.0
```

### 2. Deploy on Server

```bash
# Pull image
docker pull your-registry.com/media-api:v3.0.0

# Run with production settings
docker run -d \
  --name media-api \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  --memory="16g" \
  your-registry.com/media-api:v3.0.0
```

### 3. Multiple Instances (Load Balancing)

```yaml
# docker-compose.yml
version: '3.8'
services:
  media-api:
    # ... existing config ...
    deploy:
      replicas: 3  # Run 3 instances

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - media-api
```

## Monitoring

### View Logs

```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f media-api

# Last 100 lines
docker-compose logs --tail=100 media-api
```

### Container Stats

```bash
# Real-time stats
docker stats

# Check memory usage
docker stats media-processing-api --no-stream
```

### Health Checks

```bash
# Manual health check
curl http://localhost:8000/health

# All service health
curl http://localhost:8000/ai/health
curl http://localhost:8000/ai/speech/health
curl http://localhost:8000/ai/translate/health
curl http://localhost:8000/ai/voice-clone/health
curl http://localhost:8000/ai/lip-sync/health
curl http://localhost:8000/ai/video-merge/health
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs media-api

# Inspect container
docker inspect media-processing-api

# Check Docker daemon
sudo systemctl status docker
```

### Out of Memory

```bash
# Increase memory limit in docker-compose.yml
mem_limit: 32g

# Or run with more swap
docker run --memory="16g" --memory-swap="32g" ...
```

### GPU Not Detected

```bash
# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check NVIDIA runtime
docker run --rm --gpus all pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime nvidia-smi
```

### Models Download Slowly

```bash
# Pre-download models
docker run -it --rm \
  -v $(pwd)/models:/app/models \
  your-image \
  python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### FFmpeg Not Found

```bash
# Rebuild image (FFmpeg should be included)
docker-compose build --no-cache

# Verify FFmpeg in container
docker exec media-processing-api which ffmpeg
```

## Advanced Configuration

### Custom Dockerfile

For specific needs, modify Dockerfile:

```dockerfile
# Use different base image
FROM python:3.11-slim

# Add custom system packages
RUN apt-get install -y your-package

# Use different Python version
RUN pyenv install 3.10.0
```

### Network Configuration

```yaml
# docker-compose.yml
services:
  media-api:
    networks:
      - media-network

networks:
  media-network:
    driver: bridge
```

### Secrets Management

```yaml
# docker-compose.yml
services:
  media-api:
    secrets:
      - aws_credentials
      - hf_token

secrets:
  aws_credentials:
    file: ./secrets/aws.txt
  hf_token:
    file: ./secrets/hf_token.txt
```

## Performance Tuning

### Worker Configuration

Adjust workers based on resources:

```dockerfile
# CPU-only: More workers
CMD ["gunicorn", "main:app", "--workers", "4", ...]

# GPU: Fewer workers (memory)
CMD ["gunicorn", "main:app", "--workers", "2", ...]

# Single-threaded (debugging)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Timeout Settings

For longer operations:

```dockerfile
CMD ["gunicorn", "main:app", \
     "--timeout", "1800", \  # 30 minutes
     ...]
```

### Cache Optimization

```yaml
volumes:
  # Keep models cached
  - models_cache:/app/models

volumes:
  models_cache:
    driver: local
```

## Backup and Restore

### Backup Models

```bash
# Backup models directory
docker run --rm \
  -v media-processing-api_models:/models \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/models-backup.tar.gz /models
```

### Restore Models

```bash
# Restore from backup
docker run --rm \
  -v media-processing-api_models:/models \
  -v $(pwd):/backup \
  alpine \
  tar xzf /backup/models-backup.tar.gz -C /
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push
        run: |
          docker build -t ${{ secrets.REGISTRY }}/media-api:${{ github.sha }} .
          docker push ${{ secrets.REGISTRY }}/media-api:${{ github.sha }}
```

## Best Practices

1. **Never commit `.env`** - Use `.env.example` template
2. **Use volume for models** - Avoid re-downloading on restart
3. **Set resource limits** - Prevent OOM crashes
4. **Enable health checks** - Auto-restart on failure
5. **Monitor logs** - Use centralized logging (ELK, CloudWatch)
6. **Regular updates** - Keep base image and deps current
7. **Security scan** - Use `docker scan` or Trivy
8. **Use multi-stage builds** - Reduce final image size (optional)

## Support

For issues:
1. Check container logs: `docker-compose logs -f`
2. Verify environment variables: `docker exec media-processing-api env`
3. Test inside container: `docker exec -it media-processing-api bash`
4. Check resource usage: `docker stats`