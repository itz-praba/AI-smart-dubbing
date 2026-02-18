# Complete Media Processing API - Production Ready

End-to-end video dubbing pipeline with 6 integrated AI services.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with AWS and HuggingFace credentials

# 3. Run the API
python main.py
```

## Services Included

1. **audio_controller.py** - Video to Audio extraction
2. **text_controller.py** - Speech to Text with speaker diarization
3. **target_language.py** - Multi-language translation
4. **voice_cloning.py** - AI voice cloning with XTTS
5. **lip_sync.py** - Audio timing alignment
6. **video_rendering.py** - Multi-audio video rendering

## API Endpoints

- Video to Audio: `POST /ai/video-to-audio`
- Speech to Text: `POST /ai/speech-to-text`
- Translation: `POST /ai/translate/timed`
- Voice Cloning: `POST /ai/voice-clone-tts`
- Lip-Sync: `POST /ai/lip-sync-align`
- Video Merge: `POST /ai/video-merge`

## Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## System Requirements

- Python 3.8+
- FFmpeg
- Rubberband
- 8GB+ RAM (16GB recommended)
- CUDA GPU (optional but recommended)

## Configuration

Set in `.env`:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
HF_TOKEN=your_huggingface_token
```

## Complete Workflow Example

```python
import requests

# 1. Extract audio
audio = extract_audio("video.mp4")

# 2. Transcribe with speakers
transcript = transcribe(audio, diarize=True)

# 3. Translate
translated = translate(transcript, target="es")

# 4. Clone voices
dubbed = clone_voice(speaker_audio, translated_text)

# 5. Align timing
aligned = align_timing(dubbed, original_timing)

# 6. Merge video
final = merge_video(video, [dubbed_es, dubbed_fr])
```

## Production Deployment

```bash
gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Support

All services include:
- ✅ Comprehensive error handling
- ✅ S3 integration
- ✅ Health check endpoints
- ✅ Detailed logging
- ✅ Input validation
- ✅ GPU acceleration