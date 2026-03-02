import os
import uuid
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

import boto3
import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError, BotoCoreError
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/ai", tags=["Speech"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB
MAX_AUDIO_DURATION = 3600  # 1 hour in seconds
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.wma'}
SUPPORTED_LANGUAGES = [
    'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ru', 'zh', 'ja', 'ko',
    'ar', 'hi', 'tr', 'pl', 'cs', 'sv', 'da', 'fi', 'no', 'uk', 'vi',
    "ta"
]

# Whisper model sizes and their memory requirements
WHISPER_MODELS = {
    'tiny': 'tiny',
    'base': 'base',
    'small': 'small',
    'medium': 'medium',
    'large': 'large-v3'
}

# =====================================================
# ENVIRONMENT VALIDATION
# =====================================================
REQUIRED_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION"
]

for env_var in REQUIRED_ENV_VARS:
    if not os.getenv(env_var):
        raise RuntimeError(f"Missing required environment variable: {env_var}")

# HuggingFace token for speaker diarization
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.warning("HuggingFace token not found - speaker diarization will be disabled")

# =====================================================
# AWS S3 CLIENT
# =====================================================
try:
    from botocore.config import Config
    
    boto_config = Config(
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        },
        connect_timeout=10,
        read_timeout=120
    )
    
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
        config=boto_config
    )
    
    # Test S3 connection
    s3_client.list_buckets()
    logger.info("S3 client initialized successfully")
    
except Exception as e:
    raise RuntimeError(f"Failed to initialize S3 client: {str(e)}")

# =====================================================
# DEVICE CONFIGURATION
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"Using device: {device}, compute type: {compute_type}")

# =====================================================
# MODEL MANAGEMENT (Lazy Loading)
# =====================================================
_whisper_model = None
_diarization_pipeline = None

def get_whisper_model(model_size: str = "large-v3") -> WhisperModel:
    """
    Get or initialize Whisper model with lazy loading
    
    Args:
        model_size: Size of the Whisper model to use
        
    Returns:
        WhisperModel instance
    """
    global _whisper_model
    
    if _whisper_model is None:
        try:
            logger.info(f"Loading Whisper model: {model_size} on {device}")
            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root="./models",  # Cache models locally
                num_workers=4 if device == "cuda" else 2
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}")
    
    return _whisper_model


def get_diarization_pipeline() -> Optional[Pipeline]:
    """
    Get or initialize speaker diarization pipeline with lazy loading
    
    Returns:
        Pipeline instance or None if token not available
    """
    global _diarization_pipeline
    
    if not HUGGINGFACE_TOKEN:
        logger.warning("Diarization disabled: HuggingFace token not found")
        return None
    
    if _diarization_pipeline is None:
        try:
            logger.info("Loading speaker diarization pipeline")
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HUGGINGFACE_TOKEN
            )
            
            # Move to appropriate device
            if device == "cuda":
                _diarization_pipeline.to(torch.device("cuda"))
            
            logger.info("Diarization pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            logger.warning("Continuing without speaker diarization")
            return None
    
    return _diarization_pipeline


# =====================================================
# PYDANTIC MODELS
# =====================================================
class TranscriptSegment(BaseModel):
    """Individual transcript segment"""
    speaker: str
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class SpeechToTextRequest(BaseModel):
    """Request model for speech-to-text conversion"""
    audio_bucket: str = Field(..., min_length=1, max_length=63)
    audio_key: str = Field(..., min_length=1, max_length=1024)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    language: Optional[str] = Field(None, description="ISO language code (e.g., 'en', 'es')")
    diarize: bool = Field(True, description="Perform speaker diarization")
    model_size: str = Field("large-v3", description="Whisper model size")
    word_timestamps: bool = Field(True, description="Include word-level timestamps")
    
    @validator('audio_bucket', 'output_bucket')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name"""
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('audio_key')
    def validate_audio_key(cls, v):
        """Validate audio key and extension"""
        # Prevent path traversal
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid audio key: path traversal detected")
        
        # Check extension
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Allowed formats: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            )
        
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code"""
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        return v
    
    @validator('model_size')
    def validate_model_size(cls, v):
        """Validate Whisper model size"""
        if v not in WHISPER_MODELS.values():
            raise ValueError(
                f"Invalid model size '{v}'. "
                f"Available: {', '.join(WHISPER_MODELS.keys())}"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "audio_bucket": "my-audio-bucket",
                "audio_key": "audio/sample.wav",
                "output_bucket": "my-transcripts-bucket",
                "language": "en",
                "diarize": True,
                "model_size": "large-v3"
            }
        }


class SpeechToTextResponse(BaseModel):
    """Response model for speech-to-text conversion"""
    success: bool
    job_id: str
    language: str
    language_probability: float
    transcript_bucket: str
    transcript_key: str
    segments_count: int
    diarization_performed: bool
    speakers_detected: int
    duration_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    message: str


class TranscriptResult(BaseModel):
    """Complete transcript result"""
    job_id: str
    language: str
    language_probability: float
    segments: List[TranscriptSegment]
    diarization_performed: bool
    speakers_detected: int
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any]


# =====================================================
# HELPER FUNCTIONS
# =====================================================
@contextmanager
def temp_files(*file_paths):
    """Context manager to ensure temporary files are cleaned up"""
    try:
        yield
    finally:
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename to prevent security issues
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
    """
    filename = Path(filename).name
    
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '._-':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    result = ''.join(safe_chars)
    
    if not result or result.startswith('.'):
        result = f"audio_{uuid.uuid4().hex[:8]}"
    
    return result[:max_length]


def check_s3_file(bucket: str, key: str) -> Dict[str, Any]:
    """
    Check if S3 file exists and get metadata
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Dictionary with file metadata
        
    Raises:
        HTTPException: If file not found or access denied
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return {
            'size': response.get('ContentLength', 0),
            'content_type': response.get('ContentType', ''),
            'last_modified': response.get('LastModified')
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found in bucket '{bucket}' with key '{key}'"
            )
        elif error_code == '403':
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to bucket '{bucket}'"
            )
        else:
            logger.error(f"S3 head_object error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to check file in S3: {error_code}"
            )


def download_from_s3(bucket: str, key: str, local_path: str) -> None:
    """
    Download file from S3 with error handling
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local file path to save to
        
    Raises:
        HTTPException: If download fails
    """
    try:
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        
        if not os.path.exists(local_path):
            raise HTTPException(
                status_code=500,
                detail="File download completed but file not found locally"
            )
        
        file_size = os.path.getsize(local_path)
        if file_size == 0:
            raise HTTPException(
                status_code=500,
                detail="Downloaded file is empty"
            )
        
        logger.info(f"Successfully downloaded {file_size} bytes")
        
    except ClientError as e:
        logger.error(f"S3 download error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download from S3: {e.response['Error']['Message']}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected download error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during download: {str(e)}"
        )


def upload_to_s3(local_path: str, bucket: str, key: str, metadata: dict = None) -> None:
    """
    Upload file to S3 with error handling
    
    Args:
        local_path: Local file path to upload
        bucket: S3 bucket name
        key: S3 object key
        metadata: Optional metadata dictionary
        
    Raises:
        HTTPException: If upload fails
    """
    try:
        extra_args = {
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256"
        }
        
        if metadata:
            extra_args["Metadata"] = metadata
        
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        logger.info(f"Upload successful")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 upload error: {e}")
        
        if error_code == 'NoSuchBucket':
            raise HTTPException(
                status_code=404,
                detail=f"Destination bucket '{bucket}' does not exist"
            )
        elif error_code == 'AccessDenied':
            raise HTTPException(
                status_code=403,
                detail=f"Access denied to bucket '{bucket}'"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload to S3: {e.response['Error']['Message']}"
            )
    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during upload: {str(e)}"
        )


def perform_diarization(audio_path: str) -> List[Dict[str, Any]]:
    speaker_segments = []

    try:
        diarization_pipeline = get_diarization_pipeline()

        if diarization_pipeline is None:
            logger.warning("Diarization pipeline not available")
            return speaker_segments

        logger.info("Starting speaker diarization")
        diarization = diarization_pipeline(audio_path)

        # 🔍 DEBUG: log type and available attributes
        logger.info(f"Diarization object type: {type(diarization)}")
        logger.info(f"Available attributes: {dir(diarization)}")

        # ❌ Check if old attribute exists
        if hasattr(diarization, "speaker_diarization"):
            logger.info("Found attribute: speaker_diarization")
            annotation = diarization.speaker_diarization
        else:
            logger.error("Attribute 'speaker_diarization' NOT found.")
            annotation = diarization  # fallback to direct annotation

        # ✅ Correct for pyannote 3.x
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2)
            })

        logger.info(
            f"Diarization complete: {len(speaker_segments)} segments, "
            f"{len(set(s['speaker'] for s in speaker_segments))} unique speakers"
        )

    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)

    return speaker_segments
    """
    Perform speaker diarization on audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of speaker segments
    """


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    model_size: str = "large-v3",
    word_timestamps: bool = True
) -> tuple:
    """
    Transcribe audio file using Whisper
    
    Args:
        audio_path: Path to audio file
        language: Optional language code
        model_size: Whisper model size
        word_timestamps: Whether to include word timestamps
        
    Returns:
        Tuple of (segments, info)
        
    Raises:
        HTTPException: If transcription fails
    """
    try:
        whisper_model = get_whisper_model(model_size)
        
        logger.info(f"Starting transcription with model {model_size}, language: {language or 'auto-detect'}")
        
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            word_timestamps=word_timestamps,
            language=language,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert generator to list
        segments = list(segments)
        
        logger.info(f"Transcription complete: {len(segments)} segments, "
                   f"language: {info.language} (prob: {info.language_probability:.2f})")
        
        return segments, info
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Audio transcription failed: {str(e)}"
        )


def align_speakers_with_transcription(
    transcription_segments,
    speaker_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Align speaker diarization with transcription segments
    
    Args:
        transcription_segments: Whisper transcription segments
        speaker_segments: Speaker diarization segments
        
    Returns:
        List of aligned transcript segments
    """
    transcript = []
    
    for seg in transcription_segments:
        speaker = "SPEAKER_UNKNOWN"
        
        # Find overlapping speaker segment
        if speaker_segments:
            max_overlap = 0
            for s in speaker_segments:
                # Calculate overlap
                overlap_start = max(seg.start, s["start"])
                overlap_end = min(seg.end, s["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = s["speaker"]
        
        # Calculate average word confidence if available
        confidence = None
        if hasattr(seg, 'words') and seg.words:
            confidences = [w.probability for w in seg.words if hasattr(w, 'probability')]
            if confidences:
                confidence = sum(confidences) / len(confidences)
        
        transcript.append({
            "speaker": speaker,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "confidence": round(confidence, 3) if confidence else None
        })
    
    return transcript


# =====================================================
# API ENDPOINT
# =====================================================
@router.post(
    "/speech-to-text",
    response_model=SpeechToTextResponse,
    summary="Convert speech to text with speaker diarization",
    description="Transcribe audio files using Whisper with optional speaker diarization"
)
async def speech_to_text(request: SpeechToTextRequest) -> SpeechToTextResponse:
    """
    Convert speech to text with speaker diarization
    
    This endpoint:
    1. Downloads audio from S3
    2. Performs speaker diarization (if enabled)
    3. Transcribes audio using Whisper
    4. Aligns speakers with transcription
    5. Uploads result to S3
    6. Cleans up temporary files
    
    Args:
        request: SpeechToTextRequest with audio location and options
        
    Returns:
        SpeechToTextResponse with job details and transcript location
    """
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(f"Starting job {job_id}: {request.audio_bucket}/{request.audio_key}")
    
    # Generate file paths
    audio_filename = sanitize_filename(Path(request.audio_key).name)
    audio_path = str(TEMP_DIR / f"{job_id}_{audio_filename}")
    output_path = str(TEMP_DIR / f"{job_id}.json")
    
    try:
        with temp_files(audio_path, output_path):
            # Step 1: Check file exists and size
            file_metadata = check_s3_file(request.audio_bucket, request.audio_key)
            file_size = file_metadata['size']
            
            if file_size > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file too large ({file_size / (1024*1024):.1f}MB). "
                           f"Maximum allowed: {MAX_AUDIO_SIZE / (1024*1024):.0f}MB"
                )
            
            logger.info(f"File size: {file_size / (1024*1024):.2f}MB")
            
            # Step 2: Download audio from S3
            download_from_s3(request.audio_bucket, request.audio_key, audio_path)
            
            # Step 3: Speaker diarization (optional)
            speaker_segments = []
            if request.diarize and HUGGINGFACE_TOKEN:
                speaker_segments = perform_diarization(audio_path)
            elif request.diarize and not HUGGINGFACE_TOKEN:
                logger.warning("Diarization requested but HuggingFace token not available")
            
            # Step 4: Transcribe audio
            transcription_segments, info = transcribe_audio(
                audio_path,
                language=request.language,
                model_size=request.model_size,
                word_timestamps=request.word_timestamps
            )
            
            # Step 5: Align speakers with transcription
            transcript = align_speakers_with_transcription(
                transcription_segments,
                speaker_segments
            )
            
            # Calculate statistics
            speakers_detected = len(set(s["speaker"] for s in transcript))
            duration = transcript[-1]["end"] if transcript else 0
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Step 6: Create result object
            result = TranscriptResult(
                job_id=job_id,
                language=info.language,
                language_probability=round(info.language_probability, 3),
                segments=[TranscriptSegment(**s) for s in transcript],
                diarization_performed=len(speaker_segments) > 0,
                speakers_detected=speakers_detected,
                duration_seconds=round(duration, 2),
                metadata={
                    "source_bucket": request.audio_bucket,
                    "source_key": request.audio_key,
                    "model_size": request.model_size,
                    "processing_time_seconds": round(processing_time, 2),
                    "file_size_bytes": file_size,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Step 7: Save transcript locally
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(result.dict(), f, indent=2, ensure_ascii=False)
            
            # Step 8: Upload to S3
            output_key = f"transcripts/{job_id}.json"
            
            upload_metadata = {
                "job_id": job_id,
                "source_bucket": request.audio_bucket,
                "source_key": request.audio_key,
                "language": info.language,
                "segments_count": str(len(transcript)),
                "speakers_detected": str(speakers_detected)
            }
            
            upload_to_s3(output_path, request.output_bucket, output_key, upload_metadata)
            
            # Step 9: Return success response
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s")
            
            return SpeechToTextResponse(
                success=True,
                job_id=job_id,
                language=info.language,
                language_probability=round(info.language_probability, 3),
                transcript_bucket=request.output_bucket,
                transcript_key=output_key,
                segments_count=len(transcript),
                diarization_performed=len(speaker_segments) > 0,
                speakers_detected=speakers_detected,
                duration_seconds=round(duration, 2),
                processing_time_seconds=round(processing_time, 2),
                message="Audio successfully transcribed"
            )
    
    except HTTPException:
        logger.error(f"Job {job_id} failed with HTTP error")
        raise
    
    except Exception as e:
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during processing: {str(e)}"
        )


# =====================================================
# HEALTH CHECK
# =====================================================
@router.get("/speech/health", tags=["Health"])
async def health_check():
    """Health check endpoint for speech-to-text service"""
    
    # Check if models are loaded
    whisper_loaded = _whisper_model is not None
    diarization_loaded = _diarization_pipeline is not None
    
    return {
        "status": "healthy",
        "service": "speech-to-text",
        "device": device,
        "compute_type": compute_type,
        "whisper_model_loaded": whisper_loaded,
        "diarization_available": HUGGINGFACE_TOKEN is not None,
        "diarization_loaded": diarization_loaded,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK),
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "max_audio_size_mb": MAX_AUDIO_SIZE // (1024 * 1024)
    }


# =====================================================
# MODEL INFO ENDPOINT
# =====================================================
@router.get("/speech/models", tags=["Speech"])
async def get_model_info():
    """Get information about available models and configurations"""
    return {
        "whisper_models": list(WHISPER_MODELS.keys()),
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_audio_formats": list(ALLOWED_AUDIO_EXTENSIONS),
        "max_file_size_mb": MAX_AUDIO_SIZE // (1024 * 1024),
        "max_duration_seconds": MAX_AUDIO_DURATION,
        "device": device,
        "diarization_available": HUGGINGFACE_TOKEN is not None
    }