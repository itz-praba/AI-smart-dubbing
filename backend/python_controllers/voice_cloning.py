import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any

import torch
import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError
from botocore.config import Config

# Try importing TTS, provide helpful error if missing
try:
    from TTS.api import TTS
except ImportError:
    raise RuntimeError(
        "Coqui TTS not installed. Install with: pip install TTS"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/ai", tags=["Voice Cloning"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Voice cloning service using device: {DEVICE}")

# Supported languages for XTTS v2
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
    "hi": "Hindi",
}

# File constraints
MAX_SPEAKER_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TEXT_LENGTH = 5000
MIN_SPEAKER_AUDIO_DURATION = 3  # seconds
MAX_OUTPUT_DURATION = 300  # 5 minutes

ALLOWED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

# Model configuration
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

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

# =====================================================
# AWS S3 CLIENT
# =====================================================
try:
    boto_config = Config(
        retries={
            'max_attempts': 5,
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
# LOAD XTTS MODEL
# =====================================================
_tts_model = None

def get_tts_model() -> TTS:
    """
    Get or initialize TTS model with lazy loading
    
    Returns:
        TTS model instance
    """
    global _tts_model
    
    if _tts_model is None:
        try:
            logger.info(f"Loading XTTS v2 model on {DEVICE}...")
            _tts_model = TTS(
                model_name=XTTS_MODEL_NAME,
                progress_bar=True,
                gpu=(DEVICE == "cuda")
            )
            
            # Move to device if needed
            if DEVICE == "cuda":
                _tts_model.to(DEVICE)
            
            logger.info("XTTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise RuntimeError(f"Failed to initialize TTS model: {str(e)}")
    
    return _tts_model


# =====================================================
# PYDANTIC MODELS
# =====================================================
class VoiceCloningRequest(BaseModel):
    """Request model for voice cloning TTS"""
    speaker_audio_bucket: str = Field(..., min_length=1, max_length=63)
    speaker_audio_key: str = Field(..., min_length=1, max_length=1024)
    translated_text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    target_language: str = Field(..., min_length=2, max_length=10)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    output_key_prefix: Optional[str] = Field(None, max_length=512)
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    
    @validator('speaker_audio_bucket', 'output_bucket')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name"""
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('speaker_audio_key')
    def validate_audio_key(cls, v):
        """Validate speaker audio key and extension"""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid audio key: path traversal detected")
        
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Allowed formats: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )
        
        return v
    
    @validator('target_language')
    def validate_language(cls, v):
        """Validate language code"""
        v = v.lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )
        return v
    
    @validator('translated_text')
    def validate_text(cls, v):
        """Validate text is not just whitespace"""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()
    
    @validator('output_key_prefix')
    def validate_output_prefix(cls, v):
        """Validate output key prefix"""
        if v is not None:
            if '..' in v or v.startswith('/'):
                raise ValueError("Invalid output prefix: path traversal detected")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "speaker_audio_bucket": "audio-samples",
                "speaker_audio_key": "speakers/john_voice.wav",
                "translated_text": "Hello, this is a test of voice cloning technology.",
                "target_language": "en",
                "output_bucket": "dubbed-audio",
                "output_key_prefix": "cloned/",
                "speed": 1.0
            }
        }


class VoiceCloningResponse(BaseModel):
    """Response model for voice cloning TTS"""
    success: bool
    job_id: str
    output_bucket: str
    output_key: str
    language: str
    text_length: int
    audio_duration_seconds: Optional[float] = None
    processing_time_seconds: float
    device: str
    message: str


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
                detail=f"Speaker audio file not found in bucket '{bucket}' with key '{key}'"
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
            "ContentType": "audio/wav",
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


def get_audio_duration(file_path: str) -> float:
    """
    Get audio file duration in seconds
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        duration = waveform.shape[1] / sample_rate
        return float(duration)
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


def validate_speaker_audio(file_path: str) -> None:
    """
    Validate speaker audio file
    
    Args:
        file_path: Path to audio file
        
    Raises:
        HTTPException: If audio is invalid
    """
    try:
        duration = get_audio_duration(file_path)
        
        if duration < MIN_SPEAKER_AUDIO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker audio too short ({duration:.1f}s). "
                       f"Minimum required: {MIN_SPEAKER_AUDIO_DURATION}s for good quality cloning."
            )
        
        logger.info(f"Speaker audio validated: {duration:.2f}s duration")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Could not validate audio: {e}")
        # Don't fail if validation has issues, just log


# =====================================================
# API ENDPOINT
# =====================================================
@router.post(
    "/voice-clone-tts",
    response_model=VoiceCloningResponse,
    summary="Generate speech with cloned voice",
    description="Clone a speaker's voice and generate speech in the target language"
)
async def voice_clone_tts(request: VoiceCloningRequest) -> VoiceCloningResponse:
    """
    Voice cloning and text-to-speech generation
    
    This endpoint:
    1. Downloads speaker reference audio from S3
    2. Clones the speaker's voice using XTTS v2
    3. Generates speech from text in target language
    4. Uploads generated audio to S3
    5. Cleans up temporary files
    
    Args:
        request: VoiceCloningRequest with speaker audio and text parameters
        
    Returns:
        VoiceCloningResponse with job details and output location
    """
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(
        f"Starting voice cloning job {job_id}: "
        f"{request.speaker_audio_bucket}/{request.speaker_audio_key} -> "
        f"{request.target_language}, text length: {len(request.translated_text)}"
    )
    
    # Generate file paths
    speaker_filename = sanitize_filename(Path(request.speaker_audio_key).name)
    speaker_wav = str(TEMP_DIR / f"{job_id}_speaker_{speaker_filename}")
    output_wav = str(TEMP_DIR / f"{job_id}_dubbed.wav")
    
    # Generate output key
    if request.output_key_prefix:
        output_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}.wav"
    else:
        output_key = f"dubbed_audio/{job_id}.wav"
    
    try:
        with temp_files(speaker_wav, output_wav):
            # Step 1: Check speaker audio file exists and size
            file_metadata = check_s3_file(
                request.speaker_audio_bucket,
                request.speaker_audio_key
            )
            file_size = file_metadata['size']
            
            if file_size > MAX_SPEAKER_AUDIO_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Speaker audio too large ({file_size / (1024*1024):.1f}MB). "
                           f"Maximum allowed: {MAX_SPEAKER_AUDIO_SIZE / (1024*1024):.0f}MB"
                )
            
            logger.info(f"Speaker audio size: {file_size / (1024*1024):.2f}MB")
            
            # Step 2: Download speaker audio from S3
            download_from_s3(
                request.speaker_audio_bucket,
                request.speaker_audio_key,
                speaker_wav
            )
            
            # Step 3: Validate speaker audio
            validate_speaker_audio(speaker_wav)
            
            # Step 4: Get TTS model
            tts_model = get_tts_model()
            
            # Step 5: Voice cloning + TTS generation
            logger.info("Generating cloned voice speech...")
            try:
                tts_model.tts_to_file(
                    text=request.translated_text,
                    speaker_wav=speaker_wav,
                    language=request.target_language,
                    file_path=output_wav,
                    speed=request.speed
                )
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Voice cloning failed: {str(e)}"
                )
            
            # Step 6: Verify output file was created
            if not os.path.exists(output_wav):
                raise HTTPException(
                    status_code=500,
                    detail="TTS completed but output file not found"
                )
            
            output_size = os.path.getsize(output_wav)
            if output_size == 0:
                raise HTTPException(
                    status_code=500,
                    detail="TTS produced empty output file"
                )
            
            # Get audio duration
            audio_duration = get_audio_duration(output_wav)
            
            if audio_duration > MAX_OUTPUT_DURATION:
                logger.warning(
                    f"Generated audio is very long: {audio_duration:.1f}s. "
                    f"Consider splitting text into smaller chunks."
                )
            
            logger.info(
                f"Generated audio: {output_size} bytes, "
                f"duration: {audio_duration:.2f}s"
            )
            
            # Step 7: Upload to S3
            upload_metadata = {
                "job_id": job_id,
                "source_bucket": request.speaker_audio_bucket,
                "source_key": request.speaker_audio_key,
                "language": request.target_language,
                "text_length": str(len(request.translated_text)),
                "audio_duration": str(round(audio_duration, 2)),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            upload_to_s3(output_wav, request.output_bucket, output_key, upload_metadata)
            
            # Step 8: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Job {job_id} completed successfully in {processing_time:.2f}s. "
                f"Generated {audio_duration:.2f}s of audio"
            )
            
            return VoiceCloningResponse(
                success=True,
                job_id=job_id,
                output_bucket=request.output_bucket,
                output_key=output_key,
                language=SUPPORTED_LANGUAGES.get(
                    request.target_language,
                    request.target_language
                ),
                text_length=len(request.translated_text),
                audio_duration_seconds=round(audio_duration, 2),
                processing_time_seconds=round(processing_time, 2),
                device=DEVICE,
                message=f"Voice cloned TTS generated successfully: {audio_duration:.2f}s audio"
            )
    
    except HTTPException:
        logger.error(f"Job {job_id} failed with HTTP error")
        raise
    
    except Exception as e:
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during voice cloning: {str(e)}"
        )


# =====================================================
# UTILITY ENDPOINTS
# =====================================================
@router.get("/voice-clone/languages", tags=["Voice Cloning"])
async def get_supported_languages():
    """Get list of supported languages for voice cloning"""
    return {
        "supported_languages": [
            {"code": code, "name": name}
            for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ],
        "total": len(SUPPORTED_LANGUAGES)
    }


@router.get("/voice-clone/info", tags=["Voice Cloning"])
async def get_model_info():
    """Get information about the voice cloning model"""
    model_loaded = _tts_model is not None
    
    return {
        "model": XTTS_MODEL_NAME,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model_loaded,
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "max_text_length": MAX_TEXT_LENGTH,
        "max_speaker_audio_size_mb": MAX_SPEAKER_AUDIO_SIZE // (1024 * 1024),
        "min_speaker_duration_seconds": MIN_SPEAKER_AUDIO_DURATION,
        "supported_audio_formats": list(ALLOWED_AUDIO_FORMATS)
    }


@router.get("/voice-clone/health", tags=["Health"])
async def health_check():
    """Health check for voice cloning service"""
    model_loaded = _tts_model is not None
    
    return {
        "status": "healthy",
        "service": "voice-cloning",
        "device": DEVICE,
        "model_loaded": model_loaded,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }