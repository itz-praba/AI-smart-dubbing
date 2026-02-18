import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

import librosa
import soundfile as sf
import numpy as np
import pyrubberband as pyrb
import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/ai", tags=["Lip Sync"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Audio processing parameters
TARGET_SAMPLE_RATE = 16000
AUDIO_DTYPE = np.float32

# Time stretching limits (safe ranges)
MAX_SPEED_UP = 1.15      # +15% faster (safe for comprehension)
MAX_SLOW_DOWN = 0.85     # -15% slower (safe for naturalness)
MIN_SEGMENT_DURATION = 0.1  # 100ms minimum
MAX_SEGMENT_DURATION = 60.0  # 60 seconds maximum

# File size limits
MAX_AUDIO_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

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
        logger.warning(f"Missing environment variable: {env_var} - S3 features disabled")

# =====================================================
# AWS S3 CLIENT (Optional)
# =====================================================
s3_client = None
S3_ENABLED = all(os.getenv(var) for var in REQUIRED_ENV_VARS)

if S3_ENABLED:
    try:
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
        
        s3_client.list_buckets()
        logger.info("S3 client initialized successfully")
        
    except Exception as e:
        logger.warning(f"S3 client initialization failed: {e}. S3 features disabled.")
        S3_ENABLED = False


# =====================================================
# PYDANTIC MODELS
# =====================================================
class TimedSegment(BaseModel):
    """Segment with timing and audio file"""
    index: int = Field(..., ge=0)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    tts_audio_key: str = Field(..., min_length=1)
    
    @validator('end')
    def validate_end_time(cls, v, values):
        """Ensure end time is after start time"""
        if 'start' in values:
            duration = v - values['start']
            if duration < MIN_SEGMENT_DURATION:
                raise ValueError(
                    f"Segment too short ({duration:.2f}s). "
                    f"Minimum: {MIN_SEGMENT_DURATION}s"
                )
            if duration > MAX_SEGMENT_DURATION:
                raise ValueError(
                    f"Segment too long ({duration:.2f}s). "
                    f"Maximum: {MAX_SEGMENT_DURATION}s"
                )
        return v
    
    @validator('tts_audio_key')
    def validate_audio_key(cls, v):
        """Validate audio file key"""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid audio key: path traversal detected")
        
        ext = Path(v).suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Allowed: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )
        return v


class LipSyncRequest(BaseModel):
    """Request model for lip-sync audio alignment"""
    segments: List[TimedSegment] = Field(..., min_items=1, max_items=1000)
    tts_audio_bucket: str = Field(..., min_length=1, max_length=63)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    output_key_prefix: Optional[str] = Field(None, max_length=512)
    enable_time_stretch: bool = Field(True, description="Enable time-stretching for alignment")
    
    @validator('tts_audio_bucket', 'output_bucket')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name"""
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('segments')
    def validate_segments_timing(cls, v):
        """Validate segments don't overlap and are in order"""
        if not v:
            return v
        
        # Sort by index
        sorted_segments = sorted(v, key=lambda s: s.index)
        
        # Check for overlaps
        for i in range(len(sorted_segments) - 1):
            current = sorted_segments[i]
            next_seg = sorted_segments[i + 1]
            
            if current.end > next_seg.start:
                raise ValueError(
                    f"Segments {current.index} and {next_seg.index} overlap: "
                    f"segment {current.index} ends at {current.end}s, "
                    f"but segment {next_seg.index} starts at {next_seg.start}s"
                )
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "segments": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 3.5,
                        "tts_audio_key": "tts/seg_0.wav"
                    },
                    {
                        "index": 1,
                        "start": 3.5,
                        "end": 7.2,
                        "tts_audio_key": "tts/seg_1.wav"
                    }
                ],
                "tts_audio_bucket": "tts-audio",
                "output_bucket": "aligned-audio",
                "output_key_prefix": "aligned/",
                "enable_time_stretch": True
            }
        }


class AlignedSegmentInfo(BaseModel):
    """Information about an aligned segment"""
    index: int
    original_duration: float
    target_duration: float
    aligned_duration: float
    time_stretch_ratio: Optional[float] = None
    method: str  # "time_stretch", "silence_added", "exact_match"


class LipSyncResponse(BaseModel):
    """Response model for lip-sync audio alignment"""
    success: bool
    job_id: str
    output_bucket: str
    output_key: str
    segments_processed: int
    total_duration_seconds: float
    aligned_segments: List[AlignedSegmentInfo]
    processing_time_seconds: float
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
    """Sanitize filename to prevent security issues"""
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


def check_s3_file(bucket: str, key: str) -> Dict:
    """Check if S3 file exists and get metadata"""
    if not S3_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured. Set AWS credentials in environment variables."
        )
    
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
                detail=f"Audio file not found: s3://{bucket}/{key}"
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
    """Download file from S3"""
    if not S3_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured"
        )
    
    try:
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3_client.download_file(bucket, key, local_path)
        
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise HTTPException(
                status_code=500,
                detail="Download failed or file is empty"
            )
        
    except ClientError as e:
        logger.error(f"S3 download error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download from S3: {e.response['Error']['Message']}"
        )


def upload_to_s3(local_path: str, bucket: str, key: str, metadata: dict = None) -> None:
    """Upload file to S3"""
    if not S3_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured"
        )
    
    try:
        extra_args = {
            "ContentType": "audio/wav",
            "ServerSideEncryption": "AES256"
        }
        
        if metadata:
            extra_args["Metadata"] = metadata
        
        logger.info(f"Uploading to s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        
    except ClientError as e:
        logger.error(f"S3 upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to S3: {e.response['Error']['Message']}"
        )


# =====================================================
# AUDIO PROCESSING FUNCTIONS
# =====================================================
def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file
    
    Args:
        path: Path to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, dtype=AUDIO_DTYPE)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio from {path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load audio file: {str(e)}"
        )


def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    """Get audio duration in seconds"""
    return float(librosa.get_duration(y=audio, sr=sr))


def write_audio(path: str, audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    """
    Write audio to file
    
    Args:
        path: Output file path
        audio: Audio data
        sr: Sample rate
    """
    try:
        sf.write(path, audio, sr)
    except Exception as e:
        logger.error(f"Failed to write audio to {path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write audio file: {str(e)}"
        )


def add_silence(audio: np.ndarray, silence_sec: float, sr: int) -> np.ndarray:
    """
    Add silence to the end of audio
    
    Args:
        audio: Input audio
        silence_sec: Duration of silence to add
        sr: Sample rate
        
    Returns:
        Audio with silence appended
    """
    if silence_sec <= 0:
        return audio
    
    silence_samples = int(silence_sec * sr)
    silence = np.zeros(silence_samples, dtype=AUDIO_DTYPE)
    return np.concatenate([audio, silence])


def time_stretch_audio(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    High-quality time-stretch without pitch distortion
    
    Args:
        audio: Input audio
        sr: Sample rate
        ratio: Time stretch ratio (>1.0 speeds up, <1.0 slows down)
        
    Returns:
        Time-stretched audio
    """
    try:
        # Clamp ratio to safe limits
        ratio = max(MAX_SLOW_DOWN, min(ratio, MAX_SPEED_UP))
        
        stretched = pyrb.time_stretch(audio, sr, ratio)
        return stretched.astype(AUDIO_DTYPE)
        
    except Exception as e:
        logger.error(f"Time stretching failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Time stretching failed: {str(e)}"
        )


def align_segment(
    tts_audio: np.ndarray,
    sr: int,
    target_duration: float,
    enable_time_stretch: bool = True
) -> Tuple[np.ndarray, AlignedSegmentInfo]:
    """
    Align TTS audio to match target duration
    
    Args:
        tts_audio: Input TTS audio
        sr: Sample rate
        target_duration: Target duration in seconds
        enable_time_stretch: Whether to use time-stretching
        
    Returns:
        Tuple of (aligned_audio, segment_info)
    """
    tts_duration = get_audio_duration(tts_audio, sr)
    
    logger.debug(
        f"Aligning segment: TTS={tts_duration:.3f}s, Target={target_duration:.3f}s"
    )
    
    # Case 1: Exact match (within 50ms tolerance)
    if abs(tts_duration - target_duration) < 0.05:
        return tts_audio, AlignedSegmentInfo(
            index=0,
            original_duration=tts_duration,
            target_duration=target_duration,
            aligned_duration=tts_duration,
            time_stretch_ratio=None,
            method="exact_match"
        )
    
    # Case 2: TTS too long - need to speed up
    if tts_duration > target_duration:
        if not enable_time_stretch:
            # Just trim
            max_samples = int(target_duration * sr)
            trimmed = tts_audio[:max_samples]
            
            return trimmed, AlignedSegmentInfo(
                index=0,
                original_duration=tts_duration,
                target_duration=target_duration,
                aligned_duration=target_duration,
                time_stretch_ratio=None,
                method="trimmed"
            )
        
        # Calculate required speed-up ratio
        ratio = tts_duration / target_duration
        actual_ratio = min(ratio, MAX_SPEED_UP)
        
        # Time-stretch
        stretched = time_stretch_audio(tts_audio, sr, actual_ratio)
        
        # Trim if still slightly too long
        max_samples = int(target_duration * sr)
        final_audio = stretched[:max_samples]
        
        final_duration = get_audio_duration(final_audio, sr)
        
        return final_audio, AlignedSegmentInfo(
            index=0,
            original_duration=tts_duration,
            target_duration=target_duration,
            aligned_duration=final_duration,
            time_stretch_ratio=actual_ratio,
            method="time_stretch"
        )
    
    # Case 3: TTS too short - add silence
    else:
        silence_needed = target_duration - tts_duration
        final_audio = add_silence(tts_audio, silence_needed, sr)
        
        final_duration = get_audio_duration(final_audio, sr)
        
        return final_audio, AlignedSegmentInfo(
            index=0,
            original_duration=tts_duration,
            target_duration=target_duration,
            aligned_duration=final_duration,
            time_stretch_ratio=None,
            method="silence_added"
        )


def concatenate_audio_files(audio_files: List[np.ndarray], sr: int) -> np.ndarray:
    """
    Concatenate multiple audio arrays
    
    Args:
        audio_files: List of audio arrays
        sr: Sample rate
        
    Returns:
        Concatenated audio
    """
    if not audio_files:
        raise ValueError("No audio files to concatenate")
    
    return np.concatenate(audio_files).astype(AUDIO_DTYPE)


# =====================================================
# API ENDPOINT
# =====================================================
@router.post(
    "/lip-sync-align",
    response_model=LipSyncResponse,
    summary="Align dubbed audio segments with original timing",
    description="Align TTS-generated audio segments to match original video timing for lip-sync"
)
async def lip_sync_align(request: LipSyncRequest) -> LipSyncResponse:
    """
    Lip-sync audio alignment
    
    This endpoint:
    1. Downloads TTS audio segments from S3
    2. Aligns each segment to match original timing
    3. Uses time-stretching (speed up) or silence padding (slow down)
    4. Concatenates aligned segments
    5. Uploads final aligned audio to S3
    6. Cleans up temporary files
    
    Args:
        request: LipSyncRequest with segments and audio locations
        
    Returns:
        LipSyncResponse with aligned audio location and segment info
    """
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(
        f"Starting lip-sync alignment job {job_id}: "
        f"{len(request.segments)} segments"
    )
    
    # Create working directory
    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(exist_ok=True)
    
    temp_files_list = []
    aligned_segments_info = []
    aligned_audio_arrays = []
    
    try:
        # Process each segment
        for seg in sorted(request.segments, key=lambda s: s.index):
            logger.info(f"Processing segment {seg.index}")
            
            # Download TTS audio
            tts_filename = sanitize_filename(Path(seg.tts_audio_key).name)
            tts_path = str(work_dir / f"seg_{seg.index}_{tts_filename}")
            temp_files_list.append(tts_path)
            
            # Check file exists and size
            file_metadata = check_s3_file(
                request.tts_audio_bucket,
                seg.tts_audio_key
            )
            
            if file_metadata['size'] > MAX_AUDIO_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Segment {seg.index} audio file too large"
                )
            
            download_from_s3(
                request.tts_audio_bucket,
                seg.tts_audio_key,
                tts_path
            )
            
            # Load audio
            tts_audio, sr = load_audio(tts_path)
            
            # Align segment
            target_duration = seg.end - seg.start
            aligned_audio, seg_info = align_segment(
                tts_audio,
                sr,
                target_duration,
                enable_time_stretch=request.enable_time_stretch
            )
            
            # Update segment info
            seg_info.index = seg.index
            aligned_segments_info.append(seg_info)
            aligned_audio_arrays.append(aligned_audio)
            
            logger.info(
                f"Segment {seg.index} aligned: "
                f"{seg_info.original_duration:.2f}s → {seg_info.aligned_duration:.2f}s "
                f"(method: {seg_info.method})"
            )
        
        # Concatenate all aligned segments
        logger.info("Concatenating aligned segments...")
        final_audio = concatenate_audio_files(aligned_audio_arrays, TARGET_SAMPLE_RATE)
        
        # Write final audio
        final_audio_path = str(work_dir / "aligned_final.wav")
        temp_files_list.append(final_audio_path)
        write_audio(final_audio_path, final_audio, TARGET_SAMPLE_RATE)
        
        # Calculate total duration
        total_duration = get_audio_duration(final_audio, TARGET_SAMPLE_RATE)
        
        # Generate output key
        if request.output_key_prefix:
            output_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}.wav"
        else:
            output_key = f"aligned_audio/{job_id}.wav"
        
        # Upload to S3
        upload_metadata = {
            "job_id": job_id,
            "segments_count": str(len(request.segments)),
            "total_duration": str(round(total_duration, 2)),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        upload_to_s3(final_audio_path, request.output_bucket, output_key, upload_metadata)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Job {job_id} completed successfully in {processing_time:.2f}s. "
            f"Generated {total_duration:.2f}s of aligned audio"
        )
        
        return LipSyncResponse(
            success=True,
            job_id=job_id,
            output_bucket=request.output_bucket,
            output_key=output_key,
            segments_processed=len(request.segments),
            total_duration_seconds=round(total_duration, 2),
            aligned_segments=aligned_segments_info,
            processing_time_seconds=round(processing_time, 2),
            message=f"Successfully aligned {len(request.segments)} segments"
        )
    
    except HTTPException:
        logger.error(f"Job {job_id} failed with HTTP error")
        raise
    
    except Exception as e:
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during alignment: {str(e)}"
        )
    
    finally:
        # Cleanup
        for file_path in temp_files_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")
        
        # Remove work directory
        try:
            if work_dir.exists():
                work_dir.rmdir()
        except OSError:
            pass


# =====================================================
# UTILITY ENDPOINTS
# =====================================================
@router.get("/lip-sync/info", tags=["Lip Sync"])
async def get_alignment_info():
    """Get information about lip-sync alignment capabilities"""
    return {
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "max_speed_up": MAX_SPEED_UP,
        "max_slow_down": MAX_SLOW_DOWN,
        "min_segment_duration": MIN_SEGMENT_DURATION,
        "max_segment_duration": MAX_SEGMENT_DURATION,
        "supported_audio_formats": list(ALLOWED_AUDIO_FORMATS),
        "max_audio_file_size_mb": MAX_AUDIO_FILE_SIZE // (1024 * 1024),
        "s3_enabled": S3_ENABLED
    }


@router.get("/lip-sync/health", tags=["Health"])
async def health_check():
    """Health check for lip-sync alignment service"""
    return {
        "status": "healthy",
        "service": "lip-sync-alignment",
        "s3_enabled": S3_ENABLED,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }