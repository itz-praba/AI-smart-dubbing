import os
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

import boto3
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError, BotoCoreError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/ai", tags=["Audio"])

# Constants
TEMP_DIR = Path("temp")
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
FFMPEG_TIMEOUT = 600  # 10 minutes
MAX_FILENAME_LENGTH = 255

# Environment variable validation
REQUIRED_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION"
]

for env_var in REQUIRED_ENV_VARS:
    if not os.getenv(env_var):
        raise RuntimeError(f"Missing required environment variable: {env_var}")

# Validate FFMPEG path
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
if not shutil.which(FFMPEG_PATH):
    raise RuntimeError(f"FFmpeg not found at: {FFMPEG_PATH}. Please install FFmpeg or set FFMPEG_PATH environment variable.")

# Create temp directory
TEMP_DIR.mkdir(exist_ok=True)

# Initialize S3 client with retry configuration
try:
    from botocore.config import Config
    
    boto_config = Config(
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        },
        connect_timeout=10,
        read_timeout=60
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


# Pydantic Models
class VideoToAudioRequest(BaseModel):
    """Request model for video to audio conversion"""
    video_bucket: str = Field(..., min_length=1, max_length=63, description="Source S3 bucket name")
    video_key: str = Field(..., min_length=1, max_length=1024, description="Source video file key in S3")
    audio_bucket: str = Field(..., min_length=1, max_length=63, description="Destination S3 bucket name")
    output_key_prefix: Optional[str] = Field(None, max_length=512, description="Optional prefix for output audio key")
    
    @validator('video_bucket', 'audio_bucket')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name"""
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('video_key')
    def validate_video_key(cls, v):
        """Validate video key and extension"""
        # Prevent path traversal
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid video key: path traversal detected")
        
        # Check extension
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{ext}'. "
                f"Allowed formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )
        
        # Check filename length
        filename = Path(v).name
        if len(filename) > MAX_FILENAME_LENGTH:
            raise ValueError(f"Filename too long. Maximum length: {MAX_FILENAME_LENGTH}")
        
        return v
    
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
                "video_bucket": "my-video-bucket",
                "video_key": "videos/sample.mp4",
                "audio_bucket": "my-audio-bucket",
                "output_key_prefix": "audio/"
            }
        }


class VideoToAudioResponse(BaseModel):
    """Response model for video to audio conversion"""
    success: bool
    job_id: str
    audio_bucket: str
    audio_key: str
    file_size_bytes: int
    duration_seconds: Optional[float] = None
    message: str


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[str] = None


# Context manager for file cleanup
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


# Helper functions
def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Get just the filename without path
    filename = Path(filename).name
    
    # Remove or replace dangerous characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '._-':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    result = ''.join(safe_chars)
    
    # Ensure filename isn't empty
    if not result or result.startswith('.'):
        result = f"file_{uuid.uuid4().hex[:8]}"
    
    return result[:MAX_FILENAME_LENGTH]


def check_s3_file_exists(bucket: str, key: str) -> dict:
    """
    Check if S3 file exists and get metadata
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        Dictionary with file metadata
        
    Raises:
        HTTPException: If file not found or other S3 errors
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
                detail=f"Video file not found in bucket '{bucket}' with key '{key}'"
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
        
        # Verify download
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
    except Exception as e:
        logger.error(f"Unexpected download error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during download: {str(e)}"
        )


def convert_video_to_audio(video_path: str, audio_path: str) -> dict:
    """
    Convert video to audio using FFmpeg
    
    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
        
    Returns:
        Dictionary with conversion metadata
        
    Raises:
        HTTPException: If conversion fails
    """
    command = [
        FFMPEG_PATH,
        "-y",  # Overwrite output file
        "-i", video_path,  # Input file
        "-vn",  # No video
        "-ac", "1",  # Mono audio
        "-ar", "16000",  # Sample rate 16kHz
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
        "-af", "highpass=f=80,lowpass=f=8000,dynaudnorm",  # Audio filters
        "-map_metadata", "-1",  # Strip metadata
        "-fflags", "+bitexact",  # Ensure reproducible output
        audio_path
    ]
    
    try:
        logger.info(f"Starting FFmpeg conversion: {video_path} -> {audio_path}")
        start_time = datetime.now()
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Verify output file
        if not os.path.exists(audio_path):
            raise HTTPException(
                status_code=500,
                detail="FFmpeg completed but output file not found"
            )
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg produced empty output file"
            )
        
        logger.info(f"Conversion successful in {duration:.2f}s, output size: {file_size} bytes")
        
        return {
            'duration_seconds': duration,
            'output_size_bytes': file_size
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg conversion timeout after {FFMPEG_TIMEOUT}s")
        raise HTTPException(
            status_code=504,
            detail=f"Video conversion timeout ({FFMPEG_TIMEOUT}s). File may be too large or corrupted."
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        # Parse common FFmpeg errors
        stderr_lower = e.stderr.lower()
        if "invalid data" in stderr_lower or "corrupt" in stderr_lower:
            detail = "Video file appears to be corrupted or invalid"
        elif "no such file" in stderr_lower:
            detail = "Input file not found"
        elif "permission denied" in stderr_lower:
            detail = "Permission denied accessing file"
        else:
            detail = f"Video conversion failed: {e.stderr[:200]}"
        
        raise HTTPException(status_code=500, detail=detail)
    except Exception as e:
        logger.error(f"Unexpected FFmpeg error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during conversion: {str(e)}"
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
            "ServerSideEncryption": "AES256"  # Enable server-side encryption
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


# API Endpoint
@router.post(
    "/video-to-audio",
    response_model=VideoToAudioResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "File not found"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Server error"},
        504: {"model": ErrorResponse, "description": "Conversion timeout"}
    },
    summary="Convert video to audio",
    description="Downloads a video from S3, extracts audio track, and uploads the result as WAV"
)
async def video_to_audio(request: VideoToAudioRequest) -> VideoToAudioResponse:
    """
    Convert a video file to audio (WAV format)
    
    This endpoint:
    1. Downloads video from S3
    2. Extracts audio track using FFmpeg
    3. Converts to 16kHz mono WAV with audio normalization
    4. Uploads result to S3
    5. Cleans up temporary files
    
    Args:
        request: VideoToAudioRequest with bucket names and video key
        
    Returns:
        VideoToAudioResponse with job details and output location
        
    Raises:
        HTTPException: For various error conditions (file not found, too large, etc.)
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Starting job {job_id}: {request.video_bucket}/{request.video_key}")
    
    # Generate file paths
    video_filename = sanitize_filename(Path(request.video_key).name)
    video_path = str(TEMP_DIR / f"{job_id}_{video_filename}")
    audio_path = str(TEMP_DIR / f"{job_id}.wav")
    
    # Generate output key
    if request.output_key_prefix:
        audio_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}.wav"
    else:
        audio_key = f"audio/{job_id}.wav"
    
    try:
        # Use context manager for automatic cleanup
        with temp_files(video_path, audio_path):
            # Step 1: Check file exists and size
            file_metadata = check_s3_file_exists(request.video_bucket, request.video_key)
            file_size = file_metadata['size']
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large ({file_size / (1024*1024):.1f}MB). "
                           f"Maximum allowed: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
                )
            
            logger.info(f"File size: {file_size / (1024*1024):.2f}MB")
            
            # Step 2: Download video from S3
            download_from_s3(request.video_bucket, request.video_key, video_path)
            
            # Step 3: Convert to audio
            conversion_metadata = convert_video_to_audio(video_path, audio_path)
            
            # Step 4: Upload to S3
            upload_metadata = {
                "job_id": job_id,
                "source_bucket": request.video_bucket,
                "source_key": request.video_key,
                "conversion_timestamp": datetime.utcnow().isoformat(),
                "input_size_bytes": str(file_size),
                "output_size_bytes": str(conversion_metadata['output_size_bytes'])
            }
            
            upload_to_s3(audio_path, request.audio_bucket, audio_key, upload_metadata)
            
            # Step 5: Return success response
            logger.info(f"Job {job_id} completed successfully")
            
            return VideoToAudioResponse(
                success=True,
                job_id=job_id,
                audio_bucket=request.audio_bucket,
                audio_key=audio_key,
                file_size_bytes=conversion_metadata['output_size_bytes'],
                duration_seconds=conversion_metadata['duration_seconds'],
                message="Video successfully converted to audio"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        logger.error(f"Job {job_id} failed with HTTP error")
        raise
    
    except Exception as e:
        # Catch any unexpected errors
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during processing: {str(e)}"
        )


# Health check endpoint
@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "video-to-audio",
        "ffmpeg_available": shutil.which(FFMPEG_PATH) is not None,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }


# Optional: Add endpoint to check conversion status (for async processing)
@router.get("/job/{job_id}", tags=["Audio"])
async def get_job_status(job_id: str):
    """
    Get status of a conversion job
    
    Note: This is a placeholder. For production, you'd need to implement
    a job tracking system (e.g., using Redis, DynamoDB, or a database)
    """
    return {
        "job_id": job_id,
        "status": "This endpoint requires a job tracking system to be implemented",
        "message": "Consider using AWS Step Functions or a queue system for async processing"
    }