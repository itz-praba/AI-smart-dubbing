import os
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from contextlib import contextmanager

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
router = APIRouter(prefix="/ai", tags=["Video Rendering"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# FFmpeg configuration
FFMPEG_BIN = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_PATH", "ffprobe")

# Validate FFmpeg/FFprobe availability
if not shutil.which(FFMPEG_BIN):
    raise RuntimeError(f"FFmpeg not found at: {FFMPEG_BIN}")
if not shutil.which(FFPROBE_BIN):
    raise RuntimeError(f"FFprobe not found at: {FFPROBE_BIN}")

logger.info(f"FFmpeg: {FFMPEG_BIN}")
logger.info(f"FFprobe: {FFPROBE_BIN}")

# File constraints
MAX_VIDEO_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
MAX_AUDIO_SIZE = 500 * 1024 * 1024  # 500MB
MAX_AUDIO_TRACKS = 10
FFMPEG_TIMEOUT = 3600  # 1 hour

ALLOWED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}
ALLOWED_SUBTITLE_FORMATS = {'.srt', '.vtt', '.ass'}

# ISO 639-2 language codes (3-letter codes)
SUPPORTED_LANGUAGES = {
    "eng": "English",
    "spa": "Spanish",
    "fra": "French",
    "deu": "German",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",
    "jpn": "Japanese",
    "kor": "Korean",
    "zho": "Chinese",
    "ara": "Arabic",
    "hin": "Hindi",
    "pol": "Polish",
    "tur": "Turkish",
    "nld": "Dutch",
    "und": "Undetermined"
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
            read_timeout=300
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
class AudioTrack(BaseModel):
    """Audio track information"""
    audio_key: str = Field(..., min_length=1, description="S3 key of audio file")
    language: str = Field(..., min_length=3, max_length=3, description="ISO 639-2 language code")
    title: str = Field(..., min_length=1, max_length=100, description="Track title")
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code"""
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language code '{v}'. "
                f"Use ISO 639-2 3-letter codes like 'eng', 'spa', 'fra'"
            )
        return v
    
    @validator('audio_key')
    def validate_audio_key(cls, v):
        """Validate audio key"""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid audio key: path traversal detected")
        
        ext = Path(v).suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Allowed: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )
        return v


class VideoMergeRequest(BaseModel):
    """Request model for video merging"""
    video_bucket: str = Field(..., min_length=1, max_length=63)
    video_key: str = Field(..., min_length=1, max_length=1024)
    audio_tracks: List[AudioTrack] = Field(..., min_items=1, max_items=MAX_AUDIO_TRACKS)
    audio_bucket: str = Field(..., min_length=1, max_length=63)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    subtitle_key: Optional[str] = Field(None, max_length=1024)
    subtitle_bucket: Optional[str] = Field(None, max_length=63)
    output_key_prefix: Optional[str] = Field(None, max_length=512)
    video_codec: str = Field("copy", description="Video codec (copy, libx264, etc.)")
    audio_codec: str = Field("aac", description="Audio codec")
    audio_bitrate: str = Field("192k", description="Audio bitrate")
    
    @validator('video_bucket', 'audio_bucket', 'output_bucket', 'subtitle_bucket')
    def validate_bucket_name(cls, v):
        """Validate S3 bucket name"""
        if v is None:
            return v
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('video_key')
    def validate_video_key(cls, v):
        """Validate video key and extension"""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid video key: path traversal detected")
        
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format '{ext}'. "
                f"Allowed: {', '.join(ALLOWED_VIDEO_FORMATS)}"
            )
        return v
    
    @validator('subtitle_key')
    def validate_subtitle_key(cls, v):
        """Validate subtitle key"""
        if v is None:
            return v
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid subtitle key: path traversal detected")
        
        ext = Path(v).suffix.lower()
        if ext and ext not in ALLOWED_SUBTITLE_FORMATS:
            raise ValueError(
                f"Unsupported subtitle format '{ext}'. "
                f"Allowed: {', '.join(ALLOWED_SUBTITLE_FORMATS)}"
            )
        return v
    
    @validator('subtitle_bucket')
    def validate_subtitle_bucket_required(cls, v, values):
        """Ensure subtitle_bucket is provided if subtitle_key is provided"""
        if values.get('subtitle_key') and not v:
            raise ValueError("subtitle_bucket is required when subtitle_key is provided")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_bucket": "videos",
                "video_key": "source/original.mp4",
                "audio_tracks": [
                    {
                        "audio_key": "dubbed/english.wav",
                        "language": "eng",
                        "title": "English (Dubbed)"
                    },
                    {
                        "audio_key": "dubbed/spanish.wav",
                        "language": "spa",
                        "title": "Spanish (Dubbed)"
                    }
                ],
                "audio_bucket": "audio-files",
                "output_bucket": "final-videos",
                "subtitle_key": "subtitles/english.srt",
                "subtitle_bucket": "subtitle-files",
                "output_key_prefix": "final/",
                "video_codec": "copy",
                "audio_codec": "aac",
                "audio_bitrate": "192k"
            }
        }


class VideoMergeResponse(BaseModel):
    """Response model for video merging"""
    success: bool
    job_id: str
    output_bucket: str
    output_key: str
    video_duration_seconds: float
    audio_tracks_count: int
    has_subtitles: bool
    file_size_bytes: int
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
        result = f"video_{uuid.uuid4().hex[:8]}"
    
    return result[:max_length]


def check_s3_file(bucket: str, key: str) -> Dict:
    """Check if S3 file exists and get metadata"""
    if not S3_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured"
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
                detail=f"File not found: s3://{bucket}/{key}"
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
        raise HTTPException(status_code=503, detail="S3 not configured")
    
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
        raise HTTPException(status_code=503, detail="S3 not configured")
    
    try:
        extra_args = {
            "ContentType": "video/mp4",
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
# FFMPEG FUNCTIONS
# =====================================================
def run_ffmpeg_command(cmd: List[str], timeout: int = FFMPEG_TIMEOUT) -> str:
    """
    Run FFmpeg command with error handling
    
    Args:
        cmd: FFmpeg command as list
        timeout: Command timeout in seconds
        
    Returns:
        Command stdout
        
    Raises:
        HTTPException: If command fails
    """
    try:
        logger.info(f"Running FFmpeg command: {' '.join(cmd[:5])}...")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout
        )
        
        return result.stdout
        
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg command timeout after {timeout}s")
        raise HTTPException(
            status_code=504,
            detail=f"Video processing timeout ({timeout}s)"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {e.stderr[:200]}"
        )


def get_video_duration(video_path: str) -> float:
    """
    Get video duration using FFprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = run_ffmpeg_command(cmd, timeout=30)
        return float(result.strip())
    except Exception as e:
        logger.warning(f"Could not determine video duration: {e}")
        return 0.0


def get_video_info(video_path: str) -> Dict:
    """
    Get comprehensive video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video info
    """
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,duration",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        video_path
    ]
    
    try:
        import json
        result = run_ffmpeg_command(cmd, timeout=30)
        data = json.loads(result)
        
        video_stream = next(
            (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
            {}
        )
        
        return {
            'duration': float(data.get('format', {}).get('duration', 0)),
            'size': int(data.get('format', {}).get('size', 0)),
            'width': video_stream.get('width', 0),
            'height': video_stream.get('height', 0),
            'video_codec': video_stream.get('codec_name', 'unknown')
        }
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
        return {'duration': 0, 'size': 0, 'width': 0, 'height': 0, 'video_codec': 'unknown'}


def merge_video_with_audio(
    input_video: str,
    audio_files: List[Dict[str, str]],
    output_video: str,
    subtitle_file: Optional[str] = None,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k"
) -> None:
    """
    Merge video with multiple audio tracks and optional subtitles
    
    Args:
        input_video: Path to input video
        audio_files: List of audio file dicts with 'path', 'language', 'title'
        output_video: Path to output video
        subtitle_file: Optional path to subtitle file
        video_codec: Video codec (default: copy)
        audio_codec: Audio codec (default: aac)
        audio_bitrate: Audio bitrate (default: 192k)
    """
    # Validate input files exist
    if not os.path.exists(input_video):
        raise HTTPException(
            status_code=500,
            detail="Input video file not found"
        )
    
    for audio in audio_files:
        if not os.path.exists(audio['path']):
            raise HTTPException(
                status_code=500,
                detail=f"Audio file not found: {audio['path']}"
            )
    
    if subtitle_file and not os.path.exists(subtitle_file):
        raise HTTPException(
            status_code=500,
            detail="Subtitle file not found"
        )
    
    # Build FFmpeg command
    cmd = [FFMPEG_BIN, "-y"]
    
    # Add input video
    cmd.extend(["-i", input_video])
    
    # Add audio inputs
    for audio in audio_files:
        cmd.extend(["-i", audio['path']])
    
    # Add subtitle input
    if subtitle_file:
        cmd.extend(["-i", subtitle_file])
    
    # Map video stream
    cmd.extend(["-map", "0:v:0"])
    
    # Map audio streams
    for i in range(len(audio_files)):
        cmd.extend(["-map", f"{i+1}:a:0"])
    
    # Map subtitle stream
    if subtitle_file:
        subtitle_index = len(audio_files) + 1
        cmd.extend(["-map", f"{subtitle_index}:s:0"])
    
    # Set codecs
    cmd.extend([
        "-c:v", video_codec,
        "-c:a", audio_codec,
        "-b:a", audio_bitrate
    ])
    
    # Set subtitle codec
    if subtitle_file:
        cmd.extend(["-c:s", "mov_text"])
    
    # Add metadata for audio tracks
    for i, audio in enumerate(audio_files):
        cmd.extend([
            f"-metadata:s:a:{i}", f"language={audio['language']}",
            f"-metadata:s:a:{i}", f"title={audio['title']}"
        ])
    
    # Output file
    cmd.append(output_video)
    
    # Run command
    run_ffmpeg_command(cmd)
    
    # Verify output
    if not os.path.exists(output_video):
        raise HTTPException(
            status_code=500,
            detail="FFmpeg completed but output file not found"
        )
    
    if os.path.getsize(output_video) == 0:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg produced empty output file"
        )


# =====================================================
# API ENDPOINT
# =====================================================
@router.post(
    "/video-merge",
    response_model=VideoMergeResponse,
    summary="Merge video with dubbed audio tracks and subtitles",
    description="Combine video with multiple audio tracks (different languages) and optional subtitles"
)
async def video_merge(request: VideoMergeRequest) -> VideoMergeResponse:
    """
    Video merging with multiple audio tracks
    
    This endpoint:
    1. Downloads video from S3
    2. Downloads audio tracks from S3
    3. Downloads subtitles from S3 (optional)
    4. Merges all into single video file with multiple audio tracks
    5. Uploads final video to S3
    6. Cleans up temporary files
    
    Args:
        request: VideoMergeRequest with video, audio, and subtitle information
        
    Returns:
        VideoMergeResponse with final video location and metadata
    """
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(
        f"Starting video merge job {job_id}: "
        f"{request.video_bucket}/{request.video_key} + "
        f"{len(request.audio_tracks)} audio tracks"
    )
    
    # Create working directory
    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(exist_ok=True)
    
    temp_files_list = []
    
    try:
        # Step 1: Check and download video
        logger.info("Checking video file...")
        video_metadata = check_s3_file(request.video_bucket, request.video_key)
        
        if video_metadata['size'] > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Video too large ({video_metadata['size'] / (1024**3):.1f}GB). "
                       f"Maximum: {MAX_VIDEO_SIZE / (1024**3):.0f}GB"
            )
        
        video_filename = sanitize_filename(Path(request.video_key).name)
        video_path = str(work_dir / f"input_{video_filename}")
        temp_files_list.append(video_path)
        
        download_from_s3(request.video_bucket, request.video_key, video_path)
        
        # Get video info
        video_info = get_video_info(video_path)
        logger.info(
            f"Video info: {video_info['width']}x{video_info['height']}, "
            f"{video_info['duration']:.2f}s, codec: {video_info['video_codec']}"
        )
        
        # Step 2: Download audio tracks
        logger.info(f"Downloading {len(request.audio_tracks)} audio tracks...")
        audio_files = []
        
        for i, track in enumerate(request.audio_tracks):
            # Check file
            audio_metadata = check_s3_file(request.audio_bucket, track.audio_key)
            
            if audio_metadata['size'] > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio track {i} too large"
                )
            
            # Download
            audio_filename = sanitize_filename(Path(track.audio_key).name)
            audio_path = str(work_dir / f"audio_{i}_{audio_filename}")
            temp_files_list.append(audio_path)
            
            download_from_s3(request.audio_bucket, track.audio_key, audio_path)
            
            audio_files.append({
                'path': audio_path,
                'language': track.language,
                'title': track.title
            })
            
            logger.info(f"Downloaded audio track {i}: {track.title} ({track.language})")
        
        # Step 3: Download subtitles (optional)
        subtitle_path = None
        if request.subtitle_key and request.subtitle_bucket:
            logger.info("Downloading subtitles...")
            
            subtitle_filename = sanitize_filename(Path(request.subtitle_key).name)
            subtitle_path = str(work_dir / f"subtitle_{subtitle_filename}")
            temp_files_list.append(subtitle_path)
            
            download_from_s3(request.subtitle_bucket, request.subtitle_key, subtitle_path)
        
        # Step 4: Merge video with audio tracks
        logger.info("Merging video with audio tracks...")
        output_filename = f"{job_id}.mp4"
        output_path = str(work_dir / output_filename)
        temp_files_list.append(output_path)
        
        merge_video_with_audio(
            input_video=video_path,
            audio_files=audio_files,
            output_video=output_path,
            subtitle_file=subtitle_path,
            video_codec=request.video_codec,
            audio_codec=request.audio_codec,
            audio_bitrate=request.audio_bitrate
        )
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        logger.info(f"Output video size: {output_size / (1024**2):.2f}MB")
        
        # Step 5: Upload to S3
        if request.output_key_prefix:
            output_key = f"{request.output_key_prefix.rstrip('/')}/{output_filename}"
        else:
            output_key = f"merged_videos/{output_filename}"
        
        upload_metadata = {
            "job_id": job_id,
            "source_video": request.video_key,
            "audio_tracks_count": str(len(request.audio_tracks)),
            "has_subtitles": str(bool(subtitle_path)),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        upload_to_s3(output_path, request.output_bucket, output_key, upload_metadata)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"Job {job_id} completed successfully in {processing_time:.2f}s. "
            f"Output: s3://{request.output_bucket}/{output_key}"
        )
        
        return VideoMergeResponse(
            success=True,
            job_id=job_id,
            output_bucket=request.output_bucket,
            output_key=output_key,
            video_duration_seconds=round(video_info['duration'], 2),
            audio_tracks_count=len(request.audio_tracks),
            has_subtitles=bool(subtitle_path),
            file_size_bytes=output_size,
            processing_time_seconds=round(processing_time, 2),
            message=f"Successfully merged video with {len(request.audio_tracks)} audio tracks"
        )
    
    except HTTPException:
        logger.error(f"Job {job_id} failed with HTTP error")
        raise
    
    except Exception as e:
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during video merge: {str(e)}"
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
@router.get("/video-merge/languages", tags=["Video Rendering"])
async def get_supported_languages():
    """Get list of supported language codes"""
    return {
        "supported_languages": [
            {"code": code, "name": name}
            for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ],
        "total": len(SUPPORTED_LANGUAGES),
        "note": "Uses ISO 639-2 (3-letter codes)"
    }


@router.get("/video-merge/info", tags=["Video Rendering"])
async def get_merge_info():
    """Get information about video merging capabilities"""
    return {
        "max_video_size_gb": MAX_VIDEO_SIZE // (1024**3),
        "max_audio_size_mb": MAX_AUDIO_SIZE // (1024**2),
        "max_audio_tracks": MAX_AUDIO_TRACKS,
        "supported_video_formats": list(ALLOWED_VIDEO_FORMATS),
        "supported_audio_formats": list(ALLOWED_AUDIO_FORMATS),
        "supported_subtitle_formats": list(ALLOWED_SUBTITLE_FORMATS),
        "ffmpeg_timeout_seconds": FFMPEG_TIMEOUT,
        "s3_enabled": S3_ENABLED
    }


@router.get("/video-merge/health", tags=["Health"])
async def health_check():
    """Health check for video merging service"""
    return {
        "status": "healthy",
        "service": "video-merging",
        "ffmpeg_available": shutil.which(FFMPEG_BIN) is not None,
        "ffprobe_available": shutil.which(FFPROBE_BIN) is not None,
        "s3_enabled": S3_ENABLED,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }