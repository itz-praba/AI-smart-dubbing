import os
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import soundfile as sf
import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError
from botocore.config import Config

# Import mastering pipeline
# FIX B3: Removed top-level import of audio_mastering to break the eager S3
# initialisation that crashed the app at startup when AWS credentials were absent.
# Constants are redefined here with the same defaults; master_dubbed_audio is
# imported lazily inside video_merge() only when actually needed.
TARGET_LUFS       = -16.0
TARGET_TRUE_PEAK  = -1.5
DIALOGUE_PEAK_DB  = -3.0
BGM_DUCK_DB       = -12.0
COMP_THRESHOLD_DB = -24.0
COMP_RATIO        = 4.0
COMP_MAKEUP_DB    = 6.0

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
    "ukr": "Ukrainian",   # BUG FIX 7 — was missing
    "vie": "Vietnamese",  # BUG FIX 7 — was missing
    "ces": "Czech",       # BUG FIX 7 — was missing
    "dan": "Danish",      # BUG FIX 7 — was missing
    "fin": "Finnish",     # BUG FIX 7 — was missing
    "nor": "Norwegian",   # BUG FIX 7 — was missing
    "swe": "Swedish",     # BUG FIX 7 — was missing
    "ell": "Greek",       # BUG FIX 7 — was missing
    "heb": "Hebrew",      # BUG FIX 7 — was missing
    "tha": "Thai",        # BUG FIX 7 — was missing
    "tam": "Tamil",
    "tel": "Telugu",      # BUG FIX 7 — was missing
    "urd": "Urdu",        # BUG FIX 7 — was missing
    "tgl": "Tanglish",
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

    # ── Audio mastering options ────────────────────────────────────
    enable_audio_mastering: bool  = Field(True,  description="Run EQ+compress+loudness+ducking pipeline")
    enable_bgm_ducking: bool      = Field(True,  description="Duck background music under dialogue")
    target_lufs: float            = Field(TARGET_LUFS,        ge=-30.0, le=-6.0)
    target_true_peak: float       = Field(TARGET_TRUE_PEAK,   ge=-6.0,  le=-0.5)
    dialogue_peak_db: float       = Field(DIALOGUE_PEAK_DB,   ge=-12.0, le=0.0)
    bgm_duck_db: float            = Field(BGM_DUCK_DB,        ge=-30.0, le=0.0)
    comp_threshold_db: float      = Field(COMP_THRESHOLD_DB,  ge=-40.0, le=0.0)
    comp_ratio: float             = Field(COMP_RATIO,         ge=1.0,   le=20.0)
    comp_makeup_db: float         = Field(COMP_MAKEUP_DB,     ge=0.0,   le=24.0)
    
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


def _get_media_duration(path: str) -> float:
    """Return duration in seconds of any audio or video file via ffprobe."""
    try:
        result = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path
            ],
            capture_output=True, text=True, timeout=30, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _pad_or_trim_audio(audio_path: str, target_duration: float, work_dir: Path) -> str:
    """
    BUG FIX 5 — Audio/video duration mismatch:
    Ensure the dubbed audio track is exactly `target_duration` seconds long
    before muxing, so the final video never ends with silence or gets cut short.

    Strategy:
      • audio shorter than video → pad with silence using ffmpeg apad filter
      • audio longer  than video → trim with ffmpeg atrim filter
      • within 50ms tolerance    → use as-is (avoids unnecessary re-encode)

    Returns the path of the duration-matched audio (may be the original path
    if no adjustment was needed).
    """
    audio_dur = _get_media_duration(audio_path)
    if audio_dur <= 0:
        logger.warning(f"Could not determine audio duration for {audio_path}; skipping adjustment")
        return audio_path

    diff = abs(audio_dur - target_duration)
    if diff < 0.05:
        logger.debug(f"Audio duration {audio_dur:.3f}s within 50ms of video {target_duration:.3f}s; no adjustment")
        return audio_path

    out_path = str(work_dir / f"dur_fixed_{Path(audio_path).name}")

    if audio_dur < target_duration:
        # Pad with silence to reach target duration
        logger.info(
            f"Audio ({audio_dur:.2f}s) shorter than video ({target_duration:.2f}s); "
            f"padding {target_duration - audio_dur:.2f}s of silence"
        )
        af = f"apad=pad_dur={target_duration - audio_dur:.6f}"
        trim_args: List[str] = []
    else:
        # Trim audio to video length
        logger.info(
            f"Audio ({audio_dur:.2f}s) longer than video ({target_duration:.2f}s); "
            f"trimming {audio_dur - target_duration:.2f}s"
        )
        af = f"atrim=end={target_duration:.6f}"
        trim_args = []

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", audio_path,
        "-af", af,
        "-ar", "24000", "-ac", "1",
        "-c:a", "pcm_s16le",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        logger.info(f"Duration-adjusted audio written to {out_path}")
        return out_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Audio duration adjustment failed ({e.stderr[:200]}); using original")
        return audio_path


def merge_video_with_audio(
    input_video: str,
    audio_files: List[Dict[str, str]],
    output_video: str,
    subtitle_file: Optional[str] = None,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    work_dir: Optional[Path] = None,
) -> None:
    """
    Merge video with multiple audio tracks and optional subtitles.

    BUG FIX 5 — Duration mismatch: each audio track is padded or trimmed to
    exactly match the video duration before muxing.

    BUG FIX 6 — video_codec="copy" breaks when audio stream count changes:
    "copy" re-muxes the raw video bitstream unchanged. This is fast and lossless
    but causes container-level errors on some players when the audio layout in the
    output differs from what was encoded into the original container header.
    We keep "copy" as the default for performance but automatically fall back to
    "libx264" (with -crf 18 to be visually lossless) if the copy mux fails, rather
    than surfacing a confusing FFmpeg error to the caller.
    """
    if not os.path.exists(input_video):
        raise HTTPException(status_code=500, detail="Input video file not found")

    for audio in audio_files:
        if not os.path.exists(audio['path']):
            raise HTTPException(status_code=500, detail=f"Audio file not found: {audio['path']}")

    if subtitle_file and not os.path.exists(subtitle_file):
        raise HTTPException(status_code=500, detail="Subtitle file not found")

    # BUG FIX 5: pad/trim each audio track to exactly match video duration
    video_dur = _get_media_duration(input_video)
    if video_dur > 0 and work_dir is not None:
        adjusted_audio_files = []
        for af in audio_files:
            fixed_path = _pad_or_trim_audio(af['path'], video_dur, work_dir)
            adjusted_audio_files.append({**af, 'path': fixed_path})
        audio_files = adjusted_audio_files

    def _build_cmd(v_codec: str) -> List[str]:
        cmd = [FFMPEG_BIN, "-y"]

        # ── ROOT CAUSE 1 FIX: -an on the video input ─────────────────────────
        # Without -an, FFmpeg copies the moov atom from the source container,
        # which still declares the original audio stream. Players that parse the
        # container header (VLC, QuickTime, Android MediaPlayer) auto-select
        # stream 0:a:0 — the original language audio — even when it was not
        # mapped into the output, causing the original voice to bleed through
        # the dubbed track as a faint background echo.
        # -an on the INPUT (not the output) tells FFmpeg to treat the source as
        # video-only before mapping, so the original audio stream is never
        # carried into the output container at all.
        cmd.extend(["-an", "-i", input_video])

        for audio in audio_files:
            cmd.extend(["-i", audio['path']])
        if subtitle_file:
            cmd.extend(["-i", subtitle_file])

        # ── ROOT CAUSE 2 FIX: explicit negative audio map ────────────────────
        # Belt-and-suspenders: even with -an on the input, explicitly block
        # any accidental inclusion of stream 0's audio. -map -0:a means
        # "never include any audio from input 0, period."
        cmd.extend(["-map", "0:v:0"])
        cmd.extend(["-map", "-0:a"])   # ROOT CAUSE 2: hard block original audio

        for i in range(len(audio_files)):
            cmd.extend(["-map", f"{i+1}:a:0"])
        if subtitle_file:
            cmd.extend(["-map", f"{len(audio_files)+1}:s:0"])

        cmd.extend(["-c:v", v_codec])
        if v_codec != "copy":
            cmd.extend(["-crf", "18", "-preset", "fast"])
        cmd.extend(["-c:a", audio_codec, "-b:a", audio_bitrate])
        if subtitle_file:
            cmd.extend(["-c:s", "mov_text"])

        # ── ROOT CAUSE 4 FIX: set stream dispositions explicitly ─────────────
        # Mark the dubbed audio track as the default so every player picks it.
        # Without this, players that can't parse language metadata fall back to
        # stream index order and may find a lingering original audio stream.
        for i, audio in enumerate(audio_files):
            cmd.extend([
                f"-metadata:s:a:{i}", f"language={audio['language']}",
                f"-metadata:s:a:{i}", f"title={audio['title']}",
                f"-disposition:a:{i}", "default" if i == 0 else "none",
            ])

        # -movflags +faststart: write moov atom at the front of the file.
        # This also forces FFmpeg to re-write the container header cleanly,
        # removing any stale stream references from the source container.
        cmd.extend(["-movflags", "+faststart"])

        if video_dur > 0:
            cmd.extend(["-t", f"{video_dur:.6f}"])

        cmd.append(output_video)
        return cmd

    # BUG FIX 6: try copy first; fall back to libx264 on container error
    try:
        run_ffmpeg_command(_build_cmd(video_codec))
    except HTTPException as first_err:
        if video_codec == "copy":
            logger.warning(
                f"video_codec=copy failed ({first_err.detail[:120]}); "
                f"retrying with libx264 (crf=18)"
            )
            if os.path.exists(output_video):
                os.remove(output_video)
            run_ffmpeg_command(_build_cmd("libx264"))
        else:
            raise

    if not os.path.exists(output_video):
        raise HTTPException(status_code=500, detail="FFmpeg completed but output file not found")
    if os.path.getsize(output_video) == 0:
        raise HTTPException(status_code=500, detail="FFmpeg produced empty output file")


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
        # BUG FIX 9: initialise subtitle_path to None here so merge_video_with_audio
        # always receives a defined value even when no subtitle_key is provided.
        subtitle_path = None
        if request.subtitle_key and request.subtitle_bucket:
            logger.info("Downloading subtitles...")
            
            subtitle_filename = sanitize_filename(Path(request.subtitle_key).name)
            subtitle_path = str(work_dir / f"subtitle_{subtitle_filename}")
            temp_files_list.append(subtitle_path)
            
            download_from_s3(request.subtitle_bucket, request.subtitle_key, subtitle_path)
        
        # ── Step 3b: Audio mastering pipeline ────────────────────────
        if request.enable_audio_mastering:
            logger.info("Running audio mastering pipeline on all dubbed tracks...")
            from python_controllers.audio_mastering import master_dubbed_audio

            # Extract original video audio for BGM DUCKING REFERENCE ONLY.
            # ROOT CAUSE 3 FIX: this WAV is used purely as a sidechain signal
            # for the compressor/ducker — it must NEVER be mixed into the
            # dubbed output. master_dubbed_audio receives it as `original_path`
            # which is only read for RMS analysis. If your audio_mastering
            # implementation mixes or adds this signal, that is the source of
            # the original-voice bleed — it must only compute gain curves from
            # it, not concatenate or amix it with the dubbed track.
            original_audio_path = str(work_dir / "original_audio_ref.wav")
            try:
                extract_cmd = [
                    FFMPEG_BIN, "-y",
                    "-i", video_path,
                    "-vn",           # video-only input → extract audio
                    "-ac", "1",
                    "-ar", "24000",
                    "-acodec", "pcm_s16le",
                    original_audio_path
                ]
                subprocess.run(extract_cmd, check=True, capture_output=True,
                               timeout=300, text=True)
                has_original_audio = (
                    os.path.exists(original_audio_path) and
                    os.path.getsize(original_audio_path) > 0
                )
            except Exception as e:
                logger.warning(f"Could not extract original audio for BGM ducking: {e}")
                has_original_audio = False

            mastered_audio_files = []
            for i, af in enumerate(audio_files):
                mastered_path = str(work_dir / f"mastered_{i}_{Path(af['path']).name}")
                temp_files_list.append(mastered_path)

                # Build a minimal mastering config matching the request settings
                class _Cfg:
                    comp_threshold_db = request.comp_threshold_db
                    comp_ratio        = request.comp_ratio
                    comp_makeup_db    = request.comp_makeup_db
                    target_lufs       = request.target_lufs
                    target_true_peak  = request.target_true_peak
                    dialogue_peak_db  = request.dialogue_peak_db
                    bgm_duck_db       = request.bgm_duck_db
                    enable_eq              = True
                    enable_compression     = True
                    enable_bgm_ducking     = request.enable_bgm_ducking

                try:
                    meta = master_dubbed_audio(
                        dubbed_path   = af['path'],
                        original_path = original_audio_path if has_original_audio else None,
                        output_path   = mastered_path,
                        cfg           = _Cfg(),
                    )
                    logger.info(
                        f"Track {i} mastered: "
                        f"{meta['lufs_before']:.1f} → {meta['lufs_after']:.1f} LUFS, "
                        f"BGM duck: {meta['bgm_ducking']}"
                    )
                    mastered_audio_files.append({
                        'path':     mastered_path,
                        'language': af['language'],
                        'title':    af['title'],
                    })
                except Exception as e:
                    logger.error(f"Mastering failed for track {i}, using raw audio: {e}")
                    mastered_audio_files.append(af)

            audio_files = mastered_audio_files
            if has_original_audio:
                temp_files_list.append(original_audio_path)
        
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
            audio_bitrate=request.audio_bitrate,
            work_dir=work_dir,   # BUG FIX 5: enables per-track duration adjustment
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
        # BUG FIX 8: use shutil.rmtree instead of os.rmdir so the work directory
        # is always cleaned up even when temp files remain (e.g. on FFmpeg failure).
        # rmdir raises OSError if the directory is non-empty; rmtree does not.
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up work directory {work_dir}: {e}")


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