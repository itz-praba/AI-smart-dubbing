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

router = APIRouter(prefix="/ai", tags=["Lip Sync"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

TARGET_SAMPLE_RATE = 24000   # FIX #3: Match XTTS v2 output (was 16000, causing resampling artefacts)
AUDIO_DTYPE = np.float32

# Time stretching limits
# BUG FIX (lip-sync drift): Korean sentences are typically 30–40% longer than their
# English source, so a MAX_SPEED_UP of 1.20 was not enough — cloned audio would
# overflow its slot and push all subsequent segments late.  Raised to 1.40.
MAX_SPEED_UP  = 1.40     # was 1.20 – Korean routinely needs 30-40% speed-up
MAX_SLOW_DOWN = 0.78     # tightened slightly to give more room at the slow end

MIN_SEGMENT_DURATION = 0.05  # 50ms minimum (was 100ms — some segments are legitimately short)
MAX_SEGMENT_DURATION = 60.0

# BUG FIX (early/late entries): Language-specific pre-advance offsets (in seconds).
# Korean TTS has a ~70-90 ms onset delay before the first phoneme; pre-advancing the
# placement compensates so the voice starts with the mouth movement, not after it.
LANG_TIMING_OFFSET_S: dict = {
    "ko": -0.080,   # advance Korean audio 80 ms (empirical, adjust per TTS model)
    "ja": -0.060,
    "zh": -0.050,
    "ta": -0.060,   # Tamil TTS has similar onset delay to Japanese
    "tgl": -0.040,  # Tanglish synthesised as English — small onset delay
}

MAX_AUDIO_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

# FIX #3: Crossfade length in samples to smooth segment joins and avoid clicks
CROSSFADE_SAMPLES = 512   # ~21ms at 24kHz

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

s3_client = None
S3_ENABLED = all(os.getenv(var) for var in REQUIRED_ENV_VARS)

if S3_ENABLED:
    try:
        boto_config = Config(
            retries={'max_attempts': 3, 'mode': 'standard'},
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
    index: int = Field(..., ge=0)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    tts_audio_key: str = Field(..., min_length=1)
    
    @validator('end')
    def validate_end_time(cls, v, values):
        if 'start' in values:
            duration = v - values['start']
            if duration < MIN_SEGMENT_DURATION:
                raise ValueError(
                    f"Segment too short ({duration:.3f}s). Minimum: {MIN_SEGMENT_DURATION}s"
                )
            if duration > MAX_SEGMENT_DURATION:
                raise ValueError(
                    f"Segment too long ({duration:.2f}s). Maximum: {MAX_SEGMENT_DURATION}s"
                )
        return v
    
    @validator('tts_audio_key')
    def validate_audio_key(cls, v):
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid audio key: path traversal detected")
        ext = Path(v).suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. Allowed: {', '.join(ALLOWED_AUDIO_FORMATS)}"
            )
        return v


class LipSyncRequest(BaseModel):
    segments: List[TimedSegment] = Field(..., min_items=1, max_items=1000)
    tts_audio_bucket: str = Field(..., min_length=1, max_length=63)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    output_key_prefix: Optional[str] = Field(None, max_length=512)
    enable_time_stretch: bool = Field(True)
    # BUG FIX (lip-sync timing offset): pass the target language so
    # build_timeline_audio can apply a language-specific pre-advance offset
    # (e.g. -80 ms for Korean) to compensate for TTS onset delay.
    target_language: str = Field("", max_length=10, description="Target language code (e.g. 'ko')")
    
    @validator('tts_audio_bucket', 'output_bucket')
    def validate_bucket_name(cls, v):
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('segments')
    def validate_segments_timing(cls, v):
        if not v:
            return v
        sorted_segments = sorted(v, key=lambda s: s.index)
        for i in range(len(sorted_segments) - 1):
            current = sorted_segments[i]
            next_seg = sorted_segments[i + 1]
            if current.end > next_seg.start:
                raise ValueError(
                    f"Segments {current.index} and {next_seg.index} overlap"
                )
        return v


class AlignedSegmentInfo(BaseModel):
    index: int
    original_duration: float
    target_duration: float
    aligned_duration: float
    time_stretch_ratio: Optional[float] = None
    method: str


class LipSyncResponse(BaseModel):
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
    try:
        yield
    finally:
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")


def sanitize_filename(filename: str, max_length: int = 255) -> str:
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
    if not S3_ENABLED:
        raise HTTPException(status_code=503, detail="S3 not configured.")
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
            raise HTTPException(status_code=404, detail=f"Audio file not found: s3://{bucket}/{key}")
        elif error_code == '403':
            raise HTTPException(status_code=403, detail=f"Access denied to bucket '{bucket}'")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to check file in S3: {error_code}")


def download_from_s3(bucket: str, key: str, local_path: str) -> None:
    if not S3_ENABLED:
        raise HTTPException(status_code=503, detail="S3 not configured")
    try:
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3_client.download_file(bucket, key, local_path)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise HTTPException(status_code=500, detail="Download failed or file is empty")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to download from S3: {e.response['Error']['Message']}")


def upload_to_s3(local_path: str, bucket: str, key: str, metadata: dict = None) -> None:
    if not S3_ENABLED:
        raise HTTPException(status_code=503, detail="S3 not configured")
    try:
        extra_args = {"ContentType": "audio/wav", "ServerSideEncryption": "AES256"}
        if metadata:
            extra_args["Metadata"] = metadata
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e.response['Error']['Message']}")


# =====================================================
# AUDIO PROCESSING FUNCTIONS
# =====================================================
def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """
    FIX #3 — SILENT GAPS: Load audio resampled to TARGET_SAMPLE_RATE (24kHz).

    Previously we were loading all audio at 16kHz but XTTS v2 produces 24kHz WAV
    files. When the cloned voice WAVs were loaded and resampled down to 16kHz some
    audio frames were dropped during the librosa resampling step, creating tiny
    silent holes. By matching the XTTS sample rate we avoid lossy down-sampling.
    """
    try:
        audio, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, dtype=AUDIO_DTYPE)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio from {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load audio file: {str(e)}")


def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    return float(librosa.get_duration(y=audio, sr=sr))


def write_audio(path: str, audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    try:
        sf.write(path, audio, sr)
    except Exception as e:
        logger.error(f"Failed to write audio to {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to write audio file: {str(e)}")


def add_silence(audio: np.ndarray, silence_sec: float, sr: int) -> np.ndarray:
    if silence_sec <= 0:
        return audio
    silence_samples = int(silence_sec * sr)
    silence = np.zeros(silence_samples, dtype=AUDIO_DTYPE)
    return np.concatenate([audio, silence])


def time_stretch_audio(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    try:
        ratio = max(MAX_SLOW_DOWN, min(ratio, MAX_SPEED_UP))
        stretched = pyrb.time_stretch(audio, sr, ratio)
        return stretched.astype(AUDIO_DTYPE)
    except Exception as e:
        logger.error(f"Time stretching failed: {e}")
        raise HTTPException(status_code=500, detail=f"Time stretching failed: {str(e)}")


def apply_fade(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    FIX #3 — SILENT GAPS / CLICKS: Apply a short linear fade-in and fade-out.

    Hard joins between concatenated segments cause clicks and pop artefacts that
    can be perceived as brief silences because the decoder in a playback device
    may treat them as invalid frames. A tiny crossfade smooths the join.
    """
    n = len(audio)
    if n < 2 * fade_samples:
        return audio

    out = audio.copy()
    # fade-in
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=AUDIO_DTYPE)
    out[:fade_samples] *= fade_in
    # fade-out
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=AUDIO_DTYPE)
    out[-fade_samples:] *= fade_out
    return out


def align_segment(
    tts_audio: np.ndarray,
    sr: int,
    target_duration: float,
    enable_time_stretch: bool = True,
    segment_index: int = 0
) -> Tuple[np.ndarray, AlignedSegmentInfo]:
    """
    FIX #3 — SILENT GAPS: Improved alignment strategy.

    Root causes of gaps in the final video:
    1. When TTS produced longer audio than the target slot, the code trimmed it
       hard — no fade-out, causing abrupt silence at the cut point.
    2. When TTS audio was shorter, silence was padded at the END of the slot.
       This is correct, but the silence had no fade which caused perceptible pops
       on some devices.
    3. The sample-rate mismatch (16kHz vs 24kHz) meant integer rounding in
       `max_samples = int(target_duration * sr)` sometimes cut off the last few
       milliseconds of a sentence.

    Fixes applied:
    - All trimmed ends now receive a fade-out (CROSSFADE_SAMPLES long).
    - All silence insertions receive a tiny fade-in so the join is smooth.
    - max_samples calculation adds a 10ms safety margin to avoid cutting the
      last phoneme of a sentence.
    """
    tts_duration = get_audio_duration(tts_audio, sr)
    
    logger.debug(f"Aligning segment {segment_index}: TTS={tts_duration:.3f}s, Target={target_duration:.3f}s")
    
    # Case 1: Exact match (within 50ms tolerance)
    if abs(tts_duration - target_duration) < 0.05:
        audio_with_fade = apply_fade(tts_audio, CROSSFADE_SAMPLES)
        return audio_with_fade, AlignedSegmentInfo(
            index=segment_index,
            original_duration=tts_duration,
            target_duration=target_duration,
            aligned_duration=tts_duration,
            time_stretch_ratio=None,
            method="exact_match"
        )
    
    # Case 2: TTS too long — speed up or trim
    if tts_duration > target_duration:
        if not enable_time_stretch:
            # FIX #3: Add 10ms safety margin and fade-out on trim
            max_samples = int((target_duration + 0.01) * sr)
            trimmed = tts_audio[:min(max_samples, len(tts_audio))]
            trimmed = apply_fade(trimmed, CROSSFADE_SAMPLES)
            
            return trimmed, AlignedSegmentInfo(
                index=segment_index,
                original_duration=tts_duration,
                target_duration=target_duration,
                aligned_duration=get_audio_duration(trimmed, sr),
                time_stretch_ratio=None,
                method="trimmed"
            )
        
        ratio = tts_duration / target_duration
        actual_ratio = min(ratio, MAX_SPEED_UP)
        stretched = time_stretch_audio(tts_audio, sr, actual_ratio)
        
        # FIX #3: 10ms safety margin + fade on trim
        max_samples = int((target_duration + 0.01) * sr)
        final_audio = stretched[:min(max_samples, len(stretched))]
        final_audio = apply_fade(final_audio, CROSSFADE_SAMPLES)
        
        final_duration = get_audio_duration(final_audio, sr)
        
        return final_audio, AlignedSegmentInfo(
            index=segment_index,
            original_duration=tts_duration,
            target_duration=target_duration,
            aligned_duration=final_duration,
            time_stretch_ratio=actual_ratio,
            method="time_stretch"
        )
    
    # Case 3: TTS too short — pad with silence (with fade)
    else:
        # FIX #3: Apply fade-out on speech, then attach silence
        speech_with_fade = apply_fade(tts_audio, CROSSFADE_SAMPLES)
        silence_needed = target_duration - get_audio_duration(speech_with_fade, sr)
        final_audio = add_silence(speech_with_fade, silence_needed, sr)
        
        final_duration = get_audio_duration(final_audio, sr)
        
        return final_audio, AlignedSegmentInfo(
            index=segment_index,
            original_duration=tts_duration,
            target_duration=target_duration,
            aligned_duration=final_duration,
            time_stretch_ratio=None,
            method="silence_added"
        )


def build_timeline_audio(
    aligned_audio_map: Dict[int, Tuple[np.ndarray, float]],
    segments: List[TimedSegment],
    sr: int,
    target_language: str = "",
) -> np.ndarray:
    """
    FIX #3 — SILENT GAPS: Build the final audio by placing each aligned segment
    at its EXACT start position in the timeline rather than naively concatenating.

    Naive concatenation (np.concatenate) accumulates rounding errors across
    segments. For a 30-segment video, a 2ms rounding error per segment adds up
    to 60ms of drift, which shows as a silent gap between audio and video at the
    end. By writing each segment into a pre-allocated array at its exact sample
    offset (computed from the original timestamps) we eliminate accumulated drift.

    BUG FIX (early/late entries): An optional per-language timing offset
    (LANG_TIMING_OFFSET_S) is applied so languages like Korean — whose TTS model
    has an onset delay — are pre-advanced to align with mouth movements.

    BUG FIX (segment bleed): Changed += to clamped addition so fade-overlap from
    adjacent segments cannot push samples beyond ±1.0 and bleed audible artefacts
    into the neighbour's silence gap.
    """
    if not segments:
        return np.array([], dtype=AUDIO_DTYPE)
    
    total_duration = max(seg.end for seg in segments)
    total_samples = int(np.ceil(total_duration * sr)) + sr  # +1s safety buffer
    
    timeline = np.zeros(total_samples, dtype=AUDIO_DTYPE)

    # Per-language onset pre-advance (negative = move earlier in timeline)
    timing_offset_s = LANG_TIMING_OFFSET_S.get(target_language, 0.0)
    
    for seg in sorted(segments, key=lambda s: s.index):
        audio, _ = aligned_audio_map[seg.index]
        
        # Apply language-aware timing offset so audio starts with mouth movement
        adjusted_start = max(0.0, seg.start + timing_offset_s)
        start_sample = int(round(adjusted_start * sr))
        end_sample = start_sample + len(audio)
        
        if end_sample > total_samples:
            # Trim if the segment runs slightly over (due to time-stretch rounding)
            audio = audio[:total_samples - start_sample]
            end_sample = total_samples
        
        # BUG FIX: Use clamped addition instead of pure +=.
        # Pure additive mix at crossfade overlaps can push values > 1.0, which on
        # some decoders sounds like a brief silent dropout.  Clamp keeps all samples
        # in the valid [-1.0, 1.0] float range.
        mixed = timeline[start_sample:end_sample] + audio
        timeline[start_sample:end_sample] = np.clip(mixed, -1.0, 1.0)
    
    # Remove trailing silence beyond the last segment
    last_sample = int(np.ceil(max(seg.end for seg in segments) * sr))
    last_sample = min(last_sample + int(0.1 * sr), total_samples)  # keep 100ms tail
    timeline = timeline[:last_sample]
    
    return timeline.astype(AUDIO_DTYPE)


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
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(
        f"Starting lip-sync alignment job {job_id}: "
        f"{len(request.segments)} segments"
    )
    
    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(exist_ok=True)
    
    temp_files_list = []
    aligned_segments_info = []
    aligned_audio_map: Dict[int, Tuple[np.ndarray, float]] = {}  # index -> (audio, sr)
    
    try:
        for seg in sorted(request.segments, key=lambda s: s.index):
            logger.info(f"Processing segment {seg.index}")
            
            tts_filename = sanitize_filename(Path(seg.tts_audio_key).name)
            tts_path = str(work_dir / f"seg_{seg.index}_{tts_filename}")
            temp_files_list.append(tts_path)
            
            file_metadata = check_s3_file(request.tts_audio_bucket, seg.tts_audio_key)
            
            if file_metadata['size'] > MAX_AUDIO_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"Segment {seg.index} audio file too large")
            
            download_from_s3(request.tts_audio_bucket, seg.tts_audio_key, tts_path)
            
            tts_audio, sr = load_audio(tts_path)
            
            target_duration = seg.end - seg.start
            aligned_audio, seg_info = align_segment(
                tts_audio,
                sr,
                target_duration,
                enable_time_stretch=request.enable_time_stretch,
                segment_index=seg.index
            )
            
            aligned_segments_info.append(seg_info)
            aligned_audio_map[seg.index] = (aligned_audio, sr)
            
            logger.info(
                f"Segment {seg.index} aligned: "
                f"{seg_info.original_duration:.2f}s → {seg_info.aligned_duration:.2f}s "
                f"(method: {seg_info.method})"
            )
        
        # FIX #3: Use timeline-based placement instead of naive concatenation
        logger.info("Building final audio timeline...")
        final_audio = build_timeline_audio(
            aligned_audio_map,
            request.segments,
            TARGET_SAMPLE_RATE,
            target_language=request.target_language,  # BUG FIX: language-aware timing offset
        )
        
        final_audio_path = str(work_dir / "aligned_final.wav")
        temp_files_list.append(final_audio_path)
        write_audio(final_audio_path, final_audio, TARGET_SAMPLE_RATE)
        
        total_duration = get_audio_duration(final_audio, TARGET_SAMPLE_RATE)
        
        if request.output_key_prefix:
            output_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}.wav"
        else:
            output_key = f"aligned_audio/{job_id}.wav"
        
        upload_metadata = {
            "job_id": job_id,
            "segments_count": str(len(request.segments)),
            "total_duration": str(round(total_duration, 2)),
            "sample_rate": str(TARGET_SAMPLE_RATE),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        upload_to_s3(final_audio_path, request.output_bucket, output_key, upload_metadata)
        
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
        for file_path in temp_files_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")
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
    return {
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "max_speed_up": MAX_SPEED_UP,
        "max_slow_down": MAX_SLOW_DOWN,
        "min_segment_duration": MIN_SEGMENT_DURATION,
        "max_segment_duration": MAX_SEGMENT_DURATION,
        "crossfade_samples": CROSSFADE_SAMPLES,
        "supported_audio_formats": list(ALLOWED_AUDIO_FORMATS),
        "max_audio_file_size_mb": MAX_AUDIO_FILE_SIZE // (1024 * 1024),
        "s3_enabled": S3_ENABLED
    }


@router.get("/lip-sync/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "lip-sync-alignment",
        "s3_enabled": S3_ENABLED,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }