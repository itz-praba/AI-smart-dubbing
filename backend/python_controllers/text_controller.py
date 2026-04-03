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
    'el', 'he', 'th',
    'ta',   # Tamil
    'te',   # Telugu   — Whisper large-v3 supports natively
    'ur',   # Urdu     — Whisper large-v3 supports natively
    'tgl',  # Tanglish (romanised Tamil; Whisper transcribes as English phonetics)
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

# BUG FIX 1: warn instead of crash — lets server start without S3 in dev/test
_missing_env = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
for _v in _missing_env:
    logger.warning(f"Missing environment variable: {_v} – S3 features will be disabled")

# HuggingFace token for speaker diarization
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.warning("HuggingFace token not found - speaker diarization will be disabled")

# =====================================================
# AWS S3 CLIENT
# =====================================================
# BUG FIX 2: graceful init — never crash at module import time
S3_ENABLED = not bool(_missing_env)
s3_client   = None

if S3_ENABLED:
    try:
        from botocore.config import Config
        
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
        logger.warning(f"S3 client init failed ({e}) – S3 features disabled")
        S3_ENABLED = False

# =====================================================
# DEVICE CONFIGURATION
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
logger.info(f"Using device: {device}, compute type: {compute_type}")

# =====================================================
# MODEL MANAGEMENT (Lazy Loading)
# =====================================================
_whisper_model       = None
_whisper_model_size  = None   # FIX P4: track which size is loaded
_diarization_pipeline = None

def get_whisper_model(model_size: str = "large-v3") -> WhisperModel:
    global _whisper_model, _whisper_model_size

    # FIX P4: if a different model size is requested, discard the cached model
    # and load the correct one. Previously every call after the first returned
    # the cached model regardless of the requested size.
    if _whisper_model is not None and _whisper_model_size != model_size:
        logger.info(
            f"Model size changed ({_whisper_model_size} → {model_size}); "
            f"reloading Whisper model"
        )
        _whisper_model      = None
        _whisper_model_size = None

    if _whisper_model is None:
        try:
            logger.info(f"Loading Whisper model: {model_size} on {device}")
            _whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root="./models",
                num_workers=4 if device == "cuda" else 2
            )
            _whisper_model_size = model_size
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}")

    return _whisper_model


def get_diarization_pipeline() -> Optional[Pipeline]:
    global _diarization_pipeline
    
    if not HUGGINGFACE_TOKEN:
        logger.warning("Diarization disabled: HuggingFace token not found")
        return None
    
    if _diarization_pipeline is None:
        try:
            logger.info("Loading speaker diarization pipeline")
            # BUG FIX 6: pyannote ≥3.x uses token= not use_auth_token= (deprecated)
            try:
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HUGGINGFACE_TOKEN
                )
            except TypeError:
                # Fallback for pyannote <3.x installs that still use the old kwarg
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HUGGINGFACE_TOKEN
                )
            
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
    speaker: str
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class SpeechToTextRequest(BaseModel):
    audio_bucket: str = Field(..., min_length=1, max_length=63)
    audio_key: str = Field(..., min_length=1, max_length=1024)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    # BUG FIX 7: expose output_key_prefix so the pipeline can control
    # where the transcript JSON is stored (previously hardcoded to "transcripts/")
    output_key_prefix: Optional[str] = Field(
        "transcripts/",
        max_length=512,
        description="S3 prefix for output transcript. Default: 'transcripts/'"
    )
    language: Optional[str] = Field(None, description="ISO language code (e.g., 'en', 'es')")
    diarize: bool = Field(True, description="Perform speaker diarization")
    model_size: str = Field("large-v3", description="Whisper model size")
    word_timestamps: bool = Field(True, description="Include word-level timestamps")

    # VAD controls — exposed so callers can override when audio is being truncated
    vad_filter: bool = Field(
        False,
        description=(
            "Enable Silero VAD pre-filtering. "
            "DEFAULT IS NOW FALSE because VAD was silently dropping the last 2–3s "
            "of speech in short dubbing clips (energy fade-out at sentence end scored "
            "below threshold). "
            "Set to True only for long recordings with significant silence sections "
            "(podcasts, lectures, interviews). "
            "For dubbing source clips, always leave False."
        )
    )
    vad_threshold: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Silero VAD speech probability threshold (only used when vad_filter=True). "
            "0.15 = maximum recall — only pure silence is filtered. "
            "Raise toward 0.5 only for very noisy recordings."
        )
    )
    
    @validator('output_key_prefix')
    def validate_output_prefix(cls, v):
        if v is not None and ('..' in v or v.startswith('/')):
            raise ValueError("Invalid output prefix: path traversal detected")
        return v

    @validator('audio_bucket', 'output_bucket')
    def validate_bucket_name(cls, v):
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v
    
    @validator('audio_key')
    def validate_audio_key(cls, v):
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid audio key: path traversal detected")
        ext = Path(v).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Allowed formats: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            )
        return v
    
    @validator('language')
    def validate_language(cls, v):
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            )
        return v
    
    @validator('model_size')
    def validate_model_size(cls, v):
        # BUG FIX 4 & 5: accept both friendly keys ("large") and canonical values
        # ("large-v3"). Normalise to the canonical model name here so
        # get_whisper_model() always receives a value WhisperModel accepts.
        if v in WHISPER_MODELS:
            return WHISPER_MODELS[v]   # "large" → "large-v3", "base" → "base" etc.
        if v in WHISPER_MODELS.values():
            return v                   # already canonical ("large-v3")
        raise ValueError(
            f"Invalid model size '{v}'. "
            f"Available: {', '.join(list(WHISPER_MODELS.keys()) + list(WHISPER_MODELS.values()))}"
        )


class SpeechToTextResponse(BaseModel):
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
            raise HTTPException(status_code=403, detail=f"Access denied to bucket '{bucket}'")
        else:
            logger.error(f"S3 head_object error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to check file in S3: {error_code}")


def download_from_s3(bucket: str, key: str, local_path: str) -> None:
    try:
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        
        if not os.path.exists(local_path):
            raise HTTPException(status_code=500, detail="File download completed but file not found locally")
        
        file_size = os.path.getsize(local_path)
        if file_size == 0:
            raise HTTPException(status_code=500, detail="Downloaded file is empty")
        
        logger.info(f"Successfully downloaded {file_size} bytes")
        
    except ClientError as e:
        logger.error(f"S3 download error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download from S3: {e.response['Error']['Message']}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected download error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during download: {str(e)}")


def upload_to_s3(local_path: str, bucket: str, key: str, metadata: dict = None) -> None:
    try:
        extra_args = {
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256"
        }
        if metadata:
            extra_args["Metadata"] = metadata
        
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        logger.info("Upload successful")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 upload error: {e}")
        if error_code == 'NoSuchBucket':
            raise HTTPException(status_code=404, detail=f"Destination bucket '{bucket}' does not exist")
        elif error_code == 'AccessDenied':
            raise HTTPException(status_code=403, detail=f"Access denied to bucket '{bucket}'")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e.response['Error']['Message']}")
    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during upload: {str(e)}")


def perform_diarization(audio_path: str) -> tuple:
    """
    Run pyannote speaker diarization.
    BUG FIX 10: returns (segments, success_bool) instead of silently swallowing
    failures. The endpoint logs a warning and continues without diarization when
    success=False, but the fact is surfaced in the response metadata.
    """
    speaker_segments = []

    try:
        diarization_pipeline = get_diarization_pipeline()

        if diarization_pipeline is None:
            logger.warning("Diarization pipeline not available")
            return speaker_segments, False

        logger.info("Starting speaker diarization")
        diarization = diarization_pipeline(audio_path)

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker": speaker,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2)
            })

        logger.info(
            f"Diarization complete: {len(speaker_segments)} segments, "
            f"{len(set(s['speaker'] for s in speaker_segments))} unique speakers"
        )
        return speaker_segments, True

    except Exception as e:
        logger.error(f"Diarization failed: {e}", exc_info=True)
        return speaker_segments, False


# BUG FIX 8: language-specific initial prompts so Whisper is grounded in the
# right script/vocabulary from the first segment rather than defaulting to English.
_INITIAL_PROMPTS: Dict[str, str] = {
    "hi": "यह एक स्पष्ट हिंदी भाषण रिकॉर्डिंग है।",
    "te": "ఇది స్పష్టమైన తెలుగు మాట్లాడే రికార్డింగ్.",
    "ur": "یہ ایک واضح اردو تقریر کی ریکارڈنگ ہے۔",
    "ta": "இது ஒரு தெளிவான தமிழ் பேச்சு பதிவு.",
    "ar": "هذا تسجيل صوتي واضح باللغة العربية.",
    "zh": "这是一段清晰的普通话录音。",
    "ja": "これは明瞭な日本語の音声録音です。",
    "ko": "이것은 명확한 한국어 음성 녹음입니다.",
    "ru": "Это чёткая запись речи на русском языке.",
}


def _get_audio_duration(audio_path: str) -> float:
    """
    Return the actual duration of an audio file in seconds using librosa.
    Used to detect and log VAD truncation gaps (FIX P7).
    Falls back to 0.0 if the file cannot be read.
    """
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return float(duration)
    except Exception:
        try:
            # Fallback: soundfile
            import soundfile as sf
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except Exception:
            return 0.0


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    model_size: str = "large-v3",
    word_timestamps: bool = True,
    vad_filter: bool = False,    # P3 FIX: default OFF — prevents end-of-clip truncation
    vad_threshold: float = 0.15, # P3 FIX: lower threshold when VAD is used
) -> tuple:
    """
    Transcribe audio using Faster-Whisper with VAD settings tuned for
    maximum recall — ensuring no speech is silently dropped.

    ── Why audio was being truncated (root causes) ──────────────────────────

    FIX P1 — VAD threshold too high (was 0.35, now 0.20):
        Silero VAD scores each 30ms chunk with a speech probability [0,1].
        A threshold of 0.35 — even though lower than the default 0.5 — is
        still high enough to drop:
          • speech at the end of a sentence (energy naturally falls off)
          • accented/non-English phonemes with lower activation scores
          • soft speakers and whispered segments
          • the last 2–4 seconds of audio (energy fade-out before silence)
        Lowered to 0.20. At this threshold, only pure silence and clear
        background noise are filtered; all marginal speech passes through.

    FIX P2 — min_silence_duration_ms too short (was 200ms, now 2000ms):
        At 200ms, Silero treats every inter-word pause as a silence boundary
        and splits the audio into dozens of tiny chunks. Each chunk is then
        independently threshold-tested: chunks that begin or end on a quiet
        phoneme fall below threshold and are dropped entirely.
        At 2000ms, only genuine long pauses create boundaries. Normal
        conversational pauses (200–500ms) remain inside speech chunks where
        they are correctly transcribed.

    FIX P3 — speech_pad_ms too small (was 400ms, now 500ms each side):
        Padding is added around each speech chunk detected by VAD. Increasing
        to 500ms ensures that the consonant/vowel at the very start and end of
        each spoken phrase — where energy is lowest — is captured within the
        padded boundary rather than being clipped at the edge.

    FIX P5 — log_prob_threshold too strict (was -1.0, now disabled with -3.0):
        When the average log-probability of a decoded segment is below this
        threshold, Whisper marks the segment as a decoding failure and
        silently discards it. This hits hardest on:
          • short segments (< 1 second) — too few tokens to average reliably
          • the last segment of the file — often lower confidence
          • quiet or accented speech
        Set to -3.0 (effectively disabled) so Whisper always returns its
        best-effort transcription rather than dropping the segment.

    FIX P6 — no_speech_threshold too high (was 0.45, now 0.3):
        The no_speech token probability is Whisper's internal confidence that
        a chunk contains NO speech. At 0.45, even faint speech triggers the
        no_speech filter. 0.3 means only chunks where Whisper is very
        confident there is no speech (>70% confidence) are suppressed.

    FIX P7 — No duration logging (now measures actual vs transcribed gap):
        Added pre-transcription audio duration measurement. After transcription,
        logs a warning if the transcribed duration is more than 1 second
        shorter than the actual audio duration, making truncation visible
        in logs rather than silently producing a shorter-than-expected transcript.
    """
    try:
        whisper_model = get_whisper_model(model_size)

        # Tanglish: route as "en" so Whisper uses Latin-script phoneme decoding
        whisper_language = "en" if language == "tgl" else language

        # Language-specific seed prompt; fall back to generic English
        initial_prompt = _INITIAL_PROMPTS.get(
            whisper_language or "",
            "The following is a clear spoken audio recording:",
        )

        # FIX P7: measure actual audio duration before transcription
        actual_duration = _get_audio_duration(audio_path)
        logger.info(
            f"Starting transcription | model={model_size} | "
            f"lang={whisper_language or 'auto-detect'} | "
            f"actual_duration={actual_duration:.2f}s | "
            f"vad_filter={vad_filter} | vad_threshold={vad_threshold}"
        )

        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            word_timestamps=word_timestamps,
            language=whisper_language,

            # ── VAD CONFIGURATION (FIX P1, P2, P3) ─────────────────────────
            vad_filter=vad_filter,
            vad_parameters=dict(
                threshold=vad_threshold,         # FIX P1: caller-controlled, default 0.20
                min_silence_duration_ms=2000,    # FIX P2: only genuine long pauses split audio
                speech_pad_ms=500,               # FIX P3: 500ms padding each side
            ) if vad_filter else {},

            # ── DECODING THRESHOLDS (FIX P5, P6) ────────────────────────────
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-3.0,             # FIX P5: disabled — never drop segments
            no_speech_threshold=0.3,             # FIX P6: only suppress very confident silence
            initial_prompt=initial_prompt,
        )

        # Convert generator to list
        segments = list(segments)

        # FIX P7: log warning if transcription is significantly shorter than audio
        if segments:
            transcribed_end = segments[-1].end
            gap = actual_duration - transcribed_end
            if actual_duration > 0 and gap > 1.0:
                logger.warning(
                    f"TRUNCATION DETECTED: actual={actual_duration:.2f}s  "
                    f"transcribed_end={transcribed_end:.2f}s  "
                    f"gap={gap:.2f}s — {gap:.1f}s of audio was not transcribed. "
                    f"Consider setting vad_filter=False if this persists."
                )
            else:
                logger.info(
                    f"Transcription complete: {len(segments)} segments | "
                    f"transcribed={transcribed_end:.2f}s / actual={actual_duration:.2f}s | "
                    f"lang={info.language} (prob={info.language_probability:.2f})"
                )
        else:
            logger.warning(
                f"Transcription produced 0 segments for {actual_duration:.2f}s of audio. "
                f"The audio may be silent, or VAD filtered all content. "
                f"Check audio quality and language setting."
            )

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
    transcript = []
    
    for seg in transcription_segments:
        # Skip empty/whitespace-only segments that slipped through
        text = seg.text.strip()
        if not text:
            logger.debug(f"Skipping empty segment [{seg.start:.2f}-{seg.end:.2f}]")
            continue

        speaker = "SPEAKER_UNKNOWN"
        
        if speaker_segments:
            max_overlap = 0
            for s in speaker_segments:
                overlap_start = max(seg.start, s["start"])
                overlap_end = min(seg.end, s["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = s["speaker"]
        
        confidence = None
        if hasattr(seg, 'words') and seg.words:
            confidences = [w.probability for w in seg.words if hasattr(w, 'probability')]
            if confidences:
                confidence = sum(confidences) / len(confidences)
        
        transcript.append({
            "speaker": speaker,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
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
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    logger.info(f"Starting job {job_id}: {request.audio_bucket}/{request.audio_key}")
    
    audio_filename = sanitize_filename(Path(request.audio_key).name)
    audio_path = str(TEMP_DIR / f"{job_id}_{audio_filename}")
    output_path = str(TEMP_DIR / f"{job_id}.json")
    
    try:
        with temp_files(audio_path, output_path):
            file_metadata = check_s3_file(request.audio_bucket, request.audio_key)
            file_size = file_metadata['size']
            
            if file_size > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file too large ({file_size / (1024*1024):.1f}MB). "
                           f"Maximum allowed: {MAX_AUDIO_SIZE / (1024*1024):.0f}MB"
                )
            
            logger.info(f"File size: {file_size / (1024*1024):.2f}MB")
            
            download_from_s3(request.audio_bucket, request.audio_key, audio_path)
            
            # BUG FIX 10: perform_diarization now returns (segments, success)
            speaker_segments = []
            diarization_failed = False
            if request.diarize and HUGGINGFACE_TOKEN:
                speaker_segments, diar_ok = perform_diarization(audio_path)
                if not diar_ok:
                    diarization_failed = True
                    logger.warning(
                        "Diarization failed — continuing with SPEAKER_UNKNOWN labels. "
                        "Check pyannote model access and HF token."
                    )
            elif request.diarize and not HUGGINGFACE_TOKEN:
                logger.warning("Diarization requested but HuggingFace token not available")
            
            transcription_segments, info = transcribe_audio(
                audio_path,
                language=request.language,
                model_size=request.model_size,
                word_timestamps=request.word_timestamps,
                vad_filter=request.vad_filter,
                vad_threshold=request.vad_threshold,
            )
            
            transcript = align_speakers_with_transcription(
                transcription_segments,
                speaker_segments
            )
            
            speakers_detected = len(set(s["speaker"] for s in transcript))
            # BUG FIX 9: guard against empty transcript (silent/music-only audio)
            duration = transcript[-1]["end"] if transcript else 0.0
            processing_time = (datetime.now() - start_time).total_seconds()
            
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
                    "timestamp": datetime.utcnow().isoformat(),
                    "diarization_failed": diarization_failed,   # BUG FIX 10
                }
            )
            
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(result.dict(), f, indent=2, ensure_ascii=False)
            
            # BUG FIX 7: respect output_key_prefix from request
            prefix = (request.output_key_prefix or "transcripts/").rstrip("/")
            output_key = f"{prefix}/{job_id}.json"
            
            upload_metadata = {
                "job_id": job_id,
                "source_bucket": request.audio_bucket,
                "source_key": request.audio_key,
                "language": info.language,
                "segments_count": str(len(transcript)),
                "speakers_detected": str(speakers_detected)
            }
            
            upload_to_s3(output_path, request.output_bucket, output_key, upload_metadata)
            
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


@router.get("/speech/health", tags=["Health"])
async def health_check():
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


@router.get("/speech/models", tags=["Speech"])
async def get_model_info():
    return {
        "whisper_models": list(WHISPER_MODELS.keys()),
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_audio_formats": list(ALLOWED_AUDIO_EXTENSIONS),
        "max_file_size_mb": MAX_AUDIO_SIZE // (1024 * 1024),
        "max_duration_seconds": MAX_AUDIO_DURATION,
        "device": device,
        "diarization_available": HUGGINGFACE_TOKEN is not None
    }