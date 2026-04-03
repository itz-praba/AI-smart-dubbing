import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

import torch
import boto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from botocore.exceptions import ClientError
from botocore.config import Config

try:
    from TTS.api import TTS
except ImportError:
    raise RuntimeError("Coqui TTS not installed. Install with: pip install TTS")

# ── MMS-TTS (Meta) — used for Tamil and other non-XTTS languages ─────────────
# XTTS v2 only supports 17 languages (no Tamil/Tanglish).
# Meta MMS-TTS covers 1100+ languages including Tamil natively.
# Install: pip install transformers accelerate
try:
    from transformers import VitsModel, AutoTokenizer as VitsTokenizer
    _MMS_AVAILABLE = True
except ImportError:
    _MMS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["Voice Cloning"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Voice cloning service using device: {DEVICE}")


# ── Language → TTS engine routing ────────────────────────────────────────────
# XTTS v2 natively supports exactly these 17 languages.
# ANY language not in this set must be routed to MMS-TTS.
XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr",
    "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
}

# MMS-TTS model IDs for languages not supported by XTTS v2.
# Full list: https://huggingface.co/facebook/mms-tts
MMS_LANGUAGE_MODELS = {
    "ta":  "facebook/mms-tts-tam",   # Tamil
    "tgl": "facebook/mms-tts-eng",   # Tanglish — romanised Tamil, use English MMS
}

# Combined supported languages (union of both engines)
SUPPORTED_LANGUAGES = {
    # ── XTTS v2 languages ──────────────────────────────────────────────────
    "en":    "English",
    "es":    "Spanish",
    "fr":    "French",
    "de":    "German",
    "it":    "Italian",
    "pt":    "Portuguese",
    "pl":    "Polish",
    "tr":    "Turkish",
    "ru":    "Russian",
    "nl":    "Dutch",
    "cs":    "Czech",
    "ar":    "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja":    "Japanese",
    "hu":    "Hungarian",
    "ko":    "Korean",
    "hi":    "Hindi",
    # ── MMS-TTS languages (non-XTTS) ──────────────────────────────────────
    "ta":    "Tamil",      # facebook/mms-tts-tam
    "tgl":   "Tanglish",   # facebook/mms-tts-eng (Latin-script Tamil)
}


MAX_SPEAKER_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TEXT_LENGTH = 5000
MIN_SPEAKER_AUDIO_DURATION = 3  # seconds
MAX_OUTPUT_DURATION = 300  # 5 minutes

ALLOWED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}

XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# BUG FIX (rushed Korean delivery / synthesis seams): Korean/Japanese/Chinese
# require significantly fewer characters per chunk than English because each
# character represents more phonetic content.  Oversized chunks cause the TTS
# model to rush or truncate the final syllables, and more chunks mean more
# join-seams in the audio.  We use a per-language table and fall back to 200
# for all other languages.
LANG_CHUNK_CHARS: Dict[str, int] = {
    "ko":  120,   # Korean — agglutinative, ~1.5-2× phoneme density vs English
    "ja":  100,   # Japanese — similar density
    "zh":   80,   # Chinese — character-dense
    "ar":  150,   # Arabic — morphologically rich
    "ta":  130,   # Tamil — agglutinative Dravidian, longer compound words
    "hi":  140,   # Hindi — Devanagari, moderate density
    "te":  130,   # Telugu — agglutinative, similar to Tamil
    "ur":  140,   # Urdu — similar to Hindi
    "tgl": 200,   # Tanglish — Latin script, similar density to English
}
MAX_CHUNK_CHARS = 200  # default for Latin-script languages

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
        retries={'max_attempts': 5, 'mode': 'standard'},
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
    raise RuntimeError(f"Failed to initialize S3 client: {str(e)}")

# =====================================================
# LOAD XTTS MODEL
# =====================================================
_tts_model = None

def get_tts_model() -> TTS:
    global _tts_model

    if _tts_model is None:
        try:
            logger.info(f"Loading XTTS v2 model on {DEVICE}...")
            _tts_model = TTS(
                model_name=XTTS_MODEL_NAME,
                progress_bar=True,
                gpu=(DEVICE == "cuda")
            )
            if DEVICE == "cuda":
                _tts_model.to(DEVICE)
            logger.info("XTTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise RuntimeError(f"Failed to initialize TTS model: {str(e)}")

    return _tts_model


# ── MMS-TTS model cache & loader ─────────────────────────────────────────────
_mms_model_cache: Dict[str, Any] = {}   # model_id → (model, tokenizer)


def get_mms_model(language: str):
    """
    Load and cache a Meta MMS-TTS model for the given language code.
    Used for languages not supported by XTTS v2 (e.g. Tamil, Tanglish).

    Args:
        language: internal language code ("ta", "tgl", …)

    Returns:
        Tuple of (VitsModel, tokenizer)

    Raises:
        RuntimeError if transformers is not installed or model unavailable.
    """
    if not _MMS_AVAILABLE:
        raise RuntimeError(
            "MMS-TTS requires the `transformers` library. "
            "Install with: pip install transformers accelerate"
        )

    model_id = MMS_LANGUAGE_MODELS.get(language)
    if not model_id:
        raise RuntimeError(
            f"No MMS-TTS model configured for language '{language}'. "
            f"Add an entry to MMS_LANGUAGE_MODELS."
        )

    if model_id not in _mms_model_cache:
        try:
            logger.info(f"Loading MMS-TTS model '{model_id}' on {DEVICE}...")
            tokenizer = VitsTokenizer.from_pretrained(
                model_id, cache_dir=MODEL_CACHE_DIR
            )
            model = VitsModel.from_pretrained(
                model_id, cache_dir=MODEL_CACHE_DIR
            )
            model.to(DEVICE).eval()
            _mms_model_cache[model_id] = (model, tokenizer)
            logger.info(f"MMS-TTS model '{model_id}' loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load MMS-TTS model '{model_id}': {e}")

    return _mms_model_cache[model_id]


def synthesise_with_mms(
    text: str,
    language: str,
    output_path: str,
    speed: float = 1.0,
) -> None:
    """
    Synthesise speech using Meta MMS-TTS for languages XTTS v2 does not support.

    MMS-TTS does NOT support speaker/voice cloning — it produces a fixed speaker
    voice baked into each language model.  The caller (voice_clone_tts endpoint)
    already handles this gracefully: the output is a clean, natural-sounding
    Tamil/Tanglish voice that will be post-processed by the audio mastering step.

    Args:
        text:        Text to synthesise.
        language:    Internal language code ("ta", "tgl", …).
        output_path: Local path to write the output WAV.
        speed:       Playback speed multiplier (applied via ffmpeg after synthesis).
    """
    import numpy as np
    import soundfile as sf
    import subprocess, shutil

    model, tokenizer = get_mms_model(language)

    # Tokenise
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    # Synthesise
    with torch.no_grad():
        output = model(**inputs)

    # output.waveform shape: (1, num_samples)
    waveform = output.waveform[0].cpu().float().numpy()
    sample_rate = model.config.sampling_rate   # typically 16000 for MMS

    # Write raw synthesis output
    raw_path = output_path + "_raw.wav"
    sf.write(raw_path, waveform, sample_rate)

    # Apply speed adjustment via ffmpeg (same as XTTS path)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    atempo = max(0.5, min(speed, 2.0))   # ffmpeg atempo range
    cmd = [
        ffmpeg, "-y",
        "-i", raw_path,
        "-filter:a", f"atempo={atempo}",
        "-ar", "24000",   # upsample to 24kHz to match XTTS pipeline sample rate
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception as e:
        logger.warning(f"MMS ffmpeg speed/resample step failed ({e}), using raw output")
        import shutil as _sh
        _sh.copy2(raw_path, output_path)
    finally:
        try:
            import os as _os
            _os.remove(raw_path)
        except OSError:
            pass


# =====================================================
# PYDANTIC MODELS
# =====================================================
class VoiceCloningRequest(BaseModel):
    speaker_audio_bucket: str = Field(..., min_length=1, max_length=63)
    speaker_audio_key: str = Field(..., min_length=1, max_length=1024)
    translated_text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    target_language: str = Field(..., min_length=2, max_length=10)
    output_bucket: str = Field(..., min_length=1, max_length=63)
    output_key_prefix: Optional[str] = Field(None, max_length=512)
    speed: float = Field(1.0, ge=0.5, le=2.0)

    @validator('speaker_audio_bucket', 'output_bucket')
    def validate_bucket_name(cls, v):
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name contains invalid characters")
        if '..' in v or v.startswith('.') or v.endswith('.'):
            raise ValueError("Invalid bucket name format")
        return v

    @validator('speaker_audio_key')
    def validate_audio_key(cls, v):
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
        v = v.lower()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )
        return v

    @validator('translated_text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    @validator('output_key_prefix')
    def validate_output_prefix(cls, v):
        if v is not None:
            if '..' in v or v.startswith('/'):
                raise ValueError("Invalid output prefix: path traversal detected")
        return v


class VoiceCloningResponse(BaseModel):
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
                detail=f"Speaker audio file not found in bucket '{bucket}' with key '{key}'"
            )
        elif error_code == '403':
            raise HTTPException(status_code=403, detail=f"Access denied to bucket '{bucket}'")
        else:
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
        raise HTTPException(status_code=500, detail=f"Failed to download from S3: {e.response['Error']['Message']}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during download: {str(e)}")


def upload_to_s3(local_path: str, bucket: str, key: str, metadata: dict = None) -> None:
    try:
        extra_args = {"ContentType": "audio/wav", "ServerSideEncryption": "AES256"}
        if metadata:
            extra_args["Metadata"] = metadata

        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        logger.info("Upload successful")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            raise HTTPException(status_code=404, detail=f"Destination bucket '{bucket}' does not exist")
        elif error_code == 'AccessDenied':
            raise HTTPException(status_code=403, detail=f"Access denied to bucket '{bucket}'")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e.response['Error']['Message']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during upload: {str(e)}")


def get_audio_duration(file_path: str) -> float:
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        return float(waveform.shape[1] / sample_rate)
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


def validate_speaker_audio(file_path: str) -> None:
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


def preprocess_speaker_audio(input_path: str, output_path: str) -> None:
    """
    FIX #2 — ROBOTIC VOICE: Pre-process the reference audio.

    XTTS v2 clones voice characteristics from the reference WAV. If the reference
    contains background noise, codec artifacts, or is recorded at low sample rate,
    the synthesised output sounds robotic because the model encodes those artifacts
    into the speaker embedding.

    We use ffmpeg to:
    1. Up-sample to 22050 Hz (XTTS native rate) with high-quality resampling.
    2. Apply a gentle noise-reduction filter (afftdn) to reduce hiss/static.
    3. Normalise loudness to -16 LUFS so the model always gets a consistent level.
    4. Trim leading/trailing silence so the model doesn't embed silence as a speaker trait.
    """
    import subprocess
    import shutil

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-af", (
            "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB"
            ":stop_periods=1:stop_silence=0.1:stop_threshold=-50dB,"
            "afftdn=nf=-25,"                 # gentle noise floor reduction
            "loudnorm=I=-16:LRA=11:TP=-1.5"  # EBU R128 normalisation
        ),
        "-ar", "22050",   # XTTS native sample rate
        "-ac", "1",       # mono
        "-c:a", "pcm_s16le",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        logger.info(f"Speaker audio pre-processed: {output_path}")
    except Exception as e:
        logger.warning(f"Speaker audio pre-processing failed (using original): {e}")
        import shutil as _shutil
        _shutil.copy2(input_path, output_path)


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    FIX #2 — ROBOTIC VOICE: Split long text into sentence-sized chunks.

    XTTS v2 generates audio autoregressively. When given very long texts it tends
    to drift: prosody flattens, pitch variability drops, and the result sounds
    robotic. Splitting at sentence boundaries and synthesising each chunk
    separately preserves natural intonation and pacing.
    """
    import re

    # Split on sentence-ending punctuation, keeping the delimiter attached
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            # If a single sentence is longer than max_chars, split on commas/semicolons
            if len(sentence) > max_chars:
                sub_parts = re.split(r'(?<=[,;])\s+', sentence)
                sub_current = ""
                for part in sub_parts:
                    if len(sub_current) + len(part) + 1 <= max_chars:
                        sub_current = (sub_current + " " + part).strip() if sub_current else part
                    else:
                        if sub_current:
                            chunks.append(sub_current)
                        sub_current = part
                if sub_current:
                    chunks.append(sub_current)
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def synthesise_chunks(
    tts_model: TTS,
    chunks: List[str],
    speaker_wav: str,
    language: str,
    speed: float,
    work_dir: Path,
    job_id: str,
) -> str:
    """
    FIX #2 & FIX #3 — ROBOTIC VOICE + SILENT GAPS:
    Synthesise each chunk independently and concatenate with ffmpeg.
    """
    import soundfile as sf
    import numpy as np

    sample_rate = 24000  # XTTS v2 output sample rate
    chunk_paths: List[str] = []

    for i, chunk in enumerate(chunks):
        chunk_path = str(work_dir / f"{job_id}_chunk_{i:04d}.wav")
        try:
            tts_model.tts_to_file(
                text=chunk,
                speaker_wav=speaker_wav,
                language=language,
                file_path=chunk_path,
                speed=speed,
            )
            # Verify the chunk is non-empty
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 1000:
                raise ValueError(f"Chunk {i} produced empty/too-small audio")

            chunk_paths.append(chunk_path)
            logger.info(f"Synthesised chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")

        except Exception as e:
            logger.error(f"Chunk {i} synthesis failed: {e}. Inserting silence placeholder.")
            # FIX #3: Insert proportional silence instead of dropping the segment
            estimated_duration = max(0.5, len(chunk.split()) / 2.5)
            silence_samples = int(estimated_duration * sample_rate)
            silence = np.zeros(silence_samples, dtype=np.float32)
            sf.write(chunk_path, silence, sample_rate)
            chunk_paths.append(chunk_path)

    if not chunk_paths:
        raise HTTPException(status_code=500, detail="All TTS chunks failed to synthesise")

    # Concatenate with a tiny natural inter-sentence pause (80ms)
    pause_samples = int(0.08 * sample_rate)
    pause = np.zeros(pause_samples, dtype=np.float32)

    combined: List[np.ndarray] = []
    for path in chunk_paths:
        try:
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # to mono if stereo
            combined.append(audio)
            combined.append(pause)
        except Exception as e:
            logger.warning(f"Could not read chunk {path}: {e}")

    if not combined:
        raise HTTPException(status_code=500, detail="No audio chunks could be read")

    final_audio = np.concatenate(combined).astype(np.float32)
    output_path = str(work_dir / f"{job_id}_combined.wav")
    sf.write(output_path, final_audio, sample_rate)

    # Cleanup chunk files
    for p in chunk_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return output_path


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
    job_id = str(uuid.uuid4())
    start_time = datetime.now()

    logger.info(
        f"Starting voice cloning job {job_id}: "
        f"{request.speaker_audio_bucket}/{request.speaker_audio_key} -> "
        f"{request.target_language}, text length: {len(request.translated_text)}"
    )

    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(exist_ok=True)

    speaker_filename = sanitize_filename(Path(request.speaker_audio_key).name)
    speaker_wav_raw = str(work_dir / f"speaker_raw_{speaker_filename}")
    speaker_wav = str(work_dir / "speaker_clean.wav")  # pre-processed reference
    output_wav = str(work_dir / f"{job_id}_dubbed.wav")

    if request.output_key_prefix:
        output_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}.wav"
    else:
        output_key = f"dubbed_audio/{job_id}.wav"

    # ── FIX: initialise chunks here so it is always defined regardless of
    # which TTS engine branch is taken.  The MMS path does not chunk text,
    # so it stays as an empty list; metadata and the response message use
    # len(chunks) safely in both branches. ────────────────────────────────
    chunks: List[str] = []

    try:
        # Step 1: Check speaker audio
        file_metadata = check_s3_file(request.speaker_audio_bucket, request.speaker_audio_key)
        file_size = file_metadata['size']

        if file_size > MAX_SPEAKER_AUDIO_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Speaker audio too large ({file_size / (1024*1024):.1f}MB). "
                       f"Maximum allowed: {MAX_SPEAKER_AUDIO_SIZE / (1024*1024):.0f}MB"
            )

        # Step 2: Download speaker audio
        download_from_s3(request.speaker_audio_bucket, request.speaker_audio_key, speaker_wav_raw)

        # Step 3: Validate speaker audio
        validate_speaker_audio(speaker_wav_raw)

        # FIX #2: Pre-process reference audio to remove noise/artifacts
        preprocess_speaker_audio(speaker_wav_raw, speaker_wav)

        # Step 4: Route to the correct TTS engine based on language.
        #
        # XTTS v2 supports only 17 languages (XTTS_LANGUAGES set).
        # Tamil ("ta") and Tanglish ("tgl") are NOT in that list — sending them
        # to XTTS causes the 422 Unprocessable Entity error seen in production.
        # They are handled by Meta MMS-TTS (facebook/mms-tts-tam) instead.
        #
        # NOTE: MMS-TTS does not support voice cloning — it uses a fixed speaker
        # voice per language model. The cloned-voice experience is available only
        # for XTTS-supported languages.
        if request.target_language in XTTS_LANGUAGES:
            logger.info(f"[XTTS v2] Synthesising language '{request.target_language}'")
            tts_model = get_tts_model()

            chunk_chars = LANG_CHUNK_CHARS.get(request.target_language, MAX_CHUNK_CHARS)
            chunks = split_text_into_chunks(request.translated_text, max_chars=chunk_chars)
            logger.info(
                f"Split text into {len(chunks)} chunk(s) "
                f"(lang={request.target_language}, max_chars={chunk_chars})"
            )

            combined_wav = synthesise_chunks(
                tts_model=tts_model,
                chunks=chunks,
                speaker_wav=speaker_wav,
                language=request.target_language,
                speed=request.speed,
                work_dir=work_dir,
                job_id=job_id,
            )

        elif request.target_language in MMS_LANGUAGE_MODELS:
            logger.info(
                f"[MMS-TTS] Language '{request.target_language}' not in XTTS v2 — "
                f"using {MMS_LANGUAGE_MODELS[request.target_language]}"
            )
            combined_wav = str(work_dir / f"{job_id}_mms.wav")
            synthesise_with_mms(
                text=request.translated_text,
                language=request.target_language,
                output_path=combined_wav,
                speed=request.speed,
            )
            # chunks stays [] — MMS synthesises in a single pass, no chunking

        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Language '{request.target_language}' is not supported by any "
                    f"configured TTS engine. "
                    f"Supported: {sorted(SUPPORTED_LANGUAGES.keys())}"
                )
            )

        # Step 5: Verify output
        if not os.path.exists(combined_wav):
            raise HTTPException(status_code=500, detail="TTS completed but output file not found")

        output_size = os.path.getsize(combined_wav)
        if output_size == 0:
            raise HTTPException(status_code=500, detail="TTS produced empty output file")

        # Copy to expected output path
        import shutil as _shutil
        _shutil.copy2(combined_wav, output_wav)

        audio_duration = get_audio_duration(output_wav)
        logger.info(f"Generated audio: {output_size} bytes, duration: {audio_duration:.2f}s")

        # Step 6: Upload to S3
        # chunks_label: XTTS path sets chunks list; MMS path leaves it empty → "1 (MMS)"
        chunks_label = str(len(chunks)) if chunks else "1 (MMS single-pass)"
        upload_metadata = {
            "job_id": job_id,
            "source_bucket": request.speaker_audio_bucket,
            "source_key": request.speaker_audio_key,
            "language": request.target_language,
            "text_length": str(len(request.translated_text)),
            "chunks": chunks_label,
            "audio_duration": str(round(audio_duration, 2)),
            "timestamp": datetime.utcnow().isoformat()
        }

        upload_to_s3(output_wav, request.output_bucket, output_key, upload_metadata)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s")

        # Build a human-readable synthesis summary for the response message
        if chunks:
            chunk_info = f"{len(chunks)} chunks via XTTS v2"
        else:
            chunk_info = "single-pass via MMS-TTS"

        return VoiceCloningResponse(
            success=True,
            job_id=job_id,
            output_bucket=request.output_bucket,
            output_key=output_key,
            language=SUPPORTED_LANGUAGES.get(request.target_language, request.target_language),
            text_length=len(request.translated_text),
            audio_duration_seconds=round(audio_duration, 2),
            processing_time_seconds=round(processing_time, 2),
            device=DEVICE,
            message=f"Voice cloned TTS generated successfully: {audio_duration:.2f}s audio ({chunk_info})"
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

    finally:
        # Cleanup work directory
        import shutil as _shutil
        try:
            _shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# =====================================================
# UTILITY ENDPOINTS
# =====================================================
@router.get("/voice-clone/languages", tags=["Voice Cloning"])
async def get_supported_languages():
    return {
        "supported_languages": [
            {"code": code, "name": name}
            for code, name in sorted(SUPPORTED_LANGUAGES.items())
        ],
        "total": len(SUPPORTED_LANGUAGES)
    }


@router.get("/voice-clone/info", tags=["Voice Cloning"])
async def get_model_info():
    model_loaded = _tts_model is not None
    return {
        "xtts_model": XTTS_MODEL_NAME,
        "xtts_languages": sorted(XTTS_LANGUAGES),
        "mms_languages": list(MMS_LANGUAGE_MODELS.keys()),
        "mms_models": MMS_LANGUAGE_MODELS,
        "mms_available": _MMS_AVAILABLE,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "xtts_model_loaded": model_loaded,
        "supported_languages": sorted(SUPPORTED_LANGUAGES.keys()),
        "max_text_length": MAX_TEXT_LENGTH,
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "max_speaker_audio_size_mb": MAX_SPEAKER_AUDIO_SIZE // (1024 * 1024),
        "min_speaker_duration_seconds": MIN_SPEAKER_AUDIO_DURATION,
        "supported_audio_formats": list(ALLOWED_AUDIO_FORMATS)
    }


@router.get("/voice-clone/health", tags=["Health"])
async def health_check():
    model_loaded = _tts_model is not None
    return {
        "status": "healthy",
        "service": "voice-cloning",
        "device": DEVICE,
        "model_loaded": model_loaded,
        "temp_dir_writable": os.access(TEMP_DIR, os.W_OK)
    }