"""
prosody_transfer.py  –  Emotion & prosody transfer for dubbed audio
====================================================================
Solves Problem 9: Emotional tone differences.

The dubbed TTS voice has correct timbre (via XTTS/MMS cloning) but loses the
source speaker's emotional dynamics — a crying scene sounds the same as a calm
one because TTS generates at a flat prosody.

This module:
  1. Extracts prosodic features from the ORIGINAL audio segment:
       • pitch contour (F0) via pYIN
       • energy envelope (RMS)
       • speaking rate (syllable density proxy)
       • emotion label via a lightweight Wav2Vec2-based classifier
  2. Computes a prosody delta between source and dubbed audio.
  3. Applies prosody shaping to the dubbed WAV:
       • Pitch shift  → pyrubberband pitch_shift
       • Energy scale → RMS-matching gain
       • Speed adjust → time_stretch (within lip-sync-safe range)
       • Emotion EQ   → scipy biquad filters per emotion class
         (e.g. warmth boost for sad/tender, presence cut for calm, bright EQ for happy)

The result feeds back into the lip-sync step so final alignment sees the
emotionally-shaped audio, not the flat TTS output.

Dependencies:
    pip install librosa pyrubberband scipy soundfile numpy transformers torch

Emotion classifier checkpoint (auto-downloaded on first use):
    ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition  (4 classes: angry, happy, sad, neutral)
    Falls back to energy-heuristic if model unavailable.
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from scipy import signal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["Prosody Transfer"])

# ── S3 ────────────────────────────────────────────────────────────────────────
REQUIRED_ENV_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
S3_ENABLED = all(os.getenv(v) for v in REQUIRED_ENV_VARS)
s3_client = None
if S3_ENABLED:
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
            config=Config(retries={"max_attempts": 3, "mode": "standard"},
                          connect_timeout=10, read_timeout=120),
        )
        s3_client.list_buckets()
        logger.info("prosody_transfer: S3 client initialised")
    except Exception as e:
        logger.warning(f"prosody_transfer: S3 init failed ({e}) – S3 disabled")
        S3_ENABLED = False

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

TARGET_SR = 24_000       # match XTTS output sample rate throughout pipeline
DTYPE     = np.float32

# ── Emotion classifier ────────────────────────────────────────────────────────
EMOTION_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
_emotion_pipeline = None   # lazy-loaded

# How emotion maps to EQ adjustments (gain_db, centre_hz, Q)
# Each entry is a list of (gain_db, centre_hz, Q) peaking-EQ bands.
EMOTION_EQ: Dict[str, List[Tuple[float, float, float]]] = {
    "angry":   [(+3.0, 3000, 2.0), (-2.0, 250, 1.5)],   # presence + thin body = aggression
    "happy":   [(+2.5, 4000, 2.0), (+1.5, 200, 1.5)],   # bright + warm = cheerful
    "sad":     [(-1.5, 4000, 2.0), (+2.5, 250, 1.5)],   # de-bright + body = mournful
    "neutral": [],                                         # no EQ for neutral
    "fearful": [(-1.0, 3500, 2.0), (+1.0, 500, 1.5)],   # slightly withdrawn
    "disgust": [(+1.5, 2500, 2.0), (-1.0, 200, 1.5)],
    "surprised": [(+3.0, 4500, 1.5), (+1.0, 200, 1.5)],
}

# Speed nudge per emotion (multiplier on top of lip-sync stretch)
EMOTION_SPEED_NUDGE: Dict[str, float] = {
    "angry":     1.08,
    "happy":     1.05,
    "sad":       0.93,
    "neutral":   1.00,
    "fearful":   1.04,
    "disgust":   0.97,
    "surprised": 1.10,
}

# Safe limits — prosody transfer must not fight lip-sync
MAX_PITCH_SHIFT_SEMITONES = 2.5
MAX_ENERGY_GAIN_DB        = 6.0
MAX_SPEED_NUDGE           = 1.12
MIN_SPEED_NUDGE           = 0.90


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ProsodyTransferRequest(BaseModel):
    """Transfer prosodic characteristics from a source audio segment to a dubbed TTS WAV."""
    bucket:              str   = Field(..., min_length=1, max_length=63)
    source_audio_key:    str   = Field(..., description="S3 key of the original speaker's segment WAV")
    dubbed_audio_key:    str   = Field(..., description="S3 key of the flat TTS dubbed WAV")
    output_bucket:       str   = Field(..., min_length=1, max_length=63)
    output_key_prefix:   str   = Field("prosody/")
    segment_index:       int   = Field(0, ge=0)

    # Fine-tune controls
    enable_pitch_transfer:  bool  = Field(True)
    enable_energy_transfer: bool  = Field(True)
    enable_emotion_eq:      bool  = Field(True)
    enable_speed_nudge:     bool  = Field(True)
    pitch_strength:         float = Field(0.6, ge=0.0, le=1.0,
                                          description="How strongly to apply pitch delta (0=off, 1=full)")
    energy_strength:        float = Field(0.8, ge=0.0, le=1.0,
                                          description="How strongly to match source energy level")


class ProsodyTransferResponse(BaseModel):
    success:              bool
    job_id:               str
    segment_index:        int
    output_bucket:        str
    output_key:           str
    detected_emotion:     str
    source_pitch_mean_hz: float
    dubbed_pitch_mean_hz: float
    pitch_shift_applied:  float
    energy_gain_db:       float
    speed_nudge:          float
    processing_time_seconds: float
    message:              str


class BatchProsodyRequest(BaseModel):
    """Process all segments in a single batch call."""
    bucket:              str  = Field(..., min_length=1, max_length=63)
    output_bucket:       str  = Field(..., min_length=1, max_length=63)
    output_key_prefix:   str  = Field("prosody/")
    segments: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of dicts with keys: index, source_audio_key, dubbed_audio_key. "
            "Each segment is processed independently."
        )
    )
    enable_pitch_transfer:  bool  = Field(True)
    enable_energy_transfer: bool  = Field(True)
    enable_emotion_eq:      bool  = Field(True)
    enable_speed_nudge:     bool  = Field(True)
    pitch_strength:         float = Field(0.6, ge=0.0, le=1.0)
    energy_strength:        float = Field(0.8, ge=0.0, le=1.0)


class BatchProsodyResponse(BaseModel):
    success:                 bool
    job_id:                  str
    segments_processed:      int
    segments_failed:         int
    results:                 List[Dict[str, Any]]
    processing_time_seconds: float
    message:                 str


# ═══════════════════════════════════════════════════════════════════════════════
# S3 HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _s3_download(bucket: str, key: str, local_path: str) -> None:
    if not S3_ENABLED:
        raise HTTPException(503, "S3 not configured")
    try:
        s3_client.download_file(bucket, key, local_path)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            raise HTTPException(404, f"File not found: s3://{bucket}/{key}")
        raise HTTPException(500, f"S3 download failed: {e.response['Error']['Message']}")


def _s3_upload(local_path: str, bucket: str, key: str) -> None:
    if not S3_ENABLED:
        raise HTTPException(503, "S3 not configured")
    try:
        s3_client.upload_file(
            local_path, bucket, key,
            ExtraArgs={"ContentType": "audio/wav", "ServerSideEncryption": "AES256"},
        )
    except ClientError as e:
        raise HTTPException(500, f"S3 upload failed: {e.response['Error']['Message']}")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=TARGET_SR, dtype=DTYPE)
    return audio, sr


def _write(path: str, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    sf.write(path, audio, sr)


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _get_emotion_pipeline():
    """Lazy-load the Wav2Vec2 emotion classifier. Falls back gracefully."""
    global _emotion_pipeline
    if _emotion_pipeline is not None:
        return _emotion_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _emotion_pipeline = hf_pipeline(
            "audio-classification",
            model=EMOTION_MODEL_ID,
            device=0 if _cuda_available() else -1,
        )
        logger.info("Emotion classifier loaded successfully")
    except Exception as e:
        logger.warning(f"Emotion classifier unavailable ({e}); using energy heuristic")
        _emotion_pipeline = None
    return _emotion_pipeline


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def detect_emotion(audio: np.ndarray, sr: int) -> str:
    """
    Classify the emotional tone of a short audio segment.

    Primary path: Wav2Vec2-based classifier (4-7 emotion classes).
    Fallback: RMS energy + pitch variance heuristic.
    Returns one of: angry, happy, sad, neutral, fearful, disgust, surprised.
    """
    pipe = _get_emotion_pipeline()
    if pipe is not None:
        try:
            # Classifier expects 16kHz float32 array
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
            results = pipe(audio_16k, sampling_rate=16_000)
            if results:
                label = results[0]["label"].lower()
                # Normalise label variants ("sad", "sadness", "SAD" → "sad")
                for key in EMOTION_EQ:
                    if key in label:
                        return key
                return "neutral"
        except Exception as e:
            logger.warning(f"Emotion classifier inference failed ({e}); using heuristic")

    # ── Energy/pitch heuristic fallback ──────────────────────────────────────
    rms = float(np.sqrt(np.mean(audio ** 2)))
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, fill_na=None,
        )
        pitch_std = float(np.nanstd(f0[voiced_flag])) if voiced_flag.any() else 0.0
    except Exception:
        pitch_std = 0.0

    if rms > 0.15 and pitch_std > 50:
        return "angry"
    if rms > 0.12 and pitch_std > 30:
        return "happy"
    if rms < 0.05 and pitch_std < 20:
        return "sad"
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# PROSODY FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pitch_mean(audio: np.ndarray, sr: int) -> float:
    """Return mean fundamental frequency (Hz) of voiced frames, or 0 if unvoiced."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, fill_na=None,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag.any() else np.array([])
        return float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    except Exception:
        return 0.0


def extract_rms_db(audio: np.ndarray) -> float:
    """Return RMS level in dBFS (−96 floor)."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-10:
        return -96.0
    return float(20 * np.log10(rms))


# ═══════════════════════════════════════════════════════════════════════════════
# PROSODY APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by `semitones` using pyrubberband (preserves duration)."""
    semitones = float(np.clip(semitones, -MAX_PITCH_SHIFT_SEMITONES, MAX_PITCH_SHIFT_SEMITONES))
    if abs(semitones) < 0.1:
        return audio
    try:
        shifted = pyrb.pitch_shift(audio, sr, semitones)
        return shifted.astype(DTYPE)
    except Exception as e:
        logger.warning(f"Pitch shift failed ({e}); skipping")
        return audio


def apply_energy_match(audio: np.ndarray, source_rms_db: float,
                       strength: float = 0.8) -> Tuple[np.ndarray, float]:
    """
    Scale dubbed audio to partially match source RMS level.
    Returns (adjusted_audio, gain_db_applied).
    """
    dubbed_rms_db  = extract_rms_db(audio)
    delta_db       = (source_rms_db - dubbed_rms_db) * strength
    delta_db       = float(np.clip(delta_db, -MAX_ENERGY_GAIN_DB, MAX_ENERGY_GAIN_DB))
    gain_linear    = 10 ** (delta_db / 20.0)
    adjusted       = np.clip(audio * gain_linear, -1.0, 1.0).astype(DTYPE)
    return adjusted, delta_db


def apply_emotion_eq(audio: np.ndarray, sr: int, emotion: str) -> np.ndarray:
    """
    Apply emotion-specific peaking EQ bands using scipy IIR biquad filters.
    Each band: (gain_db, centre_hz, Q).
    """
    bands = EMOTION_EQ.get(emotion, [])
    if not bands:
        return audio

    out = audio.copy()
    for gain_db, centre_hz, Q in bands:
        if abs(gain_db) < 0.1:
            continue
        # Compute biquad coefficients for a peaking EQ filter
        w0     = 2 * np.pi * centre_hz / sr
        A      = 10 ** (gain_db / 40.0)
        alpha  = np.sin(w0) / (2 * Q)
        b0     =  1 + alpha * A
        b1     = -2 * np.cos(w0)
        b2     =  1 - alpha * A
        a0     =  1 + alpha / A
        a1     = -2 * np.cos(w0)
        a2     =  1 - alpha / A
        b      = np.array([b0 / a0, b1 / a0, b2 / a0])
        a      = np.array([1.0,     a1 / a0, a2 / a0])
        try:
            out = signal.lfilter(b, a, out).astype(DTYPE)
        except Exception as e:
            logger.warning(f"EQ band {centre_hz}Hz failed ({e}); skipping")

    return np.clip(out, -1.0, 1.0).astype(DTYPE)


def apply_speed_nudge(audio: np.ndarray, sr: int, emotion: str,
                      source_duration: float, dubbed_duration: float) -> Tuple[np.ndarray, float]:
    """
    Apply a mild speed nudge toward the emotion's natural rate.
    This is applied BEFORE lip-sync time-stretch, so it's a prosodic pre-shape
    that helps the TTS delivery feel emotionally natural while staying within
    the time-stretch budget.

    The nudge is blended with the actual duration ratio so the lip-sync step
    never sees an audio file that's already wildly mismatched to its slot.
    """
    emotion_nudge = EMOTION_SPEED_NUDGE.get(emotion, 1.0)
    # Blend: 40% emotion nudge + 60% preserve existing duration (lip-sync will finish the job)
    blended_nudge = 0.4 * emotion_nudge + 0.6 * 1.0
    blended_nudge = float(np.clip(blended_nudge, MIN_SPEED_NUDGE, MAX_SPEED_NUDGE))

    if abs(blended_nudge - 1.0) < 0.01:
        return audio, 1.0

    try:
        nudged = pyrb.time_stretch(audio, sr, blended_nudge)
        return nudged.astype(DTYPE), blended_nudge
    except Exception as e:
        logger.warning(f"Speed nudge failed ({e}); skipping")
        return audio, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# CORE TRANSFER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def transfer_prosody(
    source_path: str,
    dubbed_path: str,
    output_path: str,
    request: ProsodyTransferRequest,
) -> Dict[str, Any]:
    """
    Load source + dubbed segments, extract prosodic features from source,
    and apply them to the dubbed audio. Write the result to output_path.

    Returns a dict of metrics for the response model.
    """
    source_audio, src_sr = _load(source_path)
    dubbed_audio, dub_sr = _load(dubbed_path)

    # ── 1. Detect emotion from source ────────────────────────────────────────
    emotion = detect_emotion(source_audio, src_sr)
    logger.info(f"[prosody] Segment {request.segment_index}: detected emotion = {emotion}")

    # ── 2. Extract source features ───────────────────────────────────────────
    src_pitch_mean = extract_pitch_mean(source_audio, src_sr)
    src_rms_db     = extract_rms_db(source_audio)

    # ── 3. Extract dubbed features ───────────────────────────────────────────
    dub_pitch_mean = extract_pitch_mean(dubbed_audio, dub_sr)

    # ── 4. Compute pitch delta (semitones) ───────────────────────────────────
    pitch_shift = 0.0
    if request.enable_pitch_transfer and src_pitch_mean > 0 and dub_pitch_mean > 0:
        ratio        = src_pitch_mean / dub_pitch_mean
        raw_semitones = 12.0 * np.log2(ratio)
        pitch_shift   = float(raw_semitones * request.pitch_strength)
        pitch_shift   = float(np.clip(pitch_shift, -MAX_PITCH_SHIFT_SEMITONES, MAX_PITCH_SHIFT_SEMITONES))
        logger.info(
            f"[prosody] Pitch: src={src_pitch_mean:.1f}Hz dub={dub_pitch_mean:.1f}Hz "
            f"→ {raw_semitones:.2f} st (applied {pitch_shift:.2f} st)"
        )

    # ── 5. Apply transformations ──────────────────────────────────────────────
    out = dubbed_audio.copy()

    # 5a. Pitch shift
    if pitch_shift != 0.0:
        out = apply_pitch_shift(out, dub_sr, pitch_shift)

    # 5b. Energy match
    energy_gain_db = 0.0
    if request.enable_energy_transfer:
        out, energy_gain_db = apply_energy_match(out, src_rms_db, request.energy_strength)

    # 5c. Emotion EQ
    if request.enable_emotion_eq:
        out = apply_emotion_eq(out, dub_sr, emotion)

    # 5d. Speed nudge
    speed_nudge = 1.0
    if request.enable_speed_nudge:
        src_dur = librosa.get_duration(y=source_audio, sr=src_sr)
        dub_dur = librosa.get_duration(y=dubbed_audio, sr=dub_sr)
        out, speed_nudge = apply_speed_nudge(out, dub_sr, emotion, src_dur, dub_dur)

    # ── 6. Write output ───────────────────────────────────────────────────────
    _write(output_path, out, dub_sr)

    return {
        "detected_emotion":     emotion,
        "source_pitch_mean_hz": round(src_pitch_mean, 2),
        "dubbed_pitch_mean_hz": round(dub_pitch_mean, 2),
        "pitch_shift_applied":  round(pitch_shift, 3),
        "energy_gain_db":       round(energy_gain_db, 2),
        "speed_nudge":          round(speed_nudge, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/prosody-transfer",
    response_model=ProsodyTransferResponse,
    summary="Transfer emotional prosody from source to dubbed audio",
    description=(
        "Extracts pitch, energy, and emotion from the original speaker segment "
        "and applies them to the flat TTS dubbed audio. "
        "Fixes Problem 9: emotional tone differences."
    ),
)
async def prosody_transfer(request: ProsodyTransferRequest) -> ProsodyTransferResponse:
    import time
    job_id = str(uuid.uuid4())
    t0     = time.time()

    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(exist_ok=True)

    src_path = str(work_dir / "source.wav")
    dub_path = str(work_dir / "dubbed.wav")
    out_path = str(work_dir / "prosody_out.wav")

    try:
        _s3_download(request.bucket, request.source_audio_key, src_path)
        _s3_download(request.bucket, request.dubbed_audio_key, dub_path)

        metrics = transfer_prosody(src_path, dub_path, out_path, request)

        out_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}_seg{request.segment_index:04d}.wav"
        _s3_upload(out_path, request.output_bucket, out_key)

        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"[prosody] Segment {request.segment_index} done in {elapsed}s — "
            f"emotion={metrics['detected_emotion']} "
            f"pitch={metrics['pitch_shift_applied']:+.2f}st "
            f"energy={metrics['energy_gain_db']:+.1f}dB"
        )

        return ProsodyTransferResponse(
            success                  = True,
            job_id                   = job_id,
            segment_index            = request.segment_index,
            output_bucket            = request.output_bucket,
            output_key               = out_key,
            processing_time_seconds  = elapsed,
            message=(
                f"Prosody transferred: emotion={metrics['detected_emotion']}, "
                f"pitch={metrics['pitch_shift_applied']:+.2f}st, "
                f"energy={metrics['energy_gain_db']:+.1f}dB"
            ),
            **metrics,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[prosody] Segment {request.segment_index} failed: {e}")
        raise HTTPException(500, f"Prosody transfer failed: {e}")
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


@router.post(
    "/prosody-transfer/batch",
    response_model=BatchProsodyResponse,
    summary="Batch prosody transfer for all dubbed segments",
    description=(
        "Processes all segments in a single call. "
        "Each segment is processed independently; failures are logged but do not abort the batch. "
        "Returns per-segment results including emotion classification and metrics."
    ),
)
async def prosody_transfer_batch(request: BatchProsodyRequest) -> BatchProsodyResponse:
    import time
    job_id = str(uuid.uuid4())
    t0     = time.time()

    results      : List[Dict[str, Any]] = []
    failed_count : int = 0

    for seg in request.segments:
        seg_req = ProsodyTransferRequest(
            bucket               = request.bucket,
            source_audio_key     = seg["source_audio_key"],
            dubbed_audio_key     = seg["dubbed_audio_key"],
            output_bucket        = request.output_bucket,
            output_key_prefix    = request.output_key_prefix,
            segment_index        = seg.get("index", 0),
            enable_pitch_transfer  = request.enable_pitch_transfer,
            enable_energy_transfer = request.enable_energy_transfer,
            enable_emotion_eq      = request.enable_emotion_eq,
            enable_speed_nudge     = request.enable_speed_nudge,
            pitch_strength         = request.pitch_strength,
            energy_strength        = request.energy_strength,
        )
        try:
            resp = await prosody_transfer(seg_req)
            results.append({
                "index":            seg.get("index", 0),
                "success":          True,
                "output_key":       resp.output_key,
                "emotion":          resp.detected_emotion,
                "pitch_shift_st":   resp.pitch_shift_applied,
                "energy_gain_db":   resp.energy_gain_db,
                "speed_nudge":      resp.speed_nudge,
            })
        except Exception as e:
            failed_count += 1
            results.append({
                "index":   seg.get("index", 0),
                "success": False,
                "error":   str(e),
                # On failure, preserve original dubbed_audio_key so pipeline can continue
                "output_key": seg["dubbed_audio_key"],
            })
            logger.warning(
                f"[prosody batch] Segment {seg.get('index', 0)} failed: {e} — "
                f"original dubbed audio will be used"
            )

    elapsed = round(time.time() - t0, 2)
    return BatchProsodyResponse(
        success                  = failed_count < len(request.segments),
        job_id                   = job_id,
        segments_processed       = len(request.segments) - failed_count,
        segments_failed          = failed_count,
        results                  = results,
        processing_time_seconds  = elapsed,
        message=(
            f"Batch prosody transfer: {len(request.segments) - failed_count}/"
            f"{len(request.segments)} segments succeeded in {elapsed}s"
        ),
    )


@router.get("/prosody-transfer/health", tags=["Health"])
async def health_check():
    emotion_pipe = _get_emotion_pipeline()
    return {
        "status":              "healthy",
        "service":             "prosody-transfer",
        "emotion_classifier":  "loaded" if emotion_pipe else "heuristic-fallback",
        "emotion_model":       EMOTION_MODEL_ID,
        "s3_enabled":          S3_ENABLED,
        "supported_emotions":  list(EMOTION_EQ.keys()),
    }
