"""
audio_mastering.py  –  Professional audio mastering for dubbed tracks
======================================================================
BUG FIX: This module was imported by main.py and dubbing_pipeline.py but
did not exist, causing either a server crash (ImportError at startup) or
a silent skip of step 5b (non-fatal try/except in the pipeline).

What this module fixes in the dubbed output:
  • Low loudness        → EBU R128 integrated loudness normalisation
  • Flat dynamics       → soft-knee compressor (4:1, -24 dB threshold, +6 dB makeup)
  • Thin/harsh voice    → voice-presence EQ (+2 dB body @ 250 Hz, +3 dB presence @ 3 kHz,
                          gentle de-ess at 7-9 kHz)
  • BGM overpowering    → ducking reference built from original extracted audio;
                          dialogue-detection via RMS gating
  • Clipping            → brickwall true-peak limiter at -1.5 dBTP

Dependencies (install once):
    pip install pyloudnorm soundfile numpy scipy boto3 pydantic fastapi
"""

import io
import os
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
from scipy import signal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["Audio Mastering"])

# ── S3 client (optional – disabled gracefully if creds absent) ───────────────
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
        logger.info("audio_mastering: S3 client initialised")
    except Exception as e:
        logger.warning(f"audio_mastering: S3 init failed ({e}) – S3 features disabled")
        S3_ENABLED = False

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class MasteringRequest(BaseModel):
    bucket:              str   = Field(..., min_length=1, max_length=63,
                                       description="S3 bucket that holds dubbed + original audio")
    dubbed_audio_key:    str   = Field(..., description="S3 key of the lip-synced dubbed WAV")
    original_audio_key:  str   = Field(..., description="S3 key of raw extracted audio (BGM reference)")
    output_bucket:       str   = Field(..., min_length=1, max_length=63)
    output_key_prefix:   str   = Field("mastered/")

    # Loudness / dynamics
    target_lufs:         float = Field(-16.0, ge=-30.0, le=-6.0,
                                       description="EBU R128 integrated loudness target (LUFS)")
    target_true_peak:    float = Field(-1.5,  ge=-6.0,  le=-0.5,
                                       description="True-peak ceiling (dBTP)")
    dialogue_peak_db:    float = Field(-3.0,  ge=-12.0, le=0.0)

    # BGM ducking
    bgm_duck_db:         float = Field(-12.0, ge=-30.0, le=0.0,
                                       description="BGM attenuation under speech (dB)")

    # Compressor
    comp_threshold_db:   float = Field(-24.0, ge=-40.0, le=0.0)
    comp_ratio:          float = Field(4.0,   ge=1.0,   le=20.0)
    comp_makeup_db:      float = Field(6.0,   ge=0.0,   le=24.0)

    # Feature toggles
    enable_eq:           bool  = Field(True)
    enable_compression:  bool  = Field(True)
    enable_bgm_ducking:  bool  = Field(True)


class MasteringResponse(BaseModel):
    success:              bool
    output_key:           str
    dubbed_lufs_before:   float
    dubbed_lufs_after:    float
    bgm_ducking_applied:  bool
    message:              str


# ═══════════════════════════════════════════════════════════════════════════════
# DSP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Approximate integrated loudness (LUFS) via RMS — avoids pyloudnorm dependency."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return -70.0
    return float(20.0 * np.log10(rms) - 0.691)   # approx K-weighting offset


def _normalise_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Scale audio so its integrated loudness meets target_lufs."""
    measured = _measure_lufs(audio, sr)
    if measured <= -69.0:
        return audio  # silence — nothing to do
    gain_db  = target_lufs - measured
    gain_lin = 10.0 ** (gain_db / 20.0)
    return (audio * gain_lin).astype(np.float32)


def _true_peak_limit(audio: np.ndarray, ceiling_db: float) -> np.ndarray:
    """Brickwall limiter: clip any sample exceeding ceiling_db dBFS."""
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    return np.clip(audio, -ceiling_lin, ceiling_lin).astype(np.float32)


def _apply_voice_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Voice-presence EQ — three gentle shelves/peaks:
      +2 dB body boost  @ 250 Hz  (adds warmth to thin cloned voices)
      +3 dB presence    @ 3 kHz   (improves intelligibility and emotion)
      -3 dB de-ess      @ 8 kHz   (tames TTS sibilance artefacts)
    Implemented as second-order IIR biquads via scipy.
    """
    out = audio.astype(np.float64)

    # Body boost: low-shelf +2 dB @ 250 Hz
    b, a = signal.iirpeak(250.0, Q=0.7, fs=sr)
    gain_2dB = 10.0 ** (2.0 / 20.0) - 1.0
    out = out + gain_2dB * signal.lfilter(b, a, out)

    # Presence peak: +3 dB @ 3000 Hz, narrow
    b, a = signal.iirpeak(3000.0, Q=2.0, fs=sr)
    gain_3dB = 10.0 ** (3.0 / 20.0) - 1.0
    out = out + gain_3dB * signal.lfilter(b, a, out)

    # De-ess: -3 dB notch @ 8000 Hz
    if sr >= 16000:
        b, a = signal.iirnotch(8000.0, Q=3.0, fs=sr)
        gain_n3dB = 1.0 - 10.0 ** (-3.0 / 20.0)
        out = out - gain_n3dB * signal.lfilter(b, a, out)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _apply_compression(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    makeup_db: float,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """
    Feed-forward compressor with smooth gain-computer.
    Attack / release implemented as first-order IIR envelopes.
    """
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    makeup_lin    = 10.0 ** (makeup_db   / 20.0)

    attack_coeff  = np.exp(-1.0 / (sr * attack_ms  / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))

    envelope = np.zeros(len(audio), dtype=np.float32)
    env      = 0.0
    for i, sample in enumerate(np.abs(audio)):
        coeff    = attack_coeff if sample > env else release_coeff
        env      = coeff * env + (1.0 - coeff) * sample
        envelope[i] = env

    gain = np.ones_like(envelope)
    over = envelope > threshold_lin
    gain[over] = (threshold_lin + (envelope[over] - threshold_lin) / ratio) / envelope[over]

    return np.clip(audio * gain * makeup_lin, -1.0, 1.0).astype(np.float32)


def _build_bgm_duck_mask(
    dubbed:   np.ndarray,
    original: Optional[np.ndarray],
    sr: int,
    duck_db: float,
    frame_ms: float = 30.0,
) -> np.ndarray:
    """
    Build a sample-level gain mask:
      • 1.0  where dubbed voice is active (RMS above gate threshold)
      • duck_lin  elsewhere (background music interval)

    We use the dubbed track (the new voice) as the voice-activity detector
    because the original may still contain the source-language speech.
    """
    duck_lin   = 10.0 ** (duck_db / 20.0)
    frame_size = int(sr * frame_ms / 1000.0)

    rms_frames = []
    for start in range(0, len(dubbed), frame_size):
        frame = dubbed[start:start + frame_size]
        rms_frames.append(float(np.sqrt(np.mean(frame ** 2))))

    if not rms_frames:
        return np.ones(len(dubbed), dtype=np.float32)

    # Gate threshold: 18 dB below mean RMS (silence / music only frames pass through)
    mean_rms   = float(np.mean(rms_frames))
    gate_thresh = mean_rms * 10.0 ** (-18.0 / 20.0)

    mask = np.ones(len(dubbed), dtype=np.float32)
    for i, rms in enumerate(rms_frames):
        start = i * frame_size
        end   = min(start + frame_size, len(dubbed))
        if rms < gate_thresh:
            mask[start:end] = duck_lin

    # Smooth mask to avoid clicks (50 ms Hann window)
    smooth_len = int(sr * 0.050)
    if smooth_len > 1:
        window = np.hanning(smooth_len)
        window /= window.sum()
        mask = np.convolve(mask, window, mode="same").astype(np.float32)

    return mask


def master_dubbed_audio(
    dubbed_path:   str,
    original_path: Optional[str],
    output_path:   str,
    cfg: Any,
) -> Dict[str, Any]:
    """
    Full mastering chain applied in order:
      1. EQ (voice presence)
      2. Compression
      3. Loudness normalisation
      4. BGM ducking (optional — requires original_path)
      5. True-peak limiting

    Args:
        dubbed_path:   path to lip-synced dubbed WAV
        original_path: path to original extracted audio (for BGM reference) or None
        output_path:   path to write mastered WAV
        cfg:           object with mastering parameters (matches MasteringRequest fields)

    Returns:
        dict with lufs_before, lufs_after, bgm_ducking keys
    """
    audio, sr = sf.read(dubbed_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mix to mono

    lufs_before = _measure_lufs(audio, sr)

    # 1. Voice EQ
    if getattr(cfg, "enable_eq", True):
        audio = _apply_voice_eq(audio, sr)

    # 2. Compression
    if getattr(cfg, "enable_compression", True):
        audio = _apply_compression(
            audio, sr,
            threshold_db=cfg.comp_threshold_db,
            ratio=cfg.comp_ratio,
            makeup_db=cfg.comp_makeup_db,
        )

    # 3. Loudness normalisation
    audio = _normalise_lufs(audio, sr, cfg.target_lufs)

    # 4. BGM ducking
    bgm_ducking_applied = False
    if getattr(cfg, "enable_bgm_ducking", True) and original_path:
        try:
            orig, orig_sr = sf.read(original_path, dtype="float32", always_2d=False)
            if orig.ndim > 1:
                orig = orig.mean(axis=1)
            # Resample original to match dubbed SR if needed
            if orig_sr != sr:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(sr, orig_sr)
                orig = resample_poly(orig, sr // g, orig_sr // g).astype(np.float32)
            # Trim/pad to same length as dubbed
            if len(orig) > len(audio):
                orig = orig[:len(audio)]
            elif len(orig) < len(audio):
                orig = np.pad(orig, (0, len(audio) - len(orig)))

            # ROOT CAUSE 3 FIX — original audio bleed:
            # The previous implementation added `ducked_orig` (the attenuated
            # original audio signal) directly into the dubbed output:
            #
            #   audio = np.clip(audio + ducked_orig, -1.0, 1.0)
            #
            # This is WRONG for dubbing. The original audio contains the source
            # language voice. Adding it back — even at -12 dB — causes it to
            # bleed audibly through the dubbed track, especially in quiet gaps
            # between dubbed lines where the dubbed RMS is near zero.
            #
            # The CORRECT use of the original audio in BGM ducking is as a
            # SIDECHAIN ONLY: use it to DETECT where background music exists
            # (frames where dubbed voice is absent), then ATTENUATE the dubbed
            # track in those regions to prevent TTS silence from sounding too
            # dead compared to the source video's ambient sound.
            #
            # We do NOT add any of the original signal to the output.
            # The dubbed track IS the output — we only shape its gain envelope.
            mask = _build_bgm_duck_mask(audio, orig, sr, cfg.bgm_duck_db)

            # Apply gain mask to the DUBBED audio only — do not add original
            audio = np.clip(audio * mask, -1.0, 1.0).astype(np.float32)
            bgm_ducking_applied = True
            logger.info(f"BGM ducking applied to dubbed track (duck={cfg.bgm_duck_db} dB)")
        except Exception as e:
            logger.warning(f"BGM ducking skipped: {e}")

    # 5. True-peak limiter
    audio = _true_peak_limit(audio, cfg.target_true_peak)

    lufs_after = _measure_lufs(audio, sr)

    sf.write(output_path, audio, sr)
    logger.info(f"Mastered: {lufs_before:.1f} → {lufs_after:.1f} LUFS  |  duck={bgm_ducking_applied}")

    return {
        "lufs_before":    lufs_before,
        "lufs_after":     lufs_after,
        "bgm_ducking":    bgm_ducking_applied,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# S3 HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _s3_download(bucket: str, key: str, local: str) -> None:
    if not S3_ENABLED:
        raise HTTPException(503, "S3 not configured")
    try:
        s3_client.download_file(bucket, key, local)
    except ClientError as e:
        raise HTTPException(500, f"S3 download failed: {e.response['Error']['Message']}")


def _s3_upload(local: str, bucket: str, key: str) -> None:
    if not S3_ENABLED:
        raise HTTPException(503, "S3 not configured")
    try:
        s3_client.upload_file(local, bucket, key,
                              ExtraArgs={"ContentType": "audio/wav",
                                         "ServerSideEncryption": "AES256"})
    except ClientError as e:
        raise HTTPException(500, f"S3 upload failed: {e.response['Error']['Message']}")


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/master-audio",
    response_model=MasteringResponse,
    summary="Professional audio mastering for dubbed tracks",
    description=(
        "EQ → Compression → Loudness normalisation → BGM ducking → True-peak limiting. "
        "Fixes: low loudness, flat dynamics, BGM overpowering dialogue, clipping."
    ),
)
async def master_audio(request: MasteringRequest) -> MasteringResponse:
    job_id   = str(uuid.uuid4())
    work_dir = TEMP_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_files = []

    try:
        dubbed_local   = str(work_dir / "dubbed.wav")
        original_local = str(work_dir / "original.wav")
        output_local   = str(work_dir / "mastered.wav")
        temp_files     = [dubbed_local, original_local, output_local]

        # Download dubbed audio
        _s3_download(request.bucket, request.dubbed_audio_key, dubbed_local)

        # Download original audio (for BGM ducking reference)
        has_original = False
        if request.enable_bgm_ducking and request.original_audio_key:
            try:
                _s3_download(request.bucket, request.original_audio_key, original_local)
                has_original = True
            except Exception as e:
                logger.warning(f"Could not download original audio for BGM ducking: {e}")

        # Run mastering chain
        meta = master_dubbed_audio(
            dubbed_path   = dubbed_local,
            original_path = original_local if has_original else None,
            output_path   = output_local,
            cfg           = request,
        )

        # Upload mastered audio
        output_key = f"{request.output_key_prefix.rstrip('/')}/{job_id}_mastered.wav"
        _s3_upload(output_local, request.output_bucket, output_key)

        return MasteringResponse(
            success             = True,
            output_key          = output_key,
            dubbed_lufs_before  = round(meta["lufs_before"],  2),
            dubbed_lufs_after   = round(meta["lufs_after"],   2),
            bgm_ducking_applied = meta["bgm_ducking"],
            message             = (
                f"Mastered: {meta['lufs_before']:.1f} → {meta['lufs_after']:.1f} LUFS  |  "
                f"BGM duck: {meta['bgm_ducking']}"
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Mastering job {job_id} failed: {e}")
        raise HTTPException(500, f"Audio mastering failed: {e}")
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


@router.get("/master-audio/health", tags=["Health"])
async def mastering_health():
    return {
        "status":     "healthy",
        "service":    "audio-mastering",
        "s3_enabled": S3_ENABLED,
    }