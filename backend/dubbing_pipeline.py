"""
dubbing_pipeline.py – Complete Video Dubbing Pipeline  v4.2
============================================================
Orchestrates all 7 services to produce a fully dubbed, professionally
mastered video from a single video S3 key.

Pipeline:
  1. video-to-audio     → Extract audio from video
  2. speech-to-text     → Transcribe + diarize audio
  3. translate/timed    → Translate segments
  4. voice-clone-tts    → Clone each speaker's voice  (TTS sidecar :8002)
  5. lip-sync-align     → Align dubbed audio to original timing
  5b. audio-mastering   → EQ + compress + loudness normalise + BGM duck  ← NEW
  6. video-merge        → Merge mastered audio back into video
"""

import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import boto3
import torch
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from botocore.config import Config
from botocore.exceptions import ClientError

from python_controllers.audio_controller import router as audio_router
from python_controllers.text_controller  import router as speech_router
from python_controllers.target_language  import router as translation_router
from python_controllers.lip_sync         import router as lipsync_router
from python_controllers.video_rendering  import router as rendering_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["Dubbing Pipeline"])

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# FIX B4: Read TTS sidecar URL from environment so it works in any deployment
# (Docker, K8s, remote host) rather than being hardcoded to localhost.
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:8002")

LANG_2_TO_3 = {
    "en": "eng", "es": "spa", "fr": "fra", "de": "deu",
    "it": "ita", "pt": "por", "nl": "nld", "ru": "rus",
    "zh": "zho", "ja": "jpn", "ko": "kor", "ar": "ara",
    "hi": "hin", "tr": "tur", "pl": "pol", "uk": "ukr",
    "cs": "ces", "da": "dan", "fi": "fin", "no": "nor",
    "sv": "swe", "el": "ell", "he": "heb", "th": "tha",
    "vi": "vie", "ta": "tam",
    "tgl": "tgl",  # Tanglish — no ISO 639-2 code; use "tgl" as custom tag
}

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
    config=Config(retries={"max_attempts": 3, "mode": "standard"}),
)


# ═══════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════

class DubbingRequest(BaseModel):
    # ── Required ──────────────────────────────────────────────
    video_bucket:    str = Field(..., min_length=1, max_length=63,
                                 description="S3 bucket containing the source video")
    video_key:       str = Field(..., min_length=1, max_length=1024,
                                 description="S3 key of the source video (e.g. 'videos/original.mp4')")
    target_language: str = Field(..., min_length=2, max_length=5,
                                 description="Target language code: es, fr, de, it, pt, hi …")

    # ── Optional – pipeline control ───────────────────────────
    output_bucket:      Optional[str] = Field(None, max_length=63,
                                              description="Output bucket. Defaults to video_bucket.")
    source_language:    Optional[str] = Field(None,
                                              description="Source language. Auto-detected if omitted.")
    whisper_model_size: str           = Field("base",
                                              description="Whisper model: tiny|base|small|medium|large-v3")
    diarize:            bool          = Field(True,
                                              description="Detect multiple speakers (needs HF token)")
    enable_lip_sync:    bool          = Field(True,
                                              description="Align audio timing to original video")
    video_codec:        str           = Field("copy")
    audio_codec:        str           = Field("aac")
    audio_bitrate:      str           = Field("192k")
    output_key_prefix:  Optional[str] = Field(None, max_length=512,
                                               description="S3 prefix for outputs. Default: dubbed/<id>/")

    # ── Optional – audio mastering control  ← NEW ─────────────
    enable_audio_mastering: bool  = Field(
        True,
        description=(
            "Run EQ → compression → loudness normalisation → BGM ducking "
            "on the dubbed audio before merging into the video. "
            "Fixes: low loudness, flat dynamics, BGM overpowering voice."
        )
    )
    enable_bgm_ducking: bool  = Field(
        True,
        description="Duck background music under dubbed voice (requires enable_audio_mastering=True)"
    )
    target_lufs:       float = Field(-16.0, ge=-30.0, le=-6.0,
                                     description="EBU R128 integrated loudness target (LUFS)")
    target_true_peak:  float = Field(-1.5,  ge=-6.0,  le=-0.5,
                                     description="True-peak ceiling (dBTP)")
    dialogue_peak_db:  float = Field(-3.0,  ge=-12.0, le=0.0,
                                     description="Dialogue bus peak ceiling (dBFS)")
    bgm_duck_db:       float = Field(-12.0, ge=-30.0, le=0.0,
                                     description="BGM attenuation level under speech (dB)")
    comp_threshold_db: float = Field(-24.0, ge=-40.0, le=0.0,
                                     description="Compressor threshold (dBFS)")
    comp_ratio:        float = Field(4.0,   ge=1.0,   le=20.0,
                                     description="Compression ratio (e.g. 4.0 = 4:1)")
    comp_makeup_db:    float = Field(6.0,   ge=0.0,   le=24.0,
                                     description="Make-up gain after compression (dB)")

    @validator("video_key")
    def validate_video_key(cls, v):
        if ".." in v or v.startswith("/"):
            raise ValueError("Invalid video key – path traversal detected")
        ext = Path(v).suffix.lower()
        allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        if ext not in allowed:
            raise ValueError(f"Unsupported video format '{ext}'. Allowed: {allowed}")
        return v

    @validator("target_language", "source_language")
    def validate_language(cls, v):
        if v is None:
            return v
        supported = {
            "en","es","fr","de","it","pt","nl","ru","zh","ja","ko",
            "ar","hi","tr","pl","uk","cs","da","fi","no","sv","el","he","th","vi",
            "ta",   # Tamil
            "tgl",  # Tanglish
        }
        if v.lower() not in supported:
            raise ValueError(f"Unsupported language '{v}'. Supported: {sorted(supported)}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "video_bucket":          "ai-smart-dubbing",
                "video_key":             "videos/original.mp4",
                "target_language":       "es",
                "source_language":       "en",
                "whisper_model_size":    "base",
                "diarize":               True,
                "enable_lip_sync":       True,
                "enable_audio_mastering": True,
                "enable_bgm_ducking":    True,
                "target_lufs":           -16.0,
            }
        }


class PipelineStepStatus(BaseModel):
    step:             str
    status:           str
    duration_seconds: Optional[float] = None
    output_key:       Optional[str]   = None
    error:            Optional[str]   = None


class DubbingResponse(BaseModel):
    success:              bool
    pipeline_job_id:      str
    source_language:      str
    target_language:      str
    dubbed_video_bucket:  str
    dubbed_video_key:     str
    steps:                List[PipelineStepStatus]
    total_duration_seconds: float
    segments_count:       int
    speakers_detected:    int
    mastering_applied:    bool          # ← NEW
    mastered_audio_key:   Optional[str] = None   # ← NEW
    message:              str


# ═══════════════════════════════════════════════════════════
# INTERNAL S3 HELPERS
# ═══════════════════════════════════════════════════════════

def _s3_read_json(bucket: str, key: str) -> dict:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def _step_timer():
    t0 = datetime.now()
    return lambda: (datetime.now() - t0).total_seconds()


# ═══════════════════════════════════════════════════════════
# STEP FUNCTIONS
# ═══════════════════════════════════════════════════════════

async def _step1_video_to_audio(pipeline_id, video_bucket, video_key, output_bucket, prefix):
    from python_controllers.audio_controller import VideoToAudioRequest, video_to_audio
    req  = VideoToAudioRequest(
        video_bucket=video_bucket, video_key=video_key,
        audio_bucket=output_bucket,
        output_key_prefix=f"{prefix}audio/",
    )
    resp = await video_to_audio(req)
    if not resp.success:
        raise HTTPException(500, f"Step 1 failed: {resp.message}")
    logger.info(f"[{pipeline_id}] Step 1 ✓ → {resp.audio_key}")
    return resp.audio_bucket, resp.audio_key


async def _step2_speech_to_text(pipeline_id, audio_bucket, audio_key, output_bucket,
                                 source_language, model_size, diarize):
    from python_controllers.text_controller import SpeechToTextRequest, speech_to_text
    req  = SpeechToTextRequest(
        audio_bucket=audio_bucket, audio_key=audio_key,
        output_bucket=output_bucket,
        language=source_language,
        diarize=diarize,
        model_size=model_size,
        word_timestamps=True,
    )
    resp = await speech_to_text(req)
    if not resp.success:
        raise HTTPException(500, f"Step 2 failed: {resp.message}")
    logger.info(
        f"[{pipeline_id}] Step 2 ✓ → lang={resp.language}, "
        f"speakers={resp.speakers_detected}, segs={resp.segments_count}"
    )
    return resp.transcript_bucket, resp.transcript_key, resp.language, resp.speakers_detected, resp.segments_count


async def _step3_translate(pipeline_id, transcript_bucket, transcript_key,
                            source_lang, target_lang):
    from python_controllers.target_language import (
        TimedTranslationRequest, TimedSegment, translate_timed
    )
    transcript   = _s3_read_json(transcript_bucket, transcript_key)
    raw_segments = transcript.get("segments", [])
    if not raw_segments:
        raise HTTPException(500, "Step 3 failed: transcript is empty")

    timed_segments = [
        TimedSegment(index=i, start=s["start"], end=s["end"], text=s["text"])
        for i, s in enumerate(raw_segments)
    ]
    req  = TimedTranslationRequest(
        segments=timed_segments,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    resp = await translate_timed(req)
    if not resp.success:
        raise HTTPException(500, "Step 3 (translation) failed")

    speaker_map = {i: s.get("speaker", "SPEAKER_00") for i, s in enumerate(raw_segments)}
    translated  = [
        {
            "index":           seg.index,
            "start":           seg.start,
            "end":             seg.end,
            "speaker":         speaker_map.get(seg.index, "SPEAKER_00"),
            "source_text":     seg.source_text,
            "translated_text": seg.translated_text,
        }
        for seg in resp.segments
    ]
    logger.info(f"[{pipeline_id}] Step 3 ✓ → {len(translated)} segments translated")
    return translated


async def _step4_voice_clone(
    pipeline_id: str,
    translated_segments: list,
    audio_bucket: str,
    audio_key: str,
    output_bucket: str,
    target_lang: str,
    prefix: str,
) -> str:
    """
    Calls the TTS sidecar for each segment.
    Uses the full-audio WAV (audio_key) as the speaker reference so the
    cloned voice matches the original speaker's timbre.

    FIX B8: Each segment call is retried up to 3 times with exponential back-off
    so a single transient sidecar blip doesn't waste all prior compute.
    """
    import asyncio

    _MAX_RETRIES   = 3
    _BACKOFF_BASE  = 2   # seconds — doubles each attempt: 2s, 4s, 8s

    async def _call_tts_segment(seg: dict, attempt: int = 1) -> dict:
        output_key_prefix = f"{prefix}tts/segment_{seg['index']:04d}/"
        payload = {
            "speaker_audio_bucket": audio_bucket,
            "speaker_audio_key":    audio_key,
            "translated_text":      seg["translated_text"],
            "target_language":      target_lang,
            "output_bucket":        output_bucket,
            "output_key_prefix":    output_key_prefix,
            "speed":                1.0,
        }
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(
                    f"{TTS_SERVICE_URL}/ai/voice-clone-tts",
                    json=payload,
                )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    f"[{pipeline_id}] TTS segment {seg['index']:04d} network error "
                    f"(attempt {attempt}/{_MAX_RETRIES}), retrying in {wait}s: {exc}"
                )
                await asyncio.sleep(wait)
                return await _call_tts_segment(seg, attempt + 1)
            raise HTTPException(
                503,
                f"TTS sidecar unreachable after {_MAX_RETRIES} attempts "
                f"for segment {seg['index']}: {exc}"
            )

        if response.status_code != 200:
            if attempt < _MAX_RETRIES and response.status_code in {429, 502, 503, 504}:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    f"[{pipeline_id}] TTS segment {seg['index']:04d} HTTP {response.status_code} "
                    f"(attempt {attempt}/{_MAX_RETRIES}), retrying in {wait}s"
                )
                await asyncio.sleep(wait)
                return await _call_tts_segment(seg, attempt + 1)
            raise HTTPException(
                500,
                f"Voice cloning sidecar failed for segment {seg['index']} "
                f"after {attempt} attempt(s): {response.text}"
            )

        result = response.json()
        if not result.get("success"):
            raise HTTPException(
                500,
                f"Voice cloning returned success=false for segment {seg['index']}"
            )
        return result

    updated_segments = []
    for seg in translated_segments:
        result  = await _call_tts_segment(seg)
        tts_key = result["output_key"]
        logger.info(f"[{pipeline_id}] TTS segment {seg['index']:04d} → {tts_key}")
        updated_segments.append({**seg, "tts_audio_key": tts_key})

    logger.info(f"[{pipeline_id}] Step 4 ✓ → {len(updated_segments)} segments voiced")
    return updated_segments


async def _concat_tts_segments(
    pipeline_id: str,
    voiced_segments: list,
    audio_bucket: str,
    output_bucket: str,
    prefix: str,
) -> str:
    """
    FIX B2: Concatenate per-segment TTS WAV files into a single timeline-aligned WAV.
    Used when enable_lip_sync=False so downstream steps receive a complete audio track
    instead of just the last segment's file.

    Each segment is placed at its original start time (silence-padded), then all
    channels are summed into one WAV and uploaded to S3.
    """
    import io
    import tempfile
    import uuid as _uuid

    job_id    = str(_uuid.uuid4())[:8]
    work_dir  = TEMP_DIR / f"concat_{job_id}"
    work_dir.mkdir(exist_ok=True)

    try:
        import numpy as np
        import soundfile as sf
        import boto3 as _boto3
        from botocore.config import Config as _BConfig

        _s3c = boto3.client(
            "s3",
            aws_access_key_id    =os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name          =os.getenv("AWS_REGION"),
            config=Config(retries={"max_attempts": 3, "mode": "standard"}),
        )

        # ── 1. Download all segment WAVs and read metadata ──────────────
        segs_data = []
        target_sr = None
        for seg in voiced_segments:
            local = str(work_dir / f"seg_{seg['index']:04d}.wav")
            _s3c.download_file(audio_bucket, seg["tts_audio_key"], local)
            data, sr = sf.read(local, dtype="float32")
            if data.ndim == 1:
                data = data[:, np.newaxis]
            target_sr = sr
            segs_data.append((seg["start"], seg["end"], data, sr))

        if not segs_data or target_sr is None:
            raise ValueError("No segments to concatenate")

        # ── 2. Determine total duration from last segment end time ───────
        total_secs  = max(end for _, end, _, _ in segs_data)
        total_secs  = max(total_secs, segs_data[-1][0] + segs_data[-1][2].shape[0] / target_sr)
        total_frames = int(np.ceil(total_secs * target_sr)) + target_sr  # +1s tail
        n_channels   = segs_data[0][2].shape[1]
        canvas       = np.zeros((total_frames, n_channels), dtype=np.float32)

        # ── 3. Place each segment at its original start offset ───────────
        for start, _, data, sr in segs_data:
            offset = int(start * target_sr)
            end_f  = offset + data.shape[0]
            if end_f > canvas.shape[0]:
                canvas = np.pad(canvas, ((0, end_f - canvas.shape[0]), (0, 0)))
            canvas[offset:end_f] += data

        # clip to [-1, 1]
        canvas = np.clip(canvas, -1.0, 1.0)
        if n_channels == 1:
            canvas = canvas[:, 0]

        # ── 4. Write + upload ────────────────────────────────────────────
        out_path   = str(work_dir / f"{job_id}_concat.wav")
        sf.write(out_path, canvas, target_sr, subtype="PCM_16")

        out_key = f"{prefix}lipsync/{job_id}_concat.wav"
        _s3c.upload_file(out_path, output_bucket, out_key)
        logger.info(f"[{pipeline_id}] B2-concat ✓ → {out_key} ({total_secs:.1f}s, {len(segs_data)} segs)")
        return out_key

    finally:
        import shutil as _shutil
        _shutil.rmtree(work_dir, ignore_errors=True)

async def _step5_lip_sync(
    pipeline_id: str,
    voiced_segments: list,
    output_bucket: str,
    prefix: str,
    target_lang: str,
) -> str:
    try:
        logger.info(
            f"[{pipeline_id}] Starting lip-sync with {len(voiced_segments)} segments"
        )

        from python_controllers.lip_sync import (
            LipSyncRequest, TimedSegment as LS, lip_sync_align
        )

        lipsync_segs = [
            LS(
                index=s["index"],
                start=s["start"],
                end=s["end"],
                tts_audio_key=s["tts_audio_key"]
            )
            for s in voiced_segments
        ]

        logger.info(
            f"[{pipeline_id}] Lip-sync segments prepared. "
            f"First segment: {lipsync_segs[0].tts_audio_key}"
        )

        req = LipSyncRequest(
            segments=lipsync_segs,
            tts_audio_bucket=output_bucket,
            output_bucket=output_bucket,
            output_key_prefix=f"{prefix}lipsync/",
            enable_time_stretch=True,
            target_language=target_lang,
        )

        resp = await lip_sync_align(req)

        if not resp.success:
            raise RuntimeError(f"Lip-sync service returned failure: {resp.message}")

        logger.info(f"[{pipeline_id}] Step 5 ✓ → {resp.output_key}")

        return resp.output_key

    except Exception as e:
        logger.exception(f"[{pipeline_id}] Lip-sync internal failure")
        raise


async def _step5b_audio_mastering(
    pipeline_id: str,
    dubbed_audio_key: str,           # lip-synced WAV key in S3
    original_audio_key: str,         # raw extracted audio key for BGM reference
    audio_bucket: str,               # FIX B1: source bucket where audio files actually live
    output_bucket: str,
    prefix: str,
    request: DubbingRequest,
) -> str:
    """
    NEW Step 5b – Professional audio mastering.

    Calls audio_mastering.master_audio() directly (no HTTP hop needed since
    audio_mastering runs in the same process as main.py).

    What it fixes:
      • Low loudness    → loudness normalisation to target_lufs (default −16 LUFS)
      • Flat dynamics   → compression (4:1, −24 dB threshold, +6 dB makeup)
      • Thin voice      → voice-presence EQ (+2 dB body, +3 dB presence, de-ess)
      • BGM overpowering→ smart BGM ducking (−12 dB under speech, smooth attack/release)
      • Clipping        → brickwall true-peak limiter at −1.5 dBTP

    Returns the S3 key of the mastered WAV.
    """
    from python_controllers.audio_mastering import (
        master_audio,
        MasteringRequest,
    )

    output_key_prefix = f"{prefix}mastered/"

    req = MasteringRequest(
        bucket               = audio_bucket,  # FIX B1: source bucket where dubbed+original audio live
        dubbed_audio_key     = dubbed_audio_key,
        original_audio_key   = original_audio_key,   # used for BGM extraction
        output_bucket        = output_bucket,
        output_key_prefix    = output_key_prefix,
        target_lufs          = request.target_lufs,
        target_true_peak     = request.target_true_peak,
        dialogue_peak_db     = request.dialogue_peak_db,
        bgm_duck_db          = request.bgm_duck_db,
        comp_threshold_db    = request.comp_threshold_db,
        comp_ratio           = request.comp_ratio,
        comp_makeup_db       = request.comp_makeup_db,
        enable_eq            = True,
        enable_compression   = True,
        enable_bgm_ducking   = request.enable_bgm_ducking,
    )

    resp = await master_audio(req)

    if not resp.success:
        raise HTTPException(500, f"Step 5b (audio-mastering) failed: {resp.message}")

    logger.info(
        f"[{pipeline_id}] Step 5b ✓ → {resp.output_key} | "
        f"LUFS {resp.dubbed_lufs_before:.1f} → {resp.dubbed_lufs_after:.1f} | "
        f"BGM duck: {resp.bgm_ducking_applied}"
    )
    return resp.output_key


async def _step6_video_merge(
    pipeline_id,
    video_bucket,
    video_key,
    dubbed_audio_key,
    output_bucket,
    target_lang,
    prefix,
    video_codec,
    audio_codec,
    audio_bitrate,
    mastering_applied: bool = False,
):
    from python_controllers.video_rendering import VideoMergeRequest, AudioTrack, video_merge

    lang3 = LANG_2_TO_3.get(target_lang, target_lang[:3])
    title = f"Dubbed – {target_lang.upper()}" + (" (Mastered)" if mastering_applied else "")

    req  = VideoMergeRequest(
        video_bucket    = video_bucket,
        video_key       = video_key,
        audio_tracks    = [AudioTrack(
            audio_key = dubbed_audio_key,
            language  = lang3,
            title     = title,
        )],
        audio_bucket        = output_bucket,
        output_bucket       = output_bucket,
        output_key_prefix   = f"{prefix}final/",
        video_codec         = video_codec,
        audio_codec         = audio_codec,
        audio_bitrate       = audio_bitrate,
        # Mastering was already applied in Step 5b → disable the built-in
        # mastering inside video_rendering to avoid double-processing
        enable_audio_mastering = False,
    )
    resp = await video_merge(req)
    if not resp.success:
        raise HTTPException(500, f"Step 6 (video-merge) failed: {resp.message}")
    logger.info(f"[{pipeline_id}] Step 6 ✓ → {resp.output_key}")
    return resp.output_key


# ═══════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════

@router.post(
    "/dub-video",
    response_model=DubbingResponse,
    summary="🎬 Full Video Dubbing Pipeline",
    description=(
        "**One-shot endpoint** – give a video S3 key + target language, "
        "receive a fully dubbed, professionally mastered video.\n\n"
        "Internally chains 7 services:\n"
        "1. `video-to-audio`  → extract audio\n"
        "2. `speech-to-text`  → transcribe + diarize\n"
        "3. `translate/timed` → translate segments\n"
        "4. `voice-clone-tts` → generate dubbed speech (TTS sidecar)\n"
        "5. `lip-sync-align`  → align timing to original video\n"
        "5b. `audio-mastering` → EQ + compress + loudness + BGM duck ← NEW\n"
        "6. `video-merge`     → produce final video"
    ),
)
async def dub_video(request: DubbingRequest) -> DubbingResponse:
    pipeline_id    = str(uuid.uuid4())
    pipeline_start = datetime.now()
    steps: List[PipelineStepStatus] = []

    output_bucket = request.output_bucket or request.video_bucket
    prefix        = request.output_key_prefix or f"dubbed/{pipeline_id}/"

    logger.info(
        f"\n{'='*60}\n"
        f"[{pipeline_id}] PIPELINE START\n"
        f"  video     : {request.video_bucket}/{request.video_key}\n"
        f"  target    : {request.target_language}\n"
        f"  output    : {output_bucket}/{prefix}\n"
        f"  mastering : {request.enable_audio_mastering}  "
        f"bgm_duck={request.enable_bgm_ducking}\n"
        f"{'='*60}"
    )

    def record(step, status, t, out_key=None, error=None):
        steps.append(PipelineStepStatus(
            step=step, status=status,
            duration_seconds=round(t, 2),
            output_key=out_key, error=error,
        ))
        icon = "✓" if status == "done" else ("⚡" if status == "skipped" else "✗")
        logger.info(f"[{pipeline_id}] {icon} {step}: {status} ({t:.1f}s)")

    # ── Step 1 – video → audio ───────────────────────────────────────────────
    t = _step_timer()
    try:
        audio_bucket, audio_key = await _step1_video_to_audio(
            pipeline_id, request.video_bucket, request.video_key,
            output_bucket, prefix)
        record("1-video-to-audio", "done", t(), out_key=audio_key)
    except Exception as e:
        record("1-video-to-audio", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 1: {e}")

    # ── Step 2 – speech → text ───────────────────────────────────────────────
    t = _step_timer()
    try:
        # FIX B5: segments_count now comes from the Step 2 response — no extra S3 GET needed
        transcript_bucket, transcript_key, detected_lang, speakers, segments_count = \
            await _step2_speech_to_text(
                pipeline_id, audio_bucket, audio_key, output_bucket,
                request.source_language, request.whisper_model_size, request.diarize)
        source_lang = request.source_language or detected_lang
        record("2-speech-to-text", "done", t(), out_key=transcript_key)
    except Exception as e:
        record("2-speech-to-text", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 2: {e}")

    # ── Step 3 – translate ───────────────────────────────────────────────────
    t = _step_timer()
    try:
        translated_segments = await _step3_translate(
            pipeline_id, transcript_bucket, transcript_key,
            source_lang, request.target_language)
        record("3-translation", "done", t())
    except Exception as e:
        record("3-translation", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 3: {e}")

    # ── Step 4 – voice clone (TTS sidecar) ───────────────────────────────────
    t = _step_timer()
    try:
        voiced_segments = await _step4_voice_clone(
            pipeline_id, translated_segments,
            audio_bucket, audio_key,
            output_bucket, request.target_language, prefix)
        record("4-voice-cloning", "done", t())
    except Exception as e:
        record("4-voice-cloning", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 4: {e}")

    # ── Step 5 – lip-sync align ───────────────────────────────────────────────
    t = _step_timer()
    if request.enable_lip_sync:
        try:
            dubbed_audio_key = await _step5_lip_sync(
                pipeline_id, voiced_segments, output_bucket, prefix, request.target_language)
            record("5-lip-sync", "done", t(), out_key=dubbed_audio_key)
        except Exception as e:
            record("5-lip-sync", "failed", t(), error=str(e))
            raise HTTPException(500, f"Pipeline failed at Step 5: {e}")
    else:
        # FIX B2: Concatenate all per-segment TTS WAVs into a single timeline-aligned
        # audio file instead of returning only the last segment (which caused the final
        # video to contain only the last translated sentence).
        try:
            dubbed_audio_key = await _concat_tts_segments(
                pipeline_id, voiced_segments, audio_bucket, output_bucket, prefix)
            record("5-lip-sync", "skipped (concat fallback)", t(), out_key=dubbed_audio_key)
        except Exception as e:
            record("5-lip-sync", "failed (concat fallback)", t(), error=str(e))
            raise HTTPException(500, f"Pipeline failed at Step 5 concat fallback: {e}")

    if not dubbed_audio_key:
        raise HTTPException(500, "No dubbed audio produced after Step 5")

    # ── Step 5b – audio mastering  ← NEW ────────────────────────────────────
    mastering_applied  = False
    mastered_audio_key = None

    t = _step_timer()
    if request.enable_audio_mastering:
        try:
            mastered_audio_key = await _step5b_audio_mastering(
                pipeline_id        = pipeline_id,
                dubbed_audio_key   = dubbed_audio_key,
                original_audio_key = audio_key,     # raw extracted audio = BGM reference
                audio_bucket       = audio_bucket,  # FIX B1: source bucket
                output_bucket      = output_bucket,
                prefix             = prefix,
                request            = request,
            )
            dubbed_audio_key  = mastered_audio_key   # downstream uses mastered version
            mastering_applied = True
            record("5b-audio-mastering", "done", t(), out_key=mastered_audio_key)
        except Exception as e:
            # Non-fatal: log the error and continue with un-mastered audio
            logger.error(
                f"[{pipeline_id}] ⚠ Step 5b (audio-mastering) failed – "
                f"continuing with raw dubbed audio. Error: {e}"
            )
            record("5b-audio-mastering", "failed", t(), error=str(e))
            # Do NOT re-raise: we still want a dubbed video even if mastering fails
    else:
        record("5b-audio-mastering", "skipped", 0.0)

    # ── Step 6 – video merge ──────────────────────────────────────────────────
    t = _step_timer()
    try:
        final_video_key = await _step6_video_merge(
            pipeline_id,
            request.video_bucket, request.video_key,
            dubbed_audio_key,
            output_bucket, request.target_language, prefix,
            request.video_codec, request.audio_codec, request.audio_bitrate,
            mastering_applied=mastering_applied,
        )
        record("6-video-merge", "done", t(), out_key=final_video_key)
    except Exception as e:
        record("6-video-merge", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 6: {e}")

    # ── Done ──────────────────────────────────────────────────────────────────
    total = (datetime.now() - pipeline_start).total_seconds()
    logger.info(
        f"\n{'='*60}\n"
        f"[{pipeline_id}] PIPELINE COMPLETE in {total:.1f}s\n"
        f"  output    : {output_bucket}/{final_video_key}\n"
        f"  mastered  : {mastering_applied}\n"
        f"{'='*60}"
    )

    return DubbingResponse(
        success              = True,
        pipeline_job_id      = pipeline_id,
        source_language      = source_lang,
        target_language      = request.target_language,
        dubbed_video_bucket  = output_bucket,
        dubbed_video_key     = final_video_key,
        steps                = steps,
        total_duration_seconds = round(total, 2),
        segments_count       = segments_count,
        speakers_detected    = speakers,
        mastering_applied    = mastering_applied,
        mastered_audio_key   = mastered_audio_key,
        message=(
            f"✓ Video dubbed '{source_lang}' → '{request.target_language}' "
            f"in {total:.1f}s | {segments_count} segments | "
            f"{speakers} speaker(s) | "
            f"mastering: {'✓' if mastering_applied else '✗'}"
        ),
    )


# ═══════════════════════════════════════════════════════════
# PIPELINE HEALTH
# ═══════════════════════════════════════════════════════════

@router.get("/dub-video/health", tags=["Dubbing Pipeline"])
async def pipeline_health():
    checks = {}

    # Individual service modules
    for name, mod in {
        "audio_controller":  "python_controllers.audio_controller",
        "text_controller":   "python_controllers.text_controller",
        "target_language":   "python_controllers.target_language",
        "lip_sync":          "python_controllers.lip_sync",
        "video_rendering":   "python_controllers.video_rendering",
        "audio_mastering":   "python_controllers.audio_mastering",   # ← NEW
    }.items():
        try:
            __import__(mod)
            checks[name] = "ok"
        except Exception as e:
            checks[name] = f"error: {e}"

    # TTS sidecar (optional – runs in separate process)
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{TTS_SERVICE_URL}/health")
        checks["voice_cloning_sidecar"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        checks["voice_cloning_sidecar"] = "unreachable (sidecar :8002 not running)"

    # S3
    try:
        s3_client.list_buckets()
        checks["s3"] = "ok"
    except Exception as e:
        checks["s3"] = f"error: {e}"

    # HuggingFace token for diarization
    hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    checks["huggingface_token"] = "ok" if hf else "missing (diarization disabled)"

    all_ok = all("error" not in str(v) for v in checks.values())
    return {
        "status":    "healthy" if all_ok else "degraded",
        "device":    "cuda" if torch.cuda.is_available() else "cpu",
        "services":  checks,
        "pipeline_steps": [
            "1-video-to-audio",
            "2-speech-to-text",
            "3-translation",
            "4-voice-cloning",
            "5-lip-sync",
            "5b-audio-mastering",   # ← NEW
            "6-video-merge",
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }