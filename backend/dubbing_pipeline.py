"""
Complete Video Dubbing Pipeline
Orchestrates all 6 services to produce a fully dubbed video from a single video S3 key.

Pipeline:
  1. video-to-audio     → Extract audio from video
  2. speech-to-text     → Transcribe + diarize audio
  3. translate/timed    → Translate segments
  4. voice-clone-tts    → Clone each speaker's voice (per segment)
  5. lip-sync-align     → Align dubbed audio to original timing
  6. video-merge        → Merge dubbed audio back into video
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

LANG_2_TO_3 = {
    "en": "eng", "es": "spa", "fr": "fra", "de": "deu",
    "it": "ita", "pt": "por", "nl": "nld", "ru": "rus",
    "zh": "zho", "ja": "jpn", "ko": "kor", "ar": "ara",
    "hi": "hin", "tr": "tur", "pl": "pol", "uk": "ukr",
    "cs": "ces", "da": "dan", "fi": "fin", "no": "nor",
    "sv": "swe", "el": "ell", "he": "heb", "th": "tha",
    "vi": "vie", "ta": "tam"
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
    # Required
    video_bucket: str    = Field(..., min_length=1, max_length=63,
                                 description="S3 bucket containing the source video")
    video_key: str       = Field(..., min_length=1, max_length=1024,
                                 description="S3 key of the source video (e.g. 'videos/original.mp4')")
    target_language: str = Field(..., min_length=2, max_length=5,
                                 description="Target language code: es, fr, de, it, pt, hi …")
    # Optional
    output_bucket: Optional[str]      = Field(None, max_length=63,
                                              description="Output bucket. Defaults to video_bucket.")
    source_language: Optional[str]    = Field(None,
                                              description="Source language. Auto-detected if omitted.")
    whisper_model_size: str           = Field("base",
                                              description="Whisper model: tiny|base|small|medium|large-v3")
    diarize: bool                     = Field(True,
                                              description="Detect multiple speakers (needs HF token)")
    enable_lip_sync: bool             = Field(True,
                                              description="Align audio timing to original video")
    video_codec: str                  = Field("copy")
    audio_codec: str                  = Field("aac")
    audio_bitrate: str                = Field("192k")
    output_key_prefix: Optional[str]  = Field(None, max_length=512,
                                               description="S3 prefix for outputs. Default: dubbed/<id>/")

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
            "ar","hi","tr","pl","uk","cs","da","fi","no","sv","el","he","th","vi","ta"
        }
        if v.lower() not in supported:
            raise ValueError(f"Unsupported language '{v}'. Supported: {sorted(supported)}")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "video_bucket":       "ai-smart-dubbing",
                "video_key":          "videos/original.mp4",
                "target_language":    "es",
                "source_language":    "en",
                "whisper_model_size": "base",
                "diarize":            True,
                "enable_lip_sync":    True,
            }
        }


class PipelineStepStatus(BaseModel):
    step: str
    status: str
    duration_seconds: Optional[float] = None
    output_key: Optional[str] = None
    error: Optional[str] = None


class DubbingResponse(BaseModel):
    success: bool
    pipeline_job_id: str
    source_language: str
    target_language: str
    dubbed_video_bucket: str
    dubbed_video_key: str
    steps: List[PipelineStepStatus]
    total_duration_seconds: float
    segments_count: int
    speakers_detected: int
    message: str


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
        audio_bucket=output_bucket, output_key_prefix=f"{prefix}audio/",
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
        output_bucket=output_bucket, language=source_language,
        diarize=diarize, model_size=model_size, word_timestamps=True,
    )
    resp = await speech_to_text(req)
    if not resp.success:
        raise HTTPException(500, f"Step 2 failed: {resp.message}")
    logger.info(
        f"[{pipeline_id}] Step 2 ✓ → lang={resp.language}, "
        f"speakers={resp.speakers_detected}, segs={resp.segments_count}"
    )
    return resp.transcript_bucket, resp.transcript_key, resp.language, resp.speakers_detected


async def _step3_translate(pipeline_id, transcript_bucket, transcript_key,
                            source_lang, target_lang):
    from python_controllers.target_language import (
        TimedTranslationRequest, TimedSegment, translate_timed
    )
    transcript    = _s3_read_json(transcript_bucket, transcript_key)
    raw_segments  = transcript.get("segments", [])
    if not raw_segments:
        raise HTTPException(500, "Step 3 failed: transcript is empty")

    timed_segments = [
        TimedSegment(index=i, start=s["start"], end=s["end"], text=s["text"])
        for i, s in enumerate(raw_segments)
    ]
    req  = TimedTranslationRequest(segments=timed_segments,
                                    source_lang=source_lang, target_lang=target_lang)
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
    pipeline_id,
    translated_segments,
    audio_bucket,
    audio_key,
    output_bucket,
    target_lang,
    prefix,
):
    updated_segments = []

    for seg in translated_segments:
        output_key_prefix = f"{prefix}tts/segment_{seg['index']:04d}/"

        payload = {
            "speaker_audio_bucket": audio_bucket,
            "speaker_audio_key": audio_key,
            "translated_text": seg["translated_text"],
            "target_language": target_lang,
            "output_bucket": output_bucket,
            "output_key_prefix": output_key_prefix,
            "speed": 1.0,
        }

        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                "http://localhost:8002/ai/voice-clone-tts",
                json=payload,
            )

        if response.status_code != 200:
            raise HTTPException(
                500,
                f"Voice cloning microservice failed: {response.text}"
            )

        result = response.json()

        if not result.get("success"):
            raise HTTPException(
                500,
                "Voice cloning returned success=false"
            )

        tts_key = result["output_key"]

        logger.info(f"[{pipeline_id}] TTS generated: {tts_key}")

        updated_segments.append({**seg, "tts_audio_key": tts_key})

    logger.info(f"[{pipeline_id}] Step 4 ✓ → {len(updated_segments)} segments voiced")
    return updated_segments

async def _step5_lip_sync(pipeline_id, voiced_segments, output_bucket, prefix):
    from python_controllers.lip_sync import LipSyncRequest, TimedSegment as LS, lip_sync_align
    lipsync_segs = [
        LS(index=s["index"], start=s["start"], end=s["end"], tts_audio_key=s["tts_audio_key"])
        for s in voiced_segments
    ]
    req  = LipSyncRequest(
        segments=lipsync_segs,
        tts_audio_bucket=output_bucket,
        output_bucket=output_bucket,
        output_key_prefix=f"{prefix}lipsync/",
        enable_time_stretch=True,
    )
    resp = await lip_sync_align(req)
    if not resp.success:
        raise HTTPException(500, f"Step 5 (lip-sync) failed: {resp.message}")
    logger.info(f"[{pipeline_id}] Step 5 ✓ → {resp.output_key}")
    return resp.output_key


async def _step6_video_merge(pipeline_id, video_bucket, video_key, dubbed_audio_key,
                              output_bucket, target_lang, prefix,
                              video_codec, audio_codec, audio_bitrate):
    from python_controllers.video_rendering import VideoMergeRequest, AudioTrack, video_merge
    lang3 = LANG_2_TO_3.get(target_lang, target_lang[:3])
    req   = VideoMergeRequest(
        video_bucket=video_bucket, video_key=video_key,
        audio_tracks=[AudioTrack(
            audio_key=dubbed_audio_key,
            language=lang3,
            title=f"Dubbed – {target_lang.upper()}",
        )],
        audio_bucket=output_bucket,
        output_bucket=output_bucket,
        output_key_prefix=f"{prefix}final/",
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
    )
    resp  = await video_merge(req)
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
        "receive a fully dubbed video.\n\n"
        "Internally chains all 6 services:\n"
        "1. `video-to-audio` → extract audio\n"
        "2. `speech-to-text` → transcribe + diarize\n"
        "3. `translate/timed` → translate segments\n"
        "4. `voice-clone-tts` → generate dubbed speech\n"
        "5. `lip-sync-align` → align timing\n"
        "6. `video-merge` → produce final video"
    ),
)
async def dub_video(request: DubbingRequest) -> DubbingResponse:
    pipeline_id    = str(uuid.uuid4())
    pipeline_start = datetime.now()
    steps: List[PipelineStepStatus] = []

    output_bucket  = request.output_bucket or request.video_bucket
    prefix         = request.output_key_prefix or f"dubbed/{pipeline_id}/"

    logger.info(
        f"\n{'='*60}\n"
        f"[{pipeline_id}] PIPELINE START\n"
        f"  video  : {request.video_bucket}/{request.video_key}\n"
        f"  target : {request.target_language}\n"
        f"  output : {output_bucket}/{prefix}\n"
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

    # ── Step 1 ────────────────────────────────────────────────────────────────
    t = _step_timer()
    try:
        audio_bucket, audio_key = await _step1_video_to_audio(
            pipeline_id, request.video_bucket, request.video_key, output_bucket, prefix)
        record("1-video-to-audio", "done", t(), out_key=audio_key)
    except Exception as e:
        record("1-video-to-audio", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 1: {e}")

    # ── Step 2 ────────────────────────────────────────────────────────────────
    t = _step_timer()
    try:
        transcript_bucket, transcript_key, detected_lang, speakers = \
            await _step2_speech_to_text(
                pipeline_id, audio_bucket, audio_key, output_bucket,
                request.source_language, request.whisper_model_size, request.diarize)
        source_lang = request.source_language or detected_lang
        record("2-speech-to-text", "done", t(), out_key=transcript_key)
    except Exception as e:
        record("2-speech-to-text", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 2: {e}")

    try:
        transcript_data = _s3_read_json(transcript_bucket, transcript_key)
        segments_count  = len(transcript_data.get("segments", []))
    except Exception:
        segments_count = 0

    # ── Step 3 ────────────────────────────────────────────────────────────────
    t = _step_timer()
    try:
        translated_segments = await _step3_translate(
            pipeline_id, transcript_bucket, transcript_key,
            source_lang, request.target_language)
        record("3-translation", "done", t())
    except Exception as e:
        record("3-translation", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 3: {e}")

    # ── Step 4 ────────────────────────────────────────────────────────────────
    t = _step_timer()
    try:
        voiced_segments = await _step4_voice_clone(
            pipeline_id, translated_segments, audio_bucket, audio_key,
            output_bucket, request.target_language, prefix)
        record("4-voice-cloning", "done", t())
    except Exception as e:
        record("4-voice-cloning", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 4: {e}")

    # ── Step 5 ────────────────────────────────────────────────────────────────
    t = _step_timer()
    if request.enable_lip_sync:
        try:
            dubbed_audio_key = await _step5_lip_sync(
                pipeline_id, voiced_segments, output_bucket, prefix)
            record("5-lip-sync", "done", t(), out_key=dubbed_audio_key)
        except Exception as e:
            record("5-lip-sync", "failed", t(), error=str(e))
            raise HTTPException(500, f"Pipeline failed at Step 5: {e}")
    else:
        dubbed_audio_key = voiced_segments[-1]["tts_audio_key"] if voiced_segments else None
        record("5-lip-sync", "skipped", 0.0)

    if not dubbed_audio_key:
        raise HTTPException(500, "No dubbed audio produced")

    # ── Step 6 ────────────────────────────────────────────────────────────────
    t = _step_timer()
    try:
        final_video_key = await _step6_video_merge(
            pipeline_id, request.video_bucket, request.video_key,
            dubbed_audio_key, output_bucket, request.target_language, prefix,
            request.video_codec, request.audio_codec, request.audio_bitrate)
        record("6-video-merge", "done", t(), out_key=final_video_key)
    except Exception as e:
        record("6-video-merge", "failed", t(), error=str(e))
        raise HTTPException(500, f"Pipeline failed at Step 6: {e}")

    # ── Done ──────────────────────────────────────────────────────────────────
    total = (datetime.now() - pipeline_start).total_seconds()
    logger.info(
        f"\n{'='*60}\n"
        f"[{pipeline_id}] PIPELINE COMPLETE in {total:.1f}s\n"
        f"  output : {output_bucket}/{final_video_key}\n"
        f"{'='*60}"
    )

    return DubbingResponse(
        success=True,
        pipeline_job_id=pipeline_id,
        source_language=source_lang,
        target_language=request.target_language,
        dubbed_video_bucket=output_bucket,
        dubbed_video_key=final_video_key,
        steps=steps,
        total_duration_seconds=round(total, 2),
        segments_count=segments_count,
        speakers_detected=speakers,
        message=(
            f"✓ Video dubbed from '{source_lang}' → '{request.target_language}' "
            f"in {total:.1f}s | {segments_count} segments | "
            f"{speakers} speaker(s)"
        ),
    )


# ═══════════════════════════════════════════════════════════
# PIPELINE HEALTH
# ═══════════════════════════════════════════════════════════

@router.get("/dub-video/health", tags=["Dubbing Pipeline"])
async def pipeline_health():
    checks = {}
    for name, mod in {
        "audio_controller":  "python_controllers.audio_controller",
        "text_controller":   "python_controllers.text_controller",
        "target_language":   "python_controllers.target_language",
        "lip_sync":          "python_controllers.lip_sync",
        "video_rendering":   "python_controllers.video_rendering",
    }.items():
        try:
            __import__(mod)
            checks[name] = "ok"
        except Exception as e:
            checks[name] = f"error: {e}"

    try:
        __import__("python_controllers.voice_cloning")
        checks["voice_cloning"] = "ok"
    except ImportError:
        checks["voice_cloning"] = "not installed (TTS fallback will be used)"

    try:
        s3_client.list_buckets()
        checks["s3"] = "ok"
    except Exception as e:
        checks["s3"] = f"error: {e}"

    hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    checks["huggingface_token"] = "ok" if hf else "missing (diarization disabled)"

    all_ok = all("error" not in str(v) for v in checks.values())
    return {
        "status":    "healthy" if all_ok else "degraded",
        "device":    "cuda" if torch.cuda.is_available() else "cpu",
        "services":  checks,
        "timestamp": datetime.utcnow().isoformat(),
    }