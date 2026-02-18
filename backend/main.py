"""
main.py – Main FastAPI application v4.1
"""

import os
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

load_dotenv()

from config.db import connect_to_database
from middleware.security import setup_middleware

from python_controllers.auth_controller import router as auth_router
from python_controllers.contact_controller import router as contact_router
from python_controllers.forgot_password import router as password_router
from python_controllers.audio_controller import router as video_router
from python_controllers.text_controller import router as speech_router
from python_controllers.target_language import router as translation_router
from python_controllers.lip_sync import router as lipsync_router
from python_controllers.video_rendering import router as video_merge_router
from dubbing_pipeline import router as pipeline_router


# ── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_database()
    yield
    # Optional: await close_database()


# ── App (ONLY ONE INSTANCE) ─────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="Complete Media Processing API",
    description=(
        "End-to-end AI video dubbing.\n\n"
        "**One-shot:** `POST /ai/dub-video`\n\n"
        "Voice cloning is handled by an internal sidecar service."
    ),
    version="4.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "SUPER_SECRET_KEY"),
    same_site="lax",
    https_only=False,
)

setup_middleware(app, {
    "rate_limit": {
        "enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
        "requests_per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
        "requests_per_hour":   int(os.getenv("RATE_LIMIT_PER_HOUR", "1000")),
    },
    "api_key": {
        "enabled": os.getenv("API_KEY_ENABLED", "false").lower() == "true",
        "keys": os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else [],
    },
    "timeout": {
        "enabled": True,
        "timeout_seconds": int(os.getenv("REQUEST_TIMEOUT", "3600")),
    },
    "request_size_limit": {
        "enabled": True,
        "max_size_mb": int(os.getenv("MAX_REQUEST_SIZE_MB", "10")),
    },
})


# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(contact_router)
app.include_router(password_router)
app.include_router(pipeline_router)
app.include_router(video_router)
app.include_router(speech_router)
app.include_router(translation_router)
app.include_router(lipsync_router)
app.include_router(video_merge_router)


# ── TTS sidecar proxy ────────────────────────────────────────────────────────
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://127.0.0.1:8002")

async def call_tts_service(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=3600) as client:
        try:
            resp = await client.post(f"{TTS_SERVICE_URL}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"TTS service error: {e.response.text}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"TTS service unavailable: {str(e)}",
            )


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Complete Media Processing API",
        "version": "4.1.0",
        "architecture": "main(:8001) + tts_sidecar(:8002) behind nginx(:8000)",
    }


@app.get("/health")
async def global_health():
    tts_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{TTS_SERVICE_URL}/health")
            tts_status = "available" if r.status_code == 200 else "degraded"
    except Exception:
        tts_status = "unreachable"

    return {
        "status": "healthy",
        "services": {
            "video_to_audio": "available",
            "speech_to_text": "available",
            "translation":    "available",
            "voice_cloning":  tts_status,
            "lip_sync":       "available",
            "video_merge":    "available",
        },
    }


# ── Global exception handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
