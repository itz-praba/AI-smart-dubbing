"""
tts_service/main.py
Standalone FastAPI app for Voice Cloning (XTTS v2).
Runs in its OWN venv (tts_env) with the numpy version that Coqui TTS needs.
Exposed on :8002 (internal only – Nginx proxies from :8000).
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

# Import the existing voice_cloning router unchanged
from python_controllers.voice_cloning import router as voicecloning_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TTS] %(levelname)s - %(message)s",
)

app = FastAPI(
    title="Voice Cloning Microservice",
    description="Standalone XTTS v2 voice cloning service (isolated numpy env)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Gateway is the real gatekeeper
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the existing voice_cloning router — ZERO code changes needed
app.include_router(voicecloning_router)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "tts-voice-cloning"}


@app.exception_handler(Exception)
async def global_exc_handler(request, exc):
    logging.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tts_service.main:app",
        host="127.0.0.1",   # internal only; Nginx faces the outside
        port=8002,
        reload=False,
    )
