import os
import uuid
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/ai", tags=["Translation"])

# =====================================================
# CONSTANTS AND CONFIGURATION
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Translation service using device: {DEVICE}")

# Supported languages
SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", 
    "zh", "ja", "ko", "ar", "tr", "pl", "vi", "uk",
    "cs", "da", "fi", "no", "sv", "el", "he", "th",
}

# Language names for better UX
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese", "uk": "Ukrainian",
    "cs": "Czech", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "sv": "Swedish", "el": "Greek", "he": "Hebrew", "th": "Thai"
}

# Marian MT models (faster, optimized for specific language pairs)
MARIAN_MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-it": "Helsinki-NLP/opus-mt-en-it",
    "it-en": "Helsinki-NLP/opus-mt-it-en",
    "en-pt": "Helsinki-NLP/opus-mt-en-pt",
    "pt-en": "Helsinki-NLP/opus-mt-pt-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "ta-en": "Helsinki-NLP/opus-mt-mul-en",
}

# Fallback model for unsupported pairs
M2M100_MODEL = "facebook/m2m100_418M"

# Translation parameters
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 512
MAX_BATCH_SIZE = 32
MAX_SEGMENTS = 1000
DEFAULT_BEAM_SIZE = 5

# Model cache directory
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# =====================================================
# PYDANTIC MODELS
# =====================================================
class TimedSegment(BaseModel):
    """Individual timed segment for translation"""
    index: int = Field(..., ge=0, description="Segment index")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., gt=0, description="End time in seconds")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    
    @validator('end')
    def validate_end_time(cls, v, values):
        """Ensure end time is after start time"""
        if 'start' in values and v <= values['start']:
            raise ValueError("End time must be greater than start time")
        return v
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text is not just whitespace"""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class TimedTranslationRequest(BaseModel):
    """Request model for timed translation"""
    segments: List[TimedSegment] = Field(..., min_items=1, max_items=MAX_SEGMENTS)
    source_lang: str = Field(..., min_length=2, max_length=5)
    target_lang: str = Field(..., min_length=2, max_length=5)
    beam_size: int = Field(DEFAULT_BEAM_SIZE, ge=1, le=10, description="Beam search size")
    batch_size: int = Field(16, ge=1, le=MAX_BATCH_SIZE, description="Batch processing size")
    
    @validator('source_lang', 'target_lang')
    def validate_language(cls, v):
        """Validate language code"""
        v = v.lower()
        if v not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGS))}"
            )
        return v
    
    @validator('target_lang')
    def validate_different_languages(cls, v, values):
        """Ensure source and target languages are different"""
        if 'source_lang' in values and v == values['source_lang']:
            raise ValueError("Source and target languages must be different")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "segments": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 3.5,
                        "text": "Hello, how are you?"
                    },
                    {
                        "index": 1,
                        "start": 3.5,
                        "end": 6.8,
                        "text": "I'm doing great, thank you!"
                    }
                ],
                "source_lang": "en",
                "target_lang": "es",
                "beam_size": 5,
                "batch_size": 16
            }
        }


class SimpleTranslationRequest(BaseModel):
    """Simple translation request without timing"""
    text: str = Field(..., min_length=1, max_length=10000)
    source_lang: str = Field(..., min_length=2, max_length=5)
    target_lang: str = Field(..., min_length=2, max_length=5)
    beam_size: int = Field(DEFAULT_BEAM_SIZE, ge=1, le=10)
    
    @validator('source_lang', 'target_lang')
    def validate_language(cls, v):
        v = v.lower()
        if v not in SUPPORTED_LANGS:
            raise ValueError(f"Unsupported language '{v}'")
        return v
    
    @validator('target_lang')
    def validate_different_languages(cls, v, values):
        if 'source_lang' in values and v == values['source_lang']:
            raise ValueError("Source and target languages must be different")
        return v
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class TranslatedSegment(BaseModel):
    """Individual translated segment"""
    index: int
    start: float
    end: float
    source_text: str
    translated_text: str
    confidence: Optional[float] = None


class TimedTranslationResponse(BaseModel):
    """Response model for timed translation"""
    success: bool
    job_id: str
    source_language: str
    target_language: str
    segments: List[TranslatedSegment]
    segments_count: int
    processing_time_seconds: float
    device: str
    model_type: str
    message: str


class SimpleTranslationResponse(BaseModel):
    """Response model for simple translation"""
    success: bool
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    processing_time_seconds: float
    device: str
    model_type: str


# =====================================================
# MODEL MANAGER
# =====================================================
class ModelManager:
    """Manages translation model loading and caching"""
    
    def __init__(self):
        self._model_cache: Dict[str, Tuple[Any, Any, Optional[int]]] = {}
        self._lock = asyncio.Lock()
    
    def get_model_type(self, src: str, tgt: str) -> str:
        """Determine which model to use"""
        key = f"{src}-{tgt}"
        if key in MARIAN_MODELS:
            return "marian"
        return "m2m100"
    
    async def load_model(
        self, 
        src: str, 
        tgt: str
    ) -> Tuple[Any, Any, Optional[int], str]:
        """
        Load translation model (with caching)
        
        Args:
            src: Source language code
            tgt: Target language code
            
        Returns:
            Tuple of (model, tokenizer, forced_bos_token_id, model_type)
        """
        key = f"{src}-{tgt}"
        
        # Check cache first (thread-safe)
        if key in self._model_cache:
            model, tokenizer, forced_bos = self._model_cache[key]
            model_type = self.get_model_type(src, tgt)
            logger.debug(f"Using cached model for {key}")
            return model, tokenizer, forced_bos, model_type
        
        # Load model (with lock to prevent duplicate loading)
        async with self._lock:
            # Double-check after acquiring lock
            if key in self._model_cache:
                model, tokenizer, forced_bos = self._model_cache[key]
                model_type = self.get_model_type(src, tgt)
                return model, tokenizer, forced_bos, model_type
            
            logger.info(f"Loading translation model for {key}")
            
            try:
                # Try Marian model first (faster for supported pairs)
                if key in MARIAN_MODELS:
                    model_name = MARIAN_MODELS[key]
                    tokenizer = MarianTokenizer.from_pretrained(
                        model_name,
                        cache_dir=MODEL_CACHE_DIR
                    )
                    model = MarianMTModel.from_pretrained(
                        model_name,
                        cache_dir=MODEL_CACHE_DIR
                    )
                    forced_bos = None
                    model_type = "marian"
                    logger.info(f"Loaded Marian model: {model_name}")
                
                # Fallback to M2M100 for unsupported pairs
                else:
                    tokenizer = M2M100Tokenizer.from_pretrained(
                        M2M100_MODEL,
                        cache_dir=MODEL_CACHE_DIR
                    )
                    model = M2M100ForConditionalGeneration.from_pretrained(
                        M2M100_MODEL,
                        cache_dir=MODEL_CACHE_DIR
                    )
                    tokenizer.src_lang = src
                    forced_bos = tokenizer.get_lang_id(tgt)
                    model_type = "m2m100"
                    logger.info(f"Loaded M2M100 model for {src}->{tgt}")
                
                # Move to device and set to eval mode
                model.to(DEVICE).eval()
                
                # Cache the model
                self._model_cache[key] = (model, tokenizer, forced_bos)
                
                logger.info(f"Successfully loaded and cached model for {key}")
                return model, tokenizer, forced_bos, model_type
                
            except Exception as e:
                logger.error(f"Failed to load model for {key}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load translation model: {str(e)}"
                )
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache),
            "device": DEVICE
        }
    
    def clear_cache(self):
        """Clear model cache (useful for memory management)"""
        logger.info("Clearing model cache")
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global model manager instance
model_manager = ModelManager()


# =====================================================
# TRANSLATION FUNCTIONS
# =====================================================
async def translate_batch(
    texts: List[str],
    src: str,
    tgt: str,
    beam_size: int = DEFAULT_BEAM_SIZE
) -> Tuple[List[str], str]:
    """
    Translate a batch of texts
    
    Args:
        texts: List of texts to translate
        src: Source language code
        tgt: Target language code
        beam_size: Beam search size
        
    Returns:
        Tuple of (translated_texts, model_type)
    """
    if not texts:
        return [], "none"
    
    try:
        # Load model
        model, tokenizer, forced_bos, model_type = await model_manager.load_model(src, tgt)
        
        # Tokenize inputs
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)
        
        # Generate translations
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=beam_size,
                forced_bos_token_id=forced_bos,
                max_length=MAX_OUTPUT_LENGTH,
                early_stopping=True,
                no_repeat_ngram_size=3  # Avoid repetition
            )
        
        # Decode outputs
        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        return translations, model_type
        
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# =====================================================
# API ENDPOINTS
# =====================================================
@router.post(
    "/translate/timed",
    response_model=TimedTranslationResponse,
    summary="Translate timed segments (e.g., from transcripts)",
    description="Translate multiple timed text segments while preserving timing information"
)
async def translate_timed(request: TimedTranslationRequest) -> TimedTranslationResponse:
    """
    Translate timed segments with preserved timing
    
    This endpoint is designed for translating transcript segments or subtitles
    where timing information needs to be preserved.
    
    Args:
        request: TimedTranslationRequest with segments and language parameters
        
    Returns:
        TimedTranslationResponse with translated segments and timing
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Starting timed translation job {job_id}: "
        f"{request.source_lang}->{request.target_lang}, "
        f"{len(request.segments)} segments"
    )
    
    try:
        # Extract texts from segments
        texts = [seg.text for seg in request.segments]
        
        # Split into batches for processing
        batches = split_into_batches(texts, request.batch_size)
        all_translations = []
        model_type = None
        
        # Process each batch
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)}")
            
            # Run translation in executor to avoid blocking
            translations, mt = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: asyncio.run(translate_batch(
                    batch,
                    request.source_lang,
                    request.target_lang,
                    request.beam_size
                ))
            )
            
            all_translations.extend(translations)
            if model_type is None:
                model_type = mt
        
        # Align translations with original segments
        translated_segments = []
        for seg, translated_text in zip(request.segments, all_translations):
            translated_segments.append(
                TranslatedSegment(
                    index=seg.index,
                    start=seg.start,
                    end=seg.end,
                    source_text=seg.text,
                    translated_text=translated_text,
                    confidence=None  # Could add confidence scoring in future
                )
            )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Job {job_id} completed successfully in {processing_time:.2f}s. "
            f"Translated {len(translated_segments)} segments"
        )
        
        return TimedTranslationResponse(
            success=True,
            job_id=job_id,
            source_language=LANGUAGE_NAMES.get(request.source_lang, request.source_lang),
            target_language=LANGUAGE_NAMES.get(request.target_lang, request.target_lang),
            segments=translated_segments,
            segments_count=len(translated_segments),
            processing_time_seconds=round(processing_time, 2),
            device=DEVICE,
            model_type=model_type or "unknown",
            message=f"Successfully translated {len(translated_segments)} segments"
        )
        
    except HTTPException:
        logger.error(f"Job {job_id} failed with HTTP error")
        raise
    
    except Exception as e:
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during translation: {str(e)}"
        )


@router.post(
    "/translate",
    response_model=SimpleTranslationResponse,
    summary="Translate text",
    description="Translate a single text from source to target language"
)
async def translate_text(request: SimpleTranslationRequest) -> SimpleTranslationResponse:
    """
    Simple text translation
    
    Args:
        request: SimpleTranslationRequest with text and language parameters
        
    Returns:
        SimpleTranslationResponse with translated text
    """
    start_time = time.time()
    
    logger.info(
        f"Translating text: {request.source_lang}->{request.target_lang}, "
        f"length: {len(request.text)} chars"
    )
    
    try:
        # Translate
        translations, model_type = await translate_batch(
            [request.text],
            request.source_lang,
            request.target_lang,
            request.beam_size
        )
        
        translated_text = translations[0] if translations else ""
        processing_time = time.time() - start_time
        
        logger.info(f"Translation completed in {processing_time:.2f}s")
        
        return SimpleTranslationResponse(
            success=True,
            source_text=request.text,
            translated_text=translated_text,
            source_language=LANGUAGE_NAMES.get(request.source_lang, request.source_lang),
            target_language=LANGUAGE_NAMES.get(request.target_lang, request.target_lang),
            processing_time_seconds=round(processing_time, 2),
            device=DEVICE,
            model_type=model_type
        )
        
    except Exception as e:
        logger.exception(f"Translation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


# =====================================================
# UTILITY ENDPOINTS
# =====================================================
@router.get("/translate/languages", tags=["Translation"])
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": [
            {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
            for code in sorted(SUPPORTED_LANGS)
        ],
        "total": len(SUPPORTED_LANGS),
        "marian_optimized_pairs": len(MARIAN_MODELS)
    }


@router.get("/translate/models", tags=["Translation"])
async def get_model_info():
    """Get information about available models"""
    cache_info = model_manager.get_cache_info()
    
    return {
        "device": DEVICE,
        "marian_models": len(MARIAN_MODELS),
        "marian_pairs": list(MARIAN_MODELS.keys()),
        "fallback_model": M2M100_MODEL,
        "cached_models": cache_info["cached_models"],
        "cache_size": cache_info["cache_size"],
        "max_batch_size": MAX_BATCH_SIZE,
        "max_segments": MAX_SEGMENTS
    }


@router.post("/translate/cache/clear", tags=["Translation"])
async def clear_model_cache():
    """Clear model cache (admin endpoint - should be protected in production)"""
    model_manager.clear_cache()
    return {
        "success": True,
        "message": "Model cache cleared successfully"
    }


@router.get("/translate/health", tags=["Health"])
async def health_check():
    """Health check for translation service"""
    cache_info = model_manager.get_cache_info()
    
    return {
        "status": "healthy",
        "service": "translation",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cached_models": cache_info["cache_size"],
        "supported_languages": len(SUPPORTED_LANGS)
    }