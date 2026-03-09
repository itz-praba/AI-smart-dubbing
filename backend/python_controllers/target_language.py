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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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
    "ta",   # Tamil
    "tgl",  # Tanglish (romanised Tamil in Latin script)
}

# Language names for better UX
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "tr": "Turkish", "pl": "Polish", "vi": "Vietnamese", "uk": "Ukrainian",
    "cs": "Czech", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "sv": "Swedish", "el": "Greek", "he": "Hebrew", "th": "Thai",
    "ta": "Tamil",
    "tgl": "Tanglish",
}

# ── Marian MT models (fast, pair-specific) ────────────────────────────────────
# NOTE: "en-ta" and "ta-en" have been intentionally REMOVED from this table.
#
# Root cause of the empty/garbled Tamil translations:
#   Helsinki-NLP/opus-mt-en-mul is a generic multilingual Marian model that does
#   NOT reliably support Tamil.  It either outputs empty strings or romanised
#   gibberish because Tamil was not a primary training target for that checkpoint.
#
# Fix: Tamil is now routed exclusively to NLLB-200 (see NLLB_PAIRS below), which
# was trained on 200 languages including Tamil and produces correct Tamil script.
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
}

# ── NLLB-200 language code mapping ───────────────────────────────────────────
# facebook/nllb-200-distilled-600M uses BCP-47 script-tagged codes, not ISO 639-1.
# Any language pair listed here is routed to NLLB instead of M2M100.
# NLLB is the correct choice for Tamil: it was trained on Tamil script (tam_Taml)
# and produces accurate, fluent translations — unlike opus-mt-en-mul.
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

NLLB_LANG_CODES: Dict[str, str] = {
    "en":  "eng_Latn",
    "es":  "spa_Latn",
    "fr":  "fra_Latn",
    "de":  "deu_Latn",
    "it":  "ita_Latn",
    "pt":  "por_Latn",
    "nl":  "nld_Latn",
    "ru":  "rus_Cyrl",
    "zh":  "zho_Hans",
    "ja":  "jpn_Jpan",
    "ko":  "kor_Hang",
    "ar":  "arb_Arab",
    "hi":  "hin_Deva",
    "tr":  "tur_Latn",
    "pl":  "pol_Latn",
    "vi":  "vie_Latn",
    "uk":  "ukr_Cyrl",
    "cs":  "ces_Latn",
    "da":  "dan_Latn",
    "fi":  "fin_Latn",
    "no":  "nob_Latn",
    "sv":  "swe_Latn",
    "el":  "ell_Grek",
    "he":  "heb_Hebr",
    "th":  "tha_Thai",
    "ta":  "tam_Taml",   # Tamil — correct NLLB script tag, produces Tamil script ✓
}

# Language pairs that MUST use NLLB (Tamil + any pair not in MARIAN_MODELS).
# A pair is routed to NLLB if either the source or target appears in NLLB_LANG_CODES
# AND the pair is not covered by a Marian model.
def _needs_nllb(src: str, tgt: str) -> bool:
    """Return True if this pair should use NLLB-200 instead of M2M100."""
    pair = f"{src}-{tgt}"
    if pair in MARIAN_MODELS:
        return False   # Marian handles it
    # Route to NLLB if either language has a known NLLB code
    return src in NLLB_LANG_CODES or tgt in NLLB_LANG_CODES


# ── M2M100 fallback (for pairs not covered by Marian or NLLB) ────────────────
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
        if 'start' in values and v <= values['start']:
            raise ValueError("End time must be greater than start time")
        return v

    @validator('text')
    def validate_text(cls, v):
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
        v = v.lower()
        if v not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported language '{v}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGS))}"
            )
        return v

    @validator('target_lang')
    def validate_different_languages(cls, v, values):
        if 'source_lang' in values and v == values['source_lang']:
            raise ValueError("Source and target languages must be different")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "segments": [
                    {"index": 0, "start": 0.0,  "end": 3.5, "text": "Hello, how are you?"},
                    {"index": 1, "start": 3.5,  "end": 6.8, "text": "I'm doing great, thank you!"}
                ],
                "source_lang": "en",
                "target_lang": "ta",
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
        pair = f"{src}-{tgt}"
        if pair in MARIAN_MODELS:
            return "marian"
        if _needs_nllb(src, tgt):
            return "nllb"
        return "m2m100"

    async def load_model(
        self,
        src: str,
        tgt: str,
    ) -> Tuple[Any, Any, Optional[int], str]:
        """
        Load translation model (with caching).

        Engine selection priority:
          1. Marian  — fast, pair-specific (covers common European pairs)
          2. NLLB-200 — accurate for Tamil, Arabic, CJK, and 200 languages
          3. M2M100  — generic fallback for remaining pairs

        FIX: Tamil was previously routed to Helsinki-NLP/opus-mt-en-mul (Marian)
        which does not reliably support Tamil and produced empty/garbled output.
        Tamil is now routed to NLLB-200 (facebook/nllb-200-distilled-600M) which
        was trained on tam_Taml and produces correct Tamil script output.
        """
        key = f"{src}-{tgt}"

        if key in self._model_cache:
            model, tokenizer, forced_bos = self._model_cache[key]
            model_type = self.get_model_type(src, tgt)
            logger.debug(f"Using cached model for {key}")
            return model, tokenizer, forced_bos, model_type

        async with self._lock:
            # Double-check after acquiring lock
            if key in self._model_cache:
                model, tokenizer, forced_bos = self._model_cache[key]
                model_type = self.get_model_type(src, tgt)
                return model, tokenizer, forced_bos, model_type

            logger.info(f"Loading translation model for {key}")

            try:
                if key in MARIAN_MODELS:
                    # ── Marian (fast, specific pairs) ─────────────────────
                    model_name = MARIAN_MODELS[key]
                    tokenizer = MarianTokenizer.from_pretrained(
                        model_name, cache_dir=MODEL_CACHE_DIR
                    )
                    model = MarianMTModel.from_pretrained(
                        model_name, cache_dir=MODEL_CACHE_DIR
                    )
                    forced_bos = None
                    model_type = "marian"
                    logger.info(f"Loaded Marian model: {model_name}")

                elif _needs_nllb(src, tgt):
                    # ── NLLB-200 (Tamil, Arabic, CJK, 200 languages) ──────
                    # FIX: Tamil must use NLLB — opus-mt-en-mul cannot produce
                    # Tamil script and returns empty/garbled strings.
                    nllb_src = NLLB_LANG_CODES.get(src)
                    nllb_tgt = NLLB_LANG_CODES.get(tgt)
                    if not nllb_src or not nllb_tgt:
                        raise ValueError(
                            f"No NLLB language code for '{src}' or '{tgt}'. "
                            f"Add entries to NLLB_LANG_CODES."
                        )
                    tokenizer = AutoTokenizer.from_pretrained(
                        NLLB_MODEL,
                        cache_dir=MODEL_CACHE_DIR,
                        src_lang=nllb_src,
                    )
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        NLLB_MODEL, cache_dir=MODEL_CACHE_DIR
                    )
                    # forced_bos_token_id tells the decoder which language to generate
                    forced_bos = tokenizer.convert_tokens_to_ids(nllb_tgt)
                    model_type = "nllb"
                    logger.info(
                        f"Loaded NLLB-200 model for {src}({nllb_src}) → "
                        f"{tgt}({nllb_tgt}), forced_bos={forced_bos}"
                    )

                else:
                    # ── M2M100 fallback ───────────────────────────────────
                    tokenizer = M2M100Tokenizer.from_pretrained(
                        M2M100_MODEL, cache_dir=MODEL_CACHE_DIR
                    )
                    model = M2M100ForConditionalGeneration.from_pretrained(
                        M2M100_MODEL, cache_dir=MODEL_CACHE_DIR
                    )
                    tokenizer.src_lang = src
                    forced_bos = tokenizer.get_lang_id(tgt)
                    model_type = "m2m100"
                    logger.info(f"Loaded M2M100 model for {src}→{tgt}")

                model.to(DEVICE).eval()
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
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache),
            "device": DEVICE
        }

    def clear_cache(self):
        logger.info("Clearing model cache")
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global model manager instance
model_manager = ModelManager()


# =====================================================
# LANGUAGE-SPECIFIC TRANSLATION STYLE HINTS
# =====================================================
# Style hints nudge seq2seq models toward casual/energetic register for dubbing.
#
# IMPORTANT FIX: Style hints are ONLY applied to NLLB and M2M100 models.
# Marian models use SentencePiece tokenizers that cannot handle bracket-prefixed
# prompts — prepending a hint to Marian input causes the tokenizer to produce
# unexpected tokens, which in turn makes the model output empty strings.
# The translate_batch function checks model_type before applying hints.
TRANSLATION_STYLE_HINTS: Dict[str, str] = {
    "ko": "[casual energetic conversational Korean 해체, preserve exclamations and emotion] ",
    "ja": "[casual conversational Japanese ため口, match energy and emotion of source] ",
    "zh": "[natural conversational Mandarin, match energy and emotion of source] ",
    "ta": "[natural conversational Tamil, colloquial register, match energy and emotion of source] ",
    "tgl": (
        "[Tanglish: romanised Tamil mixed with English, casual colloquial, "
        "write Tamil words in Latin script, match energy and emotion of source] "
    ),
}


# =====================================================
# TRANSLATION FUNCTIONS
# =====================================================
async def translate_batch(
    texts: List[str],
    src: str,
    tgt: str,
    beam_size: int = DEFAULT_BEAM_SIZE,
) -> Tuple[List[str], str]:
    """
    Translate a batch of texts using the best available model for the language pair.

    Engine routing:
      • Marian  → fast European pairs (NO style hints — breaks SentencePiece tokenizer)
      • NLLB    → Tamil, Arabic, CJK, 200-language coverage (style hints safe here)
      • M2M100  → generic fallback (style hints safe here)

    Tanglish normalisation:
      "tgl" has no model token.  We route it as "en" so the model uses its English
      vocabulary for Latin-script output; the style hint steers register to Tanglish.

    FIX — empty Tamil translations:
      The previous code mapped "en-ta" to Helsinki-NLP/opus-mt-en-mul, which is a
      multilingual Marian model with very poor Tamil support.  It produced empty
      strings or romanised gibberish.  Tamil is now routed to NLLB-200 which was
      explicitly trained on tam_Taml and reliably produces Tamil script output.

    FIX — style hint breaking Marian:
      Style hints are now gated behind `if model_type != "marian"` so they are
      never prepended to Marian inputs.
    """
    if not texts:
        return [], "none"

    # Normalise Tanglish → English for model routing; style hint handles register
    model_src = "en" if src == "tgl" else src
    model_tgt = "en" if tgt == "tgl" else tgt

    model, tokenizer, forced_bos, model_type = await model_manager.load_model(
        model_src, model_tgt
    )

    # ── Style hints: ONLY for NLLB / M2M100, never Marian ────────────────────
    # Marian's SentencePiece tokenizer treats the bracket prefix as unknown tokens
    # which corrupts the input and causes the model to output empty strings.
    style_hint = ""
    if model_type != "marian":
        style_hint = TRANSLATION_STYLE_HINTS.get(tgt, "")

    if style_hint:
        hinted_texts = [style_hint + t for t in texts]
    else:
        hinted_texts = texts

    # ── For NLLB: set src_lang on the tokenizer before each call ─────────────
    # The tokenizer is shared/cached; we must set src_lang here because it may
    # have been set to a different language by a previous call.
    if model_type == "nllb":
        nllb_src = NLLB_LANG_CODES.get(model_src)
        if nllb_src:
            tokenizer.src_lang = nllb_src

    try:
        inputs = tokenizer(
            hinted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(DEVICE)

        with torch.no_grad():
            generate_kwargs: Dict[str, Any] = dict(
                **inputs,
                num_beams=beam_size,
                max_length=MAX_OUTPUT_LENGTH,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            if forced_bos is not None:
                generate_kwargs["forced_bos_token_id"] = forced_bos

            generated = model.generate(**generate_kwargs)

        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Strip style hint echo if model leaked it into the output
        if style_hint:
            translations = [t.replace(style_hint.strip(), "").strip() for t in translations]

        # Safety check: if any translation is empty, log a warning
        empty_count = sum(1 for t in translations if not t.strip())
        if empty_count:
            logger.warning(
                f"translate_batch: {empty_count}/{len(translations)} empty translations "
                f"for {src}→{tgt} using {model_type}. Check model coverage."
            )

        return translations, model_type

    except Exception as e:
        logger.error(f"Translation failed for {src}→{tgt}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
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
    job_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Starting timed translation job {job_id}: "
        f"{request.source_lang}->{request.target_lang}, "
        f"{len(request.segments)} segments"
    )

    try:
        texts = [seg.text for seg in request.segments]
        batches = split_into_batches(texts, request.batch_size)
        all_translations: List[str] = []
        model_type = None

        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)}")
            # Run translation in executor to avoid blocking the event loop
            translations, mt = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda b=batch: asyncio.run(translate_batch(
                    b,
                    request.source_lang,
                    request.target_lang,
                    request.beam_size,
                ))
            )
            all_translations.extend(translations)
            if model_type is None:
                model_type = mt

        translated_segments = [
            TranslatedSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                source_text=seg.text,
                translated_text=translated_text,
                confidence=None,
            )
            for seg, translated_text in zip(request.segments, all_translations)
        ]

        processing_time = time.time() - start_time
        logger.info(
            f"Job {job_id} completed in {processing_time:.2f}s — "
            f"{len(translated_segments)} segments translated via {model_type}"
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
    start_time = time.time()

    logger.info(
        f"Translating text: {request.source_lang}->{request.target_lang}, "
        f"length: {len(request.text)} chars"
    )

    try:
        translations, model_type = await translate_batch(
            [request.text],
            request.source_lang,
            request.target_lang,
            request.beam_size,
        )

        translated_text = translations[0] if translations else ""
        processing_time = time.time() - start_time

        logger.info(f"Translation completed in {processing_time:.2f}s via {model_type}")

        return SimpleTranslationResponse(
            success=True,
            source_text=request.text,
            translated_text=translated_text,
            source_language=LANGUAGE_NAMES.get(request.source_lang, request.source_lang),
            target_language=LANGUAGE_NAMES.get(request.target_lang, request.target_lang),
            processing_time_seconds=round(processing_time, 2),
            device=DEVICE,
            model_type=model_type,
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
    return {
        "supported_languages": [
            {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
            for code in sorted(SUPPORTED_LANGS)
        ],
        "total": len(SUPPORTED_LANGS),
        "marian_optimized_pairs": len(MARIAN_MODELS),
        "nllb_supported_languages": sorted(NLLB_LANG_CODES.keys()),
    }


@router.get("/translate/models", tags=["Translation"])
async def get_model_info():
    cache_info = model_manager.get_cache_info()
    return {
        "device": DEVICE,
        "marian_models": len(MARIAN_MODELS),
        "marian_pairs": list(MARIAN_MODELS.keys()),
        "nllb_model": NLLB_MODEL,
        "nllb_languages": sorted(NLLB_LANG_CODES.keys()),
        "fallback_model": M2M100_MODEL,
        "cached_models": cache_info["cached_models"],
        "cache_size": cache_info["cache_size"],
        "max_batch_size": MAX_BATCH_SIZE,
        "max_segments": MAX_SEGMENTS,
    }


@router.post("/translate/cache/clear", tags=["Translation"])
async def clear_model_cache():
    model_manager.clear_cache()
    return {"success": True, "message": "Model cache cleared successfully"}


@router.get("/translate/health", tags=["Health"])
async def health_check():
    cache_info = model_manager.get_cache_info()
    return {
        "status": "healthy",
        "service": "translation",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cached_models": cache_info["cache_size"],
        "supported_languages": len(SUPPORTED_LANGS),
        "nllb_available": True,
    }