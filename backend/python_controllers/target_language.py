import os
import json
import uuid
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import boto3
from botocore.config import Config as BotoCoreConfig
from botocore.exceptions import ClientError
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
# S3 CLIENT (graceful — disabled if creds absent)
# =====================================================
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

_S3_REQUIRED = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
S3_ENABLED   = all(os.getenv(v) for v in _S3_REQUIRED)
s3_client    = None

if S3_ENABLED:
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name           = os.getenv("AWS_REGION"),
            config=BotoCoreConfig(
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )
        s3_client.list_buckets()
        logger.info("translation: S3 client initialised")
    except Exception as e:
        logger.warning(f"translation: S3 init failed ({e}) – S3 upload disabled")
        S3_ENABLED = False
else:
    logger.warning("translation: AWS env vars missing – S3 upload disabled")

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
    "hi",   # Hindi
    "te",   # Telugu
    "ur",   # Urdu
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
    "ta":  "Tamil",
    "hi":  "Hindi",
    "te":  "Telugu",
    "ur":  "Urdu",
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
# P2/P5 FIX: Upgrade to nllb-200-distilled-1.3B for Indic languages (hi/te/ur/ta).
# The 600M model produces grammatically broken Tamil/Hindi (broken verb phrases,
# wrong word order). The 1.3B model has significantly better Indic language quality.
# Falls back to 600M if the 1.3B model is not available (OOM on small GPUs).
NLLB_MODEL         = "facebook/nllb-200-distilled-1.3B"
NLLB_MODEL_FALLBACK = "facebook/nllb-200-distilled-600M"

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
    "hi":  "hin_Deva",   # Hindi — Devanagari script
    "te":  "tel_Telu",   # Telugu — Telugu script  (P5 FIX)
    "ur":  "urd_Arab",   # Urdu   — Nastaliq/Arabic script  (P5 FIX)
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
    "ta":  "tam_Taml",   # Tamil — correct NLLB script tag ✓
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
import re as _re

# P1 FIX: Pre-compiled pattern to strip any leading [...] block that the model
# echoed back.  The simple str.replace used previously failed because NLLB
# sometimes adds Unicode spacing or changes bracket characters during decode.
_HINT_ECHO_RE = _re.compile(r'^\s*\[.*?\]\s*', _re.DOTALL)

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

    # S3 output — optional. When output_bucket is provided the translated JSON
    # is uploaded to S3 and the response includes output_bucket / output_key.
    output_bucket:     Optional[str] = Field(None, max_length=63,
                                             description="S3 bucket to save the translated JSON")
    output_key_prefix: Optional[str] = Field("translations/",
                                             description="S3 key prefix for the output file")

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

    # S3 output — optional
    output_bucket:     Optional[str] = Field(None, max_length=63,
                                             description="S3 bucket to save the translated JSON")
    output_key_prefix: Optional[str] = Field("translations/",
                                             description="S3 key prefix for the output file")

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
    # S3 output fields — None when output_bucket was not requested
    output_bucket: Optional[str] = None
    output_key:    Optional[str] = None
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
    confidence: Optional[float] = None   # P4 FIX: real sequence confidence score
    # S3 output fields — None when output_bucket was not requested
    output_bucket: Optional[str] = None
    output_key:    Optional[str] = None


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

            # FIX: Define a synchronous loader to run in a thread pool,
            # so the heavy HuggingFace from_pretrained() calls never block
            # the async event loop (which caused VS Code / server freezes).
            def _sync_load_model():
                if key in MARIAN_MODELS:
                    model_name = MARIAN_MODELS[key]
                    tok = MarianTokenizer.from_pretrained(
                        model_name, cache_dir=MODEL_CACHE_DIR
                    )
                    mdl = MarianMTModel.from_pretrained(
                        model_name, cache_dir=MODEL_CACHE_DIR
                    )
                    logger.info(f"Loaded Marian model: {model_name}")
                    return mdl, tok, None, "marian"

                elif _needs_nllb(src, tgt):
                    nllb_src = NLLB_LANG_CODES.get(src)
                    nllb_tgt = NLLB_LANG_CODES.get(tgt)
                    if not nllb_src or not nllb_tgt:
                        raise ValueError(
                            f"No NLLB language code for '{src}' or '{tgt}'. "
                            f"Add entries to NLLB_LANG_CODES."
                        )

                    def _load_nllb(model_id: str):
                        tok = AutoTokenizer.from_pretrained(
                            model_id, cache_dir=MODEL_CACHE_DIR, src_lang=nllb_src,
                        )
                        mdl = AutoModelForSeq2SeqLM.from_pretrained(
                            model_id, cache_dir=MODEL_CACHE_DIR
                        )
                        return tok, mdl

                    try:
                        tok, mdl = _load_nllb(NLLB_MODEL)
                        nllb_model_used = NLLB_MODEL
                    except (RuntimeError, Exception) as oom_err:
                        if "out of memory" in str(oom_err).lower() or "cuda" in str(oom_err).lower():
                            logger.warning(
                                f"NLLB 1.3B OOM ({oom_err}); falling back to 600M model"
                            )
                            tok, mdl = _load_nllb(NLLB_MODEL_FALLBACK)
                            nllb_model_used = NLLB_MODEL_FALLBACK
                        else:
                            raise

                    fbs = tok.convert_tokens_to_ids(nllb_tgt)
                    logger.info(
                        f"Loaded NLLB model '{nllb_model_used}' for "
                        f"{src}({nllb_src}) → {tgt}({nllb_tgt}), forced_bos={fbs}"
                    )
                    return mdl, tok, fbs, "nllb"

                else:
                    tok = M2M100Tokenizer.from_pretrained(
                        M2M100_MODEL, cache_dir=MODEL_CACHE_DIR
                    )
                    mdl = M2M100ForConditionalGeneration.from_pretrained(
                        M2M100_MODEL, cache_dir=MODEL_CACHE_DIR
                    )
                    tok.src_lang = src
                    fbs = tok.get_lang_id(tgt)
                    logger.info(f"Loaded M2M100 model for {src}→{tgt}")
                    return mdl, tok, fbs, "m2m100"

            try:
                # FIX: Run the blocking load in a thread pool — never on the event loop
                model, tokenizer, forced_bos, model_type = await asyncio.to_thread(
                    _sync_load_model
                )
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
    "ko":  "[casual energetic conversational Korean 해체, preserve exclamations and emotion] ",
    "ja":  "[casual conversational Japanese ため口, match energy and emotion of source] ",
    "zh":  "[natural conversational Mandarin, match energy and emotion of source] ",
    # P2 FIX: improved Tamil hint — explicitly requests correct verb agreement and
    # natural sentence order, which the 600M model was ignoring.
    "ta":  "[natural fluent Tamil with correct verb agreement and word order, "
           "colloquial spoken register, match energy and emotion of source] ",
    "hi":  "[natural conversational Hindi, correct verb agreement, colloquial Hindustani "
           "register, match energy and emotion of source] ",
    "te":  "[natural conversational Telugu, correct verb agreement and agglutinative "
           "morphology, colloquial register, match energy and emotion of source] ",
    "ur":  "[natural conversational Urdu, correct verb agreement, colloquial register, "
           "match energy and emotion of source] ",
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
) -> Tuple[List[str], str, List[Optional[float]]]:
    """
    Translate a batch of texts using the best available model for the language pair.
    Returns (translations, model_type, confidences).

    Engine routing:
      • Marian  → fast European pairs (NO style hints — breaks SentencePiece tokenizer)
      • NLLB    → Indic, Tamil, Arabic, CJK, 200-language coverage (style hints safe)
      • M2M100  → generic fallback (style hints safe)

    P1 FIX — style hint echo: regex-based [...] prefix stripping replaces the
      broken str.replace approach that failed when NLLB changed spacing/brackets.

    P2 FIX — grammar quality: upgraded to NLLB-1.3B for Indic language pairs.

    P4 FIX — confidence scores: output_scores=True + return_dict_in_generate=True
      lets us compute a real mean-token log-prob per sequence and convert it to a
      probability in [0, 1], replacing the always-null confidence field.
    """
    if not texts:
        return [], "none", []

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
                # P4 FIX: return sequence-level scores so we can compute a
                # real confidence value instead of always returning None.
                # output_scores=True returns per-step token logits;
                # return_dict_in_generate=True wraps them in a named dict.
                output_scores=True,
                return_dict_in_generate=True,
            )
            if forced_bos is not None:
                generate_kwargs["forced_bos_token_id"] = forced_bos

            # FIX: Offload blocking model.generate() to thread pool so the
            # async event loop stays free during inference (prevents server freeze).
            outputs = await asyncio.to_thread(model.generate, **generate_kwargs)

        # Extract sequence ids and scores
        generated     = outputs.sequences
        # sequences_scores: mean log-prob per generated token for each beam-best sequence.
        # Shape: (batch_size,).  Convert log-prob → probability in [0, 1].
        seq_scores_raw = outputs.sequences_scores  # log-probs, typically negative floats
        # Normalise to [0, 1]: e^(score) where score ≈ mean token log-prob.
        # A mean log-prob of 0 means every token was assigned probability 1 (perfect).
        # A mean log-prob of -5 means average token prob ≈ e^-5 ≈ 0.007 (very uncertain).
        # We clamp to [0, 1] to handle edge cases.
        confidences: List[Optional[float]] = []
        if seq_scores_raw is not None:
            try:
                for score in seq_scores_raw.cpu().float().tolist():
                    prob = float(torch.exp(torch.tensor(score)).clamp(0.0, 1.0))
                    confidences.append(round(prob, 4))
            except Exception:
                confidences = [None] * len(generated)
        else:
            confidences = [None] * len(generated)

        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # P1 FIX — Style hint echo removal (see comment above)
        if style_hint:
            translations = [_HINT_ECHO_RE.sub("", t).strip() for t in translations]
            translations = [
                _re.sub(r'^[\[【「『\s]*[^\]】」』\u0B80-\u0BFF]{0,120}[\]】」』]\s*', '', t).strip()
                if t.startswith(('[', '【', '「', '『')) else t
                for t in translations
            ]

        # Safety check: if any translation is empty, log a warning
        empty_count = sum(1 for t in translations if not t.strip())
        if empty_count:
            logger.warning(
                f"translate_batch: {empty_count}/{len(translations)} empty translations "
                f"for {src}→{tgt} using {model_type}. Check model coverage."
            )

        # Pad confidences list to match translations length (safety)
        while len(confidences) < len(translations):
            confidences.append(None)

        return translations, model_type, confidences

    except Exception as e:
        logger.error(f"Translation failed for {src}→{tgt}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _upload_translation_to_s3(
    payload: dict,
    bucket: str,
    key_prefix: str,
    job_id: str,
) -> str:
    """
    Serialise `payload` to JSON and upload to S3.
    Returns the S3 key.  Raises HTTPException(500) on failure.
    """
    if not S3_ENABLED or s3_client is None:
        raise HTTPException(
            status_code=503,
            detail="S3 not configured – set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION",
        )

    key = f"{key_prefix.rstrip('/')}/{job_id}.json"
    local_path = str(TEMP_DIR / f"{job_id}_translation.json")

    try:
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        s3_client.upload_file(
            local_path,
            bucket,
            key,
            ExtraArgs={
                "ContentType":          "application/json; charset=utf-8",
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "job_id":       job_id,
                    "content_type": "translation",
                    "timestamp":    datetime.utcnow().isoformat(),
                },
            },
        )
        logger.info(f"Translation saved to s3://{bucket}/{key}")
        return key

    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"S3 upload failed: {e.response['Error']['Message']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation S3 upload failed: {e}")
    finally:
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except OSError:
            pass


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
        all_translations: List[str]          = []
        all_confidences:  List[Optional[float]] = []
        model_type = None

        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)}")
            # FIX: Call translate_batch directly — no nested asyncio.run() or
            # run_in_executor needed. Blocking work is handled inside load_model
            # via asyncio.to_thread, keeping the event loop free.
            translations, mt, confs = await translate_batch(
                batch,
                request.source_lang,
                request.target_lang,
                request.beam_size,
            )
            all_translations.extend(translations)
            all_confidences.extend(confs)
            if model_type is None:
                model_type = mt

        # Pad confidences if shorter than translations (safety)
        while len(all_confidences) < len(all_translations):
            all_confidences.append(None)

        translated_segments = [
            TranslatedSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                source_text=seg.text,
                translated_text=translated_text,
                confidence=conf,   # P4 FIX: real confidence, not always None
            )
            for seg, translated_text, conf in zip(
                request.segments, all_translations, all_confidences
            )
        ]

        processing_time = time.time() - start_time
        logger.info(
            f"Job {job_id} completed in {processing_time:.2f}s — "
            f"{len(translated_segments)} segments translated via {model_type}"
        )

        # ── S3 upload (when output_bucket is provided) ────────────────────
        output_bucket_out: Optional[str] = None
        output_key_out:    Optional[str] = None

        if request.output_bucket:
            s3_payload = {
                "job_id":          job_id,
                "source_language": request.source_lang,
                "target_language": request.target_lang,
                "model_type":      model_type or "unknown",
                "segments_count":  len(translated_segments),
                "processing_time_seconds": round(processing_time, 2),
                "timestamp":       datetime.utcnow().isoformat(),
                "segments": [
                    {
                        "index":           seg.index,
                        "start":           seg.start,
                        "end":             seg.end,
                        "source_text":     seg.source_text,
                        "translated_text": seg.translated_text,
                        "confidence":      seg.confidence,
                    }
                    for seg in translated_segments
                ],
            }
            output_key_out = _upload_translation_to_s3(
                s3_payload,
                request.output_bucket,
                request.output_key_prefix or "translations/",
                job_id,
            )
            output_bucket_out = request.output_bucket

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
            output_bucket=output_bucket_out,
            output_key=output_key_out,
            message=(
                f"Successfully translated {len(translated_segments)} segments"
                + (f" — saved to s3://{output_bucket_out}/{output_key_out}"
                   if output_key_out else "")
            ),
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
        # P4 FIX: translate_batch now returns (translations, model_type, confidences)
        translations, model_type, confs = await translate_batch(
            [request.text],
            request.source_lang,
            request.target_lang,
            request.beam_size,
        )

        translated_text = translations[0] if translations else ""
        confidence      = confs[0] if confs else None
        processing_time = time.time() - start_time

        logger.info(
            f"Translation completed in {processing_time:.2f}s via {model_type}"
            + (f" | confidence={confidence:.4f}" if confidence is not None else "")
        )

        # ── S3 upload (when output_bucket is provided) ────────────────────
        output_bucket_out: Optional[str] = None
        output_key_out:    Optional[str] = None

        if request.output_bucket:
            s3_payload = {
                "job_id":          str(uuid.uuid4()),
                "source_language": request.source_lang,
                "target_language": request.target_lang,
                "model_type":      model_type,
                "source_text":     request.text,
                "translated_text": translated_text,
                "confidence":      confidence,
                "processing_time_seconds": round(processing_time, 2),
                "timestamp":       datetime.utcnow().isoformat(),
            }
            _job_id = s3_payload["job_id"]
            output_key_out = _upload_translation_to_s3(
                s3_payload,
                request.output_bucket,
                request.output_key_prefix or "translations/",
                _job_id,
            )
            output_bucket_out = request.output_bucket

        return SimpleTranslationResponse(
            success=True,
            source_text=request.text,
            translated_text=translated_text,
            source_language=LANGUAGE_NAMES.get(request.source_lang, request.source_lang),
            target_language=LANGUAGE_NAMES.get(request.target_lang, request.target_lang),
            processing_time_seconds=round(processing_time, 2),
            device=DEVICE,
            model_type=model_type,
            confidence=confidence,
            output_bucket=output_bucket_out,
            output_key=output_key_out,
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