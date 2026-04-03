"""
cultural_adapter.py  –  LLM-powered cultural & idiomatic post-translation
==========================================================================
Solves Problem 3 (translation accuracy — idioms/jokes/cultural refs) and
Problem 10 (cultural adaptation — unnatural expressions, honorifics, regional refs).

The base NLLB/M2M100 translation is word-for-word accurate but culturally
flat. This module runs a second-pass LLM rewrite that:

  1. Adapts idioms and proverbs to target-culture equivalents.
  2. Corrects honorific register (formal/informal, kinship terms in Indic langs).
  3. Localises cultural references (food, places, sports, units of measure).
  4. Preserves dubbing constraints: approximate syllable length, lip-sync density,
     emotional tone, character voice (formal / casual / child / elder).
  5. Flags segments that need human review (jokes, wordplay, brand names).

Architecture:
  • Uses the Anthropic Messages API (claude-sonnet-4-20250514) via httpx.
  • API key read from ANTHROPIC_API_KEY env var.
  • Batch mode: sends up to 20 segments per request to amortise API round-trips.
  • Graceful fallback: if API unavailable, returns original translation unchanged.

Dependencies:
    pip install httpx fastapi pydantic
"""

import os
import json
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["Cultural Adaptation"])

# ── Anthropic API ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL  = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL    = "claude-sonnet-4-20250514"
ANTHROPIC_MAX_TOKENS = 4096
API_TIMEOUT_S      = 120
MAX_RETRIES        = 3
BATCH_SIZE         = 20   # segments per LLM call

# ── Per-language cultural notes injected into the system prompt ───────────────
CULTURAL_NOTES: Dict[str, str] = {
    "hi": (
        "Hindi cultural notes for dubbing:\n"
        "- Use appropriate honorifics: 'aap' (formal/elder), 'tum' (peer/semi-formal), 'tu' (intimate/rude).\n"
        "- Kinship terms are mandatory: didi (elder sister), bhaiya (elder brother), chacha/mama (uncle), "
        "  dadi/nani (grandmother). Use contextually.\n"
        "- Replace Western food references with Indian equivalents where natural "
        "  (e.g. 'sandwich' → 'parantha' only if the scene is domestic/Indian-set).\n"
        "- Cricket is universally understood; other sport references may need adaptation.\n"
        "- Avoid direct English loanword calques when a natural Hindi word exists.\n"
        "- Colloquial Hindustani is preferred for drama; formal Shuddh Hindi for documentaries.\n"
        "- Common fillers: 'yaar', 'arre', 'accha', 'theek hai' — use contextually.\n"
        "- Numbers in Lakh/Crore for large figures; 'kilometre' not 'mile'."
    ),
    "te": (
        "Telugu cultural notes for dubbing:\n"
        "- Honorifics: '-gaaru' suffix for respect (Ramesh-gaaru), 'meeru' (formal you), 'nuvvu' (informal).\n"
        "- Kinship: anna (elder brother), akka (elder sister), baava (brother-in-law).\n"
        "- Use 'babu' as a warm form of address for boys/men.\n"
        "- Rice-based food culture; adapt food references accordingly.\n"
        "- Telugu cinema (Tollywood) references are well understood.\n"
        "- Colloquial Telugu mixes English freely; tech terms can stay in English.\n"
        "- Regional variants: Rayalaseema is more formal than coastal Andhra; "
        "  use standard Telugu unless scene is region-specific.\n"
        "- Common exclamations: 'arey', 'baagundi' (good), 'chala' (very)."
    ),
    "ur": (
        "Urdu cultural notes for dubbing:\n"
        "- Urdu has strong Persian/Arabic influence; use authentic Urdu vocabulary "
        "  rather than direct Hindi cognates where register demands.\n"
        "- Honorifics: 'aap' (formal), 'tum' (peer), 'tu' (intimate). "
        "  'Janab' / 'Sahib' for formal address.\n"
        "- Islamic references (Inshallah, MashAllah, Alhamdulillah) are natural and expected.\n"
        "- Pakistan-set content: use Pakistani institutional names, currency (PKR), "
        "  cities (Lahore, Karachi, Islamabad) correctly.\n"
        "- India-set content: Urdu dubbing for Indian productions should preserve "
        "  Indian cultural references but in Urdu register.\n"
        "- Poetry and literary allusions are culturally valued; preserve them.\n"
        "- Avoid Romanised/transliterated slang in formal registers."
    ),
    "ta": (
        "Tamil cultural notes for dubbing:\n"
        "- Formal Tamil (centamil) vs Colloquial Tamil (kodum Tamil) differ significantly. "
        "  Use colloquial for drama, formal for documentary.\n"
        "- Honorifics: '-nga' suffix for respect; 'ungalukku' (formal you), 'unakku' (informal).\n"
        "- Kinship: anna/akka (elder bro/sis), thambi/thangai (younger bro/sis), "
        "  appa/amma (father/mother as address terms, even to non-parents in Tamil Nadu).\n"
        "- Kollywood (Tamil cinema) cultural references are widely understood.\n"
        "- Food: sambar, idli, dosa are everyday; avoid substituting unnecessarily.\n"
        "- Use Tamil months/seasons for nature references in rural scenes."
    ),
    "tgl": (
        "Tanglish (romanised Tamil-English) notes:\n"
        "- Write Tamil words in Latin script. English words stay as-is.\n"
        "- Natural code-switching: the more emotional/intimate the line, the more Tamil.\n"
        "- Maintain colloquial flow; avoid overly formal or bookish Tamil words."
    ),
    "ko": (
        "Korean cultural notes for dubbing:\n"
        "- Speech levels are critical: 합쇼체 (formal), 해요체 (polite casual), 해체 (informal/intimate).\n"
        "- Age-based hierarchy: always use appropriate speech level based on character relationship.\n"
        "- Honorific address: '씨' (Mr/Ms), '님' (respectful suffix), '선생님' (teacher/doctor).\n"
        "- Food references: adapt Western food to Korean where natural (빵 for bread, 밥 for rice meal).\n"
        "- K-drama tropes are well understood; avoid over-explaining them."
    ),
    "ja": (
        "Japanese cultural notes for dubbing:\n"
        "- Keigo (敬語) levels must match character relationships precisely.\n"
        "- Sentence-final particles (ね、よ、な、わ、ぞ) carry emotional weight — match the original.\n"
        "- Do not drop subjects in formal contexts even though Japanese normally omits them.\n"
        "- Loanwords (カタカナ) are natural for foreign concepts; use them freely."
    ),
    "zh": (
        "Chinese (Simplified) cultural notes for dubbing:\n"
        "- Measure words (量词) must be correct; generic '个' is always safe but specific ones are better.\n"
        "- Regional variants: Mainland uses Simplified; avoid Traditional forms.\n"
        "- Chengyu (成语) — 4-character idioms — are stylistically valued; replace Western idioms with them.\n"
        "- Politeness: '您' vs '你' for formal/informal; critical in workplace/elder scenes."
    ),
    "ar": (
        "Arabic cultural notes for dubbing:\n"
        "- Modern Standard Arabic (MSA/Fus'ha) for documentaries and formal content.\n"
        "- Egyptian/Levantine/Gulf dialects for drama if region-specific. Default to MSA for pan-Arab.\n"
        "- Islamic phrases are culturally embedded; include naturally.\n"
        "- Gender agreement (grammatical) must be strictly maintained.\n"
        "- Numbers: Arabic-Indic numerals (٠١٢...) are standard in text; Latin in tech contexts."
    ),
}

# Fallback: generic instructions for languages without specific notes
GENERIC_CULTURAL_NOTE = (
    "Adapt idioms, proverbs, and cultural references to natural equivalents in the target language. "
    "Preserve the emotional register and approximate spoken duration. "
    "Ensure the result sounds like natural, colloquial speech for dubbing."
)

# ── Character limits per dubbing constraint ───────────────────────────────────
# Translated text for dubbing should not exceed the source length by more than
# this factor — otherwise lip-sync time-stretch will hit its ceiling.
MAX_LENGTH_EXPANSION = 1.35  # 35% longer than source is the hard ceiling


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentForAdaptation(BaseModel):
    index:           int   = Field(..., ge=0)
    start:           float
    end:             float
    source_text:     str   = Field(..., min_length=1)
    translated_text: str   = Field(..., min_length=1)
    speaker:         Optional[str] = None
    emotion_hint:    Optional[str] = None  # injected by prosody_transfer if available


class AdaptedSegment(BaseModel):
    index:             int
    start:             float
    end:               float
    source_text:       str
    original_translation: str
    adapted_text:      str
    was_changed:       bool
    change_notes:      Optional[str] = None
    needs_review:      bool = False
    review_reason:     Optional[str] = None


class CulturalAdaptationRequest(BaseModel):
    segments:            List[SegmentForAdaptation] = Field(..., min_items=1, max_items=500)
    source_lang:         str = Field(..., min_length=2, max_length=5)
    target_lang:         str = Field(..., min_length=2, max_length=5)
    content_type:        str = Field("drama",
                                     description="drama | documentary | comedy | children | news")
    character_profile:   Optional[str] = Field(None,
                                     description="Brief character description e.g. 'elderly grandmother, warm, formal'")
    preserve_length:     bool  = Field(True,
                                     description="Keep adapted text within 35% of source length for lip-sync")
    flag_for_review:     bool  = Field(True,
                                     description="Flag segments with jokes, wordplay, or brand names")
    fallback_on_error:   bool  = Field(True,
                                     description="If LLM call fails, return original translation unchanged")

    @validator("content_type")
    def validate_content_type(cls, v):
        allowed = {"drama", "documentary", "comedy", "children", "news", "animation", "thriller"}
        if v not in allowed:
            raise ValueError(f"content_type must be one of {allowed}")
        return v


class CulturalAdaptationResponse(BaseModel):
    success:                bool
    job_id:                 str
    source_lang:            str
    target_lang:            str
    segments_adapted:       int
    segments_changed:       int
    segments_flagged:       int
    segments_failed:        int
    adapted_segments:       List[AdaptedSegment]
    processing_time_seconds: float
    llm_calls_made:         int
    message:                str


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(
    source_lang: str, target_lang: str, content_type: str,
    character_profile: Optional[str],
) -> str:
    cultural_note = CULTURAL_NOTES.get(target_lang, GENERIC_CULTURAL_NOTE)
    char_section  = (
        f"\nCharacter profile for this speaker: {character_profile}\n"
        if character_profile else ""
    )
    return f"""You are a professional dubbing adaptation specialist.
Your task: take machine-translated subtitle segments and rewrite them to sound
natural, culturally appropriate, and emotionally accurate for {target_lang} dubbing.

Source language: {source_lang}
Target language: {target_lang}
Content type: {content_type}
{char_section}
CULTURAL GUIDANCE:
{cultural_note}

DUBBING CONSTRAINTS (critical):
- The adapted text will be spoken aloud by a voice actor; it must match the approximate
  speaking duration of the original (source_text length ±35%).
- Preserve emotional register exactly — sad stays sad, angry stays angry.
- Do NOT add or remove lines. One adapted segment per input segment.
- If the original translation is already natural and culturally appropriate, return it unchanged.
- Mark segments needing human review with needs_review=true (jokes, puns, brand names,
  culturally-untranslatable references).

OUTPUT FORMAT (respond with valid JSON only, no markdown fences):
{{
  "adapted_segments": [
    {{
      "index": <int>,
      "adapted_text": "<string>",
      "was_changed": <bool>,
      "change_notes": "<brief note on what was changed, or null>",
      "needs_review": <bool>,
      "review_reason": "<reason if needs_review, else null>"
    }},
    ...
  ]
}}"""


def _build_user_message(segments: List[SegmentForAdaptation]) -> str:
    seg_list = []
    for s in segments:
        seg_list.append({
            "index":           s.index,
            "source_text":     s.source_text,
            "translated_text": s.translated_text,
            "duration_s":      round(s.end - s.start, 2),
            "emotion_hint":    s.emotion_hint or "neutral",
        })
    return (
        "Please adapt the following translated segments for natural dubbing:\n\n"
        + json.dumps(seg_list, ensure_ascii=False, indent=2)
    )


async def _call_llm(
    system_prompt: str,
    user_message:  str,
    retry:         int = 0,
) -> Optional[Dict[str, Any]]:
    """Call the Anthropic Messages API. Returns parsed JSON or None on failure."""
    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — cultural adaptation skipped")
        return None

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model":      ANTHROPIC_MODEL,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_message}],
    }

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT_S) as client:
            response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)

        if response.status_code == 429 and retry < MAX_RETRIES:
            wait = 2 ** retry
            logger.warning(f"LLM rate-limited; retrying in {wait}s (attempt {retry+1})")
            await asyncio.sleep(wait)
            return await _call_llm(system_prompt, user_message, retry + 1)

        if response.status_code != 200:
            logger.error(f"LLM API error {response.status_code}: {response.text[:200]}")
            return None

        data     = response.json()
        raw_text = data["content"][0]["text"].strip()

        # Strip any accidental markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        return json.loads(raw_text)

    except json.JSONDecodeError as e:
        logger.error(f"LLM response not valid JSON: {e}")
        return None
    except Exception as e:
        if retry < MAX_RETRIES:
            wait = 2 ** retry
            logger.warning(f"LLM call failed ({e}); retrying in {wait}s")
            await asyncio.sleep(wait)
            return await _call_llm(system_prompt, user_message, retry + 1)
        logger.error(f"LLM call failed after {MAX_RETRIES} retries: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _enforce_length(adapted: str, source: str, original: str) -> str:
    """
    If the adapted text is > MAX_LENGTH_EXPANSION × source length,
    truncate at a sentence boundary or fall back to the original translation.
    """
    max_len = int(len(source) * MAX_LENGTH_EXPANSION)
    if len(adapted) <= max_len:
        return adapted
    # Try to truncate at the last sentence boundary within max_len
    truncated = adapted[:max_len]
    for punct in ("।", "。", "！", "？", ".", "!", "?"):
        last = truncated.rfind(punct)
        if last > max_len // 2:
            return truncated[:last + 1].strip()
    # No good boundary — use original translation (already length-checked by NLLB)
    logger.warning(
        f"Adapted text too long ({len(adapted)} > {max_len}); reverting to original translation"
    )
    return original


async def _adapt_batch(
    segments:           List[SegmentForAdaptation],
    system_prompt:      str,
    preserve_length:    bool,
    fallback_on_error:  bool,
    llm_calls:          List[int],
) -> List[AdaptedSegment]:
    """
    Send one batch to the LLM and parse the results.
    Returns a list of AdaptedSegment — one per input segment.
    """
    user_msg = _build_user_message(segments)
    llm_calls[0] += 1
    result   = await _call_llm(system_prompt, user_msg)

    adapted_out: List[AdaptedSegment] = []

    if result is None:
        # LLM unavailable — return all segments unchanged
        if not fallback_on_error:
            raise HTTPException(502, "Cultural adaptation LLM unavailable")
        for seg in segments:
            adapted_out.append(AdaptedSegment(
                index                = seg.index,
                start                = seg.start,
                end                  = seg.end,
                source_text          = seg.source_text,
                original_translation = seg.translated_text,
                adapted_text         = seg.translated_text,
                was_changed          = False,
                change_notes         = "LLM unavailable — original translation used",
            ))
        return adapted_out

    # Build a lookup for the LLM's response by index
    llm_map: Dict[int, Dict[str, Any]] = {}
    for item in result.get("adapted_segments", []):
        idx = item.get("index")
        if idx is not None:
            llm_map[idx] = item

    for seg in segments:
        llm_item = llm_map.get(seg.index)
        if llm_item is None:
            logger.warning(f"[cultural] LLM did not return result for segment {seg.index}")
            adapted_out.append(AdaptedSegment(
                index                = seg.index,
                start                = seg.start,
                end                  = seg.end,
                source_text          = seg.source_text,
                original_translation = seg.translated_text,
                adapted_text         = seg.translated_text,
                was_changed          = False,
                change_notes         = "LLM response missing for this segment",
            ))
            continue

        adapted_raw = llm_item.get("adapted_text", seg.translated_text).strip()

        # Enforce length constraint
        if preserve_length:
            adapted_raw = _enforce_length(adapted_raw, seg.source_text, seg.translated_text)

        was_changed = adapted_raw != seg.translated_text

        adapted_out.append(AdaptedSegment(
            index                = seg.index,
            start                = seg.start,
            end                  = seg.end,
            source_text          = seg.source_text,
            original_translation = seg.translated_text,
            adapted_text         = adapted_raw,
            was_changed          = was_changed,
            change_notes         = llm_item.get("change_notes"),
            needs_review         = bool(llm_item.get("needs_review", False)),
            review_reason        = llm_item.get("review_reason"),
        ))

    return adapted_out


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/cultural-adapt",
    response_model=CulturalAdaptationResponse,
    summary="LLM-powered cultural and idiomatic adaptation of translated text",
    description=(
        "Second-pass post-processor after NLLB/M2M100 translation. "
        "Rewrites idioms, honorifics, cultural references, and register "
        "to sound natural in the target language. "
        "Fixes Problems 3 (translation accuracy) and 10 (cultural adaptation)."
    ),
)
async def cultural_adapt(request: CulturalAdaptationRequest) -> CulturalAdaptationResponse:
    job_id = str(uuid.uuid4())
    t0     = datetime.utcnow()

    system_prompt = _build_system_prompt(
        request.source_lang, request.target_lang,
        request.content_type, request.character_profile,
    )

    # Split into batches of BATCH_SIZE
    batches: List[List[SegmentForAdaptation]] = [
        request.segments[i:i + BATCH_SIZE]
        for i in range(0, len(request.segments), BATCH_SIZE)
    ]

    all_adapted:    List[AdaptedSegment] = []
    failed_count:   int = 0
    llm_calls:      List[int] = [0]  # mutable counter

    for batch in batches:
        try:
            adapted_batch = await _adapt_batch(
                batch, system_prompt,
                request.preserve_length,
                request.fallback_on_error,
                llm_calls,
            )
            all_adapted.extend(adapted_batch)
        except HTTPException:
            raise
        except Exception as e:
            failed_count += len(batch)
            logger.error(f"[cultural] Batch failed: {e}")
            if not request.fallback_on_error:
                raise HTTPException(500, f"Cultural adaptation batch failed: {e}")
            # Fallback: use original translations for this batch
            for seg in batch:
                all_adapted.append(AdaptedSegment(
                    index                = seg.index,
                    start                = seg.start,
                    end                  = seg.end,
                    source_text          = seg.source_text,
                    original_translation = seg.translated_text,
                    adapted_text         = seg.translated_text,
                    was_changed          = False,
                    change_notes         = f"Batch error — original translation used: {e}",
                ))

    # Sort by index
    all_adapted.sort(key=lambda s: s.index)

    changed_count  = sum(1 for s in all_adapted if s.was_changed)
    flagged_count  = sum(1 for s in all_adapted if s.needs_review)
    elapsed        = (datetime.utcnow() - t0).total_seconds()

    logger.info(
        f"[cultural] Job {job_id}: {len(all_adapted)} segments, "
        f"{changed_count} changed, {flagged_count} flagged, "
        f"{llm_calls[0]} LLM calls, {elapsed:.1f}s"
    )

    return CulturalAdaptationResponse(
        success                  = True,
        job_id                   = job_id,
        source_lang              = request.source_lang,
        target_lang              = request.target_lang,
        segments_adapted         = len(all_adapted) - failed_count,
        segments_changed         = changed_count,
        segments_flagged         = flagged_count,
        segments_failed          = failed_count,
        adapted_segments         = all_adapted,
        processing_time_seconds  = round(elapsed, 2),
        llm_calls_made           = llm_calls[0],
        message=(
            f"Cultural adaptation: {changed_count}/{len(all_adapted)} segments rewritten, "
            f"{flagged_count} flagged for review, "
            f"{llm_calls[0]} LLM calls in {elapsed:.1f}s"
        ),
    )


@router.get("/cultural-adapt/health", tags=["Health"])
async def health_check():
    has_key = bool(ANTHROPIC_API_KEY)
    return {
        "status":              "healthy" if has_key else "degraded",
        "service":             "cultural-adapter",
        "llm_model":           ANTHROPIC_MODEL,
        "api_key_configured":  has_key,
        "batch_size":          BATCH_SIZE,
        "supported_langs":     list(CULTURAL_NOTES.keys()) + ["(all others with generic guidance)"],
        "content_types":       ["drama", "documentary", "comedy", "children", "news", "animation", "thriller"],
        "fallback_on_error":   True,
    }
