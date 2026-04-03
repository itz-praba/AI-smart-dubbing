"""
segment_validator.py  –  Segment completeness & dialogue diff checker
======================================================================
Solves Problem 7: Missing or extra dialogue.

After translation (Step 3), this validator compares the translated segments
against the original transcript segments and flags:

  1. MISSING segments   — original segment has no corresponding translation
  2. EXTRA segments     — translated output has segments not in the original
  3. EMPTY translations — translated_text is blank or whitespace-only
  4. TRUNCATED text     — translated text is suspiciously short vs source
                          (ratio-based heuristic, calibrated per language pair)
  5. COUNT MISMATCH     — total segment counts differ
  6. TIMING DRIFT       — cumulative timing gap between source and translated
                          segments exceeds a safe threshold

On finding issues the validator can:
  - WARN  — log and include in the response, continue pipeline
  - BLOCK — raise HTTPException(422) to halt the pipeline (configurable)

A repair step can auto-fill empty/missing translations by falling back to
the source text (with a flag so the caller knows it's a fallback).

Dependencies: standard library only (no ML models needed).
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["Segment Validation"])


# ── Severity levels ───────────────────────────────────────────────────────────
SEVERITY_WARN  = "warn"
SEVERITY_ERROR = "error"

# ── Language-pair translation length ratios ───────────────────────────────────
# Languages expand or compress relative to English source.
# Ratios: (min_acceptable, max_acceptable) relative to source text LENGTH.
# These are conservative bounds; genuine translations rarely go outside them.
LANG_LENGTH_RATIOS: Dict[str, Tuple[float, float]] = {
    # target_lang: (min_ratio, max_ratio) — ratio = len(translated) / len(source)
    "de":  (0.80, 1.60),  # German tends longer
    "fr":  (0.75, 1.50),
    "es":  (0.70, 1.50),
    "it":  (0.70, 1.50),
    "pt":  (0.70, 1.50),
    "ru":  (0.60, 1.80),  # Cyrillic can look shorter in bytes but encodes more
    "zh":  (0.25, 0.80),  # Chinese is much denser per character
    "ja":  (0.25, 0.85),
    "ko":  (0.35, 0.90),
    "ar":  (0.50, 1.60),
    "hi":  (0.55, 1.60),  # Devanagari — moderate expansion
    "te":  (0.60, 1.70),  # Telugu — agglutinative, can expand
    "ur":  (0.50, 1.60),  # Urdu — similar to Hindi
    "ta":  (0.60, 1.70),  # Tamil — agglutinative
    "tgl": (0.70, 1.40),  # Tanglish — close to English
    "nl":  (0.75, 1.50),
    "pl":  (0.75, 1.60),
    "tr":  (0.65, 1.55),
    "vi":  (0.75, 1.60),
    "th":  (0.30, 0.90),
    "uk":  (0.60, 1.80),
    "cs":  (0.70, 1.60),
    "sv":  (0.70, 1.50),
    "da":  (0.70, 1.50),
    "no":  (0.70, 1.50),
    "fi":  (0.65, 1.55),
    "el":  (0.75, 1.60),
    "he":  (0.40, 1.20),
}
DEFAULT_LENGTH_RATIO = (0.30, 2.20)  # very wide fallback for unknown pairs

# ── Timing drift threshold ────────────────────────────────────────────────────
MAX_TIMING_DRIFT_S = 0.5   # allow up to 0.5s cumulative drift per segment
MAX_TOTAL_DRIFT_S  = 5.0   # total across entire sequence


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SourceSegment(BaseModel):
    index: int  = Field(..., ge=0)
    start: float
    end:   float
    text:  str


class TranslatedSegment(BaseModel):
    index:           int   = Field(..., ge=0)
    start:           float
    end:             float
    source_text:     str
    translated_text: str


class ValidationIssue(BaseModel):
    issue_type:  str           # missing | extra | empty | truncated | count_mismatch | timing_drift
    severity:    str           # warn | error
    segment_index: Optional[int] = None
    detail:      str
    auto_repaired: bool = False
    repair_value:  Optional[str] = None


class ValidationRequest(BaseModel):
    source_segments:     List[SourceSegment]     = Field(..., min_items=1)
    translated_segments: List[TranslatedSegment] = Field(..., min_items=0)
    source_lang:         str = Field(..., min_length=2, max_length=5)
    target_lang:         str = Field(..., min_length=2, max_length=5)

    # Behaviour controls
    block_on_missing:    bool  = Field(True,  description="Raise 422 if any segments are missing")
    block_on_empty:      bool  = Field(True,  description="Raise 422 if any translated text is empty")
    block_on_count_mismatch: bool = Field(False, description="Raise 422 on count mismatch (lenient default)")
    auto_repair_empty:   bool  = Field(True,  description="Fill empty translations with source text (flagged)")
    truncation_check:    bool  = Field(True,  description="Flag suspiciously short translations")
    timing_check:        bool  = Field(True,  description="Flag timing drift between source and translated")


class ValidationResponse(BaseModel):
    success:             bool
    job_id:              str
    is_valid:            bool
    issues:              List[ValidationIssue]
    issue_count:         int
    error_count:         int
    warn_count:          int
    repaired_segments:   int
    validated_segments:  List[TranslatedSegment]  # may contain auto-repaired segments
    source_count:        int
    translated_count:    int
    processing_time_ms:  float
    message:             str


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _length_ratio_ok(source_text: str, translated_text: str, target_lang: str) -> bool:
    """Return True if the translated length is within expected bounds for the language pair."""
    if not source_text:
        return True
    ratio = len(translated_text) / max(1, len(source_text))
    min_r, max_r = LANG_LENGTH_RATIOS.get(target_lang, DEFAULT_LENGTH_RATIO)
    return min_r <= ratio <= max_r


def validate_segments(request: ValidationRequest) -> ValidationResponse:
    """
    Core validation: diff source vs translated, return issues and repaired segments.
    """
    job_id = str(uuid.uuid4())
    t0     = datetime.utcnow()

    issues:           List[ValidationIssue]    = []
    repaired_count:   int                      = 0
    validated:        List[TranslatedSegment]  = list(request.translated_segments)

    src_map  = {s.index: s for s in request.source_segments}
    tgt_map  = {t.index: t for t in request.translated_segments}

    src_indices = set(src_map.keys())
    tgt_indices = set(tgt_map.keys())

    # ── 1. Count mismatch ────────────────────────────────────────────────────
    if len(request.source_segments) != len(request.translated_segments):
        issues.append(ValidationIssue(
            issue_type    = "count_mismatch",
            severity      = SEVERITY_ERROR if request.block_on_count_mismatch else SEVERITY_WARN,
            detail        = (
                f"Source has {len(request.source_segments)} segments but "
                f"translation has {len(request.translated_segments)}. "
                f"Difference: {len(request.source_segments) - len(request.translated_segments):+d}"
            ),
        ))
        logger.warning(
            f"[validator] Count mismatch: {len(request.source_segments)} src vs "
            f"{len(request.translated_segments)} tgt"
        )

    # ── 2. Missing segments (in source but not in translation) ───────────────
    missing_indices = src_indices - tgt_indices
    for idx in sorted(missing_indices):
        src = src_map[idx]
        repair_text = src.text if request.auto_repair_empty else None
        repaired    = repair_text is not None
        if repaired:
            validated.append(TranslatedSegment(
                index           = idx,
                start           = src.start,
                end             = src.end,
                source_text     = src.text,
                translated_text = src.text,   # fallback: use source text verbatim
            ))
            repaired_count += 1
            logger.warning(
                f"[validator] Segment {idx} missing from translation — "
                f"auto-filled with source text"
            )
        issues.append(ValidationIssue(
            issue_type    = "missing",
            severity      = SEVERITY_ERROR,
            segment_index = idx,
            detail        = (
                f"Segment {idx} ({src.start:.2f}s–{src.end:.2f}s) "
                f"'{src.text[:60]}...' has no translation"
            ),
            auto_repaired = repaired,
            repair_value  = repair_text,
        ))

    # ── 3. Extra segments (in translation but not in source) ─────────────────
    extra_indices = tgt_indices - src_indices
    for idx in sorted(extra_indices):
        tgt = tgt_map[idx]
        issues.append(ValidationIssue(
            issue_type    = "extra",
            severity      = SEVERITY_WARN,
            segment_index = idx,
            detail        = (
                f"Segment {idx} exists in translation but not in source transcript. "
                f"Text: '{tgt.translated_text[:60]}'"
            ),
        ))
        logger.warning(f"[validator] Extra translated segment {idx} not in source")

    # ── 4. Empty / whitespace translations ───────────────────────────────────
    for seg in validated:
        if not seg.translated_text or not seg.translated_text.strip():
            src = src_map.get(seg.index)
            repair_text = src.text if (request.auto_repair_empty and src) else None
            repaired    = repair_text is not None
            if repaired:
                # Mutate the segment in-place on the validated list
                for i, v in enumerate(validated):
                    if v.index == seg.index:
                        validated[i] = v.copy(update={"translated_text": repair_text})
                        break
                repaired_count += 1
                logger.warning(
                    f"[validator] Segment {seg.index} has empty translation — "
                    f"auto-filled with source text"
                )
            issues.append(ValidationIssue(
                issue_type    = "empty",
                severity      = SEVERITY_ERROR,
                segment_index = seg.index,
                detail        = f"Segment {seg.index} has an empty or whitespace-only translation",
                auto_repaired = repaired,
                repair_value  = repair_text,
            ))

    # ── 5. Truncation check ───────────────────────────────────────────────────
    if request.truncation_check:
        for seg in tgt_map.values():
            src = src_map.get(seg.index)
            if not src or not seg.translated_text.strip():
                continue
            if not _length_ratio_ok(src.text, seg.translated_text, request.target_lang):
                ratio = len(seg.translated_text) / max(1, len(src.text))
                min_r, max_r = LANG_LENGTH_RATIOS.get(request.target_lang, DEFAULT_LENGTH_RATIO)
                issues.append(ValidationIssue(
                    issue_type    = "truncated",
                    severity      = SEVERITY_WARN,
                    segment_index = seg.index,
                    detail        = (
                        f"Segment {seg.index}: translated length ratio {ratio:.2f} "
                        f"is outside expected range [{min_r:.2f}, {max_r:.2f}] "
                        f"for {request.source_lang}→{request.target_lang}. "
                        f"Source: '{src.text[:50]}' → Translated: '{seg.translated_text[:50]}'"
                    ),
                ))
                logger.warning(
                    f"[validator] Segment {seg.index} length ratio {ratio:.2f} "
                    f"outside [{min_r:.2f}, {max_r:.2f}]"
                )

    # ── 6. Timing drift check ─────────────────────────────────────────────────
    if request.timing_check:
        cumulative_drift = 0.0
        for src_seg in sorted(request.source_segments, key=lambda s: s.index):
            tgt_seg = tgt_map.get(src_seg.index)
            if tgt_seg is None:
                continue
            start_drift = abs(tgt_seg.start - src_seg.start)
            end_drift   = abs(tgt_seg.end   - src_seg.end)
            seg_drift   = max(start_drift, end_drift)
            cumulative_drift += seg_drift

            if seg_drift > MAX_TIMING_DRIFT_S:
                issues.append(ValidationIssue(
                    issue_type    = "timing_drift",
                    severity      = SEVERITY_WARN,
                    segment_index = src_seg.index,
                    detail        = (
                        f"Segment {src_seg.index} timing drift {seg_drift:.3f}s "
                        f"(src: {src_seg.start:.2f}–{src_seg.end:.2f}s, "
                        f"tgt: {tgt_seg.start:.2f}–{tgt_seg.end:.2f}s)"
                    ),
                ))

        if cumulative_drift > MAX_TOTAL_DRIFT_S:
            issues.append(ValidationIssue(
                issue_type = "timing_drift",
                severity   = SEVERITY_WARN,
                detail     = (
                    f"Total cumulative timing drift {cumulative_drift:.2f}s "
                    f"exceeds threshold {MAX_TOTAL_DRIFT_S}s across all segments. "
                    f"Lip-sync may be affected."
                ),
            ))

    # ── Sort validated segments by index ─────────────────────────────────────
    validated.sort(key=lambda s: s.index)

    # ── Determine blocking errors ─────────────────────────────────────────────
    error_issues = [i for i in issues if i.severity == SEVERITY_ERROR]
    warn_issues  = [i for i in issues if i.severity == SEVERITY_WARN]

    blocking_errors: List[str] = []
    for issue in error_issues:
        if issue.issue_type == "missing"        and request.block_on_missing     and not issue.auto_repaired:
            blocking_errors.append(issue.detail)
        if issue.issue_type == "empty"          and request.block_on_empty       and not issue.auto_repaired:
            blocking_errors.append(issue.detail)
        if issue.issue_type == "count_mismatch" and request.block_on_count_mismatch:
            blocking_errors.append(issue.detail)

    elapsed_ms = (datetime.utcnow() - t0).total_seconds() * 1000

    response = ValidationResponse(
        success            = True,
        job_id             = job_id,
        is_valid           = len(blocking_errors) == 0,
        issues             = issues,
        issue_count        = len(issues),
        error_count        = len(error_issues),
        warn_count         = len(warn_issues),
        repaired_segments  = repaired_count,
        validated_segments = validated,
        source_count       = len(request.source_segments),
        translated_count   = len(request.translated_segments),
        processing_time_ms = round(elapsed_ms, 1),
        message=(
            f"Validation {'PASSED' if not blocking_errors else 'FAILED'}: "
            f"{len(issues)} issue(s) — "
            f"{len(error_issues)} error(s), {len(warn_issues)} warning(s). "
            f"{repaired_count} segment(s) auto-repaired."
        ),
    )

    if blocking_errors:
        detail_str = "\n".join(blocking_errors[:5])
        if len(blocking_errors) > 5:
            detail_str += f"\n...and {len(blocking_errors)-5} more"
        raise HTTPException(
            status_code=422,
            detail=(
                f"Segment validation failed — pipeline halted to prevent bad output.\n"
                f"{detail_str}"
            ),
        )

    if issues:
        logger.warning(
            f"[validator] {len(issues)} issue(s): "
            f"{len(error_issues)} error(s) {len(warn_issues)} warning(s) "
            f"| {repaired_count} auto-repaired"
        )
    else:
        logger.info("[validator] All segments passed validation ✓")

    return response


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/validate-segments",
    response_model=ValidationResponse,
    summary="Validate translated segments against source transcript",
    description=(
        "Diffs source transcript segments against the translation output. "
        "Detects missing, extra, empty, and truncated translations. "
        "Optionally auto-repairs empty segments using source text. "
        "Fixes Problem 7: missing or extra dialogue."
    ),
)
async def validate_segments_endpoint(request: ValidationRequest) -> ValidationResponse:
    return validate_segments(request)


@router.post(
    "/validate-segments/strict",
    response_model=ValidationResponse,
    summary="Strict validation — blocks pipeline on any error",
    description="Same as /validate-segments but blocks on count mismatch and does not auto-repair.",
)
async def validate_segments_strict(request: ValidationRequest) -> ValidationResponse:
    strict = request.copy(update={
        "block_on_missing":       True,
        "block_on_empty":         True,
        "block_on_count_mismatch": True,
        "auto_repair_empty":      False,
    })
    return validate_segments(strict)


@router.get("/validate-segments/health", tags=["Health"])
async def health_check():
    return {
        "status":  "healthy",
        "service": "segment-validator",
        "checks":  [
            "missing_segments",
            "extra_segments",
            "empty_translations",
            "truncation_ratio",
            "timing_drift",
            "count_mismatch",
        ],
        "supported_target_langs": list(LANG_LENGTH_RATIOS.keys()),
    }
