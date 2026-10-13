"""Fallback correction path (BE-04): guarantees the pipeline always returns usable
text, even when Gemini is down, rate-limited, or misbehaving (FR-5).

This is the ONLY function bot handlers should call for correction - it wraps
gemini_client.correct_text() and never lets a correction failure propagate as an
exception. Worst case, the user gets the raw ASR transcription instead of a
corrected one; they never get a hard failure or no response at all.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from .gemini_client import GeminiCorrectionError, correct_text

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """What the caller actually needs: the text to show the user, plus enough
    metadata to log/monitor without changing what the user sees."""
    text: str
    used_fallback: bool
    error_detail: Optional[str] = None


def correct_with_fallback(raw_text: str, **kwargs) -> CorrectionResult:
    """Try Gemini correction; on ANY correction failure, fall back to the raw
    ASR text rather than raising. `**kwargs` pass straight through to
    correct_text() (model_name, timeout, max_retries, api_key).

    Always returns a CorrectionResult - never raises for a correction failure.
    A genuinely unexpected exception (not GeminiCorrectionError) still propagates,
    since that likely signals a real bug, not an expected external-service failure.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return CorrectionResult(text="", used_fallback=False)

    try:
        corrected = correct_text(raw_text, **kwargs)
        return CorrectionResult(text=corrected, used_fallback=False)
    except GeminiCorrectionError as e:
        logger.warning(
            "Gemini correction failed, falling back to raw ASR text: %s", e.detail
        )
        return CorrectionResult(
            text=raw_text, used_fallback=True, error_detail=e.detail
        )