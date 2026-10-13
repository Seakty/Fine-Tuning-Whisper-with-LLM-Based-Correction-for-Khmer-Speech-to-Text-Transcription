"""LLM correction layer (FR-4): send raw Whisper output to Gemini, get corrected Khmer text.

This module ONLY talks to Gemini. It does not decide what happens if Gemini is unreachable
or blocks a response - that's src/correction/fallback.py (BE-04), which should catch
GeminiCorrectionError and fall back to the raw ASR text.

Correction-only system prompt: fixes spacing/segmentation and likely misrecognitions from
context, without changing meaning, translating, or adding commentary. Directly targets the
baseline error patterns from docs/results.md - word-level hallucination, mishandled
code-switched terms, and Khmer's inconsistent word-spacing.
"""
import os
import time
from typing import List, Optional

from ..audio.validate import AudioError

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SEC = 10
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 1.5

# Gemini 3.x removed temperature/top_p/top_k entirely - "thinking_level" is the only
# generation control now (low/medium/high). "low" fits a correction task: fast, minimal
# reasoning. Note this is NOT equivalent to the old temperature=0.0 determinism guarantee -
# there is no determinism knob in the new API.
THINKING_LEVEL = "low"

# If Gemini's output length is wildly different from the input, something went wrong
# (refusal, added commentary, truncation) - safer to reject and let the fallback path
# return the raw ASR text than to return a corrupted "correction".
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.5

CORRECTION_SYSTEM_PROMPT = """You are a correction-only post-processor for Khmer speech-to-text output.

You will receive raw, uncorrected Khmer text produced by an automatic speech recognition
(ASR) model. The ASR model sometimes:
- inserts incorrect word spacing (too many or too few breaks between syllables/words)
- substitutes a plausible-sounding but wrong Khmer word for what was actually said
- garbles English/foreign loanwords spoken mid-sentence (names, technical terms, brands)

Your job is ONLY to correct these ASR errors. You must:
1. Fix word spacing/segmentation to standard, natural Khmer.
2. Fix words that are very likely ASR misrecognitions, using surrounding context to infer
   the intended word.
3. Recognize garbled foreign loanwords and restore them to their standard form (e.g. the
   correct spelling of an English name or technical term embedded in the Khmer sentence).

You must NOT:
- Change the meaning, add information, remove information, or summarize.
- Translate any part of the text into another language.
- Fix grammar/style choices that are not ASR errors (leave the speaker's phrasing alone).
- Add any commentary, explanation, labels, quotation marks, or preamble of any kind.

Output ONLY the corrected Khmer text, and nothing else. If the input is empty or you
cannot confidently improve it, return it unchanged."""


class GeminiCorrectionError(AudioError):
    """Gemini call failed, was blocked, or returned an unusable response.
    `.user_message` is intentionally generic; the fallback path decides what to show."""


class GeminiConfigError(GeminiCorrectionError):
    """Missing/invalid API key or bad configuration - fails fast, before any API call."""


def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiConfigError(
            "Text correction is not configured.",
            detail="GEMINI_API_KEY not set (see .env.example)",
        )
    return key


def _clean_response(text: str) -> str:
    """Strip common preamble/wrappers the model adds despite instructions not to.
    Prefixes and quote-wrapping can be stacked (e.g. `Prefix: "text"`), so keep
    stripping layers until nothing more changes."""
    text = text.strip()
    prefixes = ("Corrected text:", "Corrected Khmer text:", "Here is the corrected text:",
               "Here's the corrected text:", "កំណែពិន្ត្យ:")
    while True:
        before = text
        for wrapper in ('"""', "```", '"', "'"):
            if text.startswith(wrapper) and text.endswith(wrapper) and len(text) > 2 * len(wrapper):
                text = text[len(wrapper):-len(wrapper)].strip()
        for p in prefixes:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()
        if text == before:
            return text


SHORT_TEXT_CHARS = 15       # below this, a length ratio is too noisy to be meaningful
SHORT_TEXT_MAX_ABS_DIFF = 20  # so judge short text by absolute character difference instead


def _is_reasonable_correction(raw: str, corrected: str) -> bool:
    if not corrected:
        return False
    if len(raw) < SHORT_TEXT_CHARS:
        return abs(len(corrected) - len(raw)) <= SHORT_TEXT_MAX_ABS_DIFF
    ratio = len(corrected) / len(raw)
    return MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO


def correct_text(
    raw_text: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Send `raw_text` to Gemini for correction-only post-processing.

    Returns the corrected text. Raises GeminiCorrectionError (never returns garbage
    silently) - callers that want graceful degradation should catch it (see fallback.py).
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""   # nothing to correct, and don't spend an API call on it

    key = _get_api_key(api_key)

    from google import genai   # lazy: keep import cheap when correction isn't used
    from google.genai import types

    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    except TypeError:
        # http_options signature can vary across google-genai releases - degrade to no
        # explicit timeout rather than hard-failing on a version mismatch.
        client = genai.Client(api_key=key)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            interaction = client.interactions.create(
                model=model_name,
                system_instruction=CORRECTION_SYSTEM_PROMPT,
                input=raw_text,
                generation_config={"thinking_level": THINKING_LEVEL},
            )
        except Exception as e:
            # 400/404 (e.g. bad/retired model name) won't fix itself on retry.
            status = getattr(e, "code", None)
            if status in (400, 404) or attempt >= max_retries:
                raise GeminiCorrectionError(
                    "Text correction is temporarily unavailable.",
                    detail=f"Gemini call failed after {attempt + 1} attempt(s): {e!r}",
                )
            last_error = e
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

        text = getattr(interaction, "output_text", None)
        if not text:
            last_error = RuntimeError("interaction had no output_text (possibly blocked/empty)")
            if attempt < max_retries:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            raise GeminiCorrectionError(
                "Text correction is temporarily unavailable.",
                detail=f"Gemini returned no usable text: {last_error!r}",
            )

        cleaned = _clean_response(text)
        if not _is_reasonable_correction(raw_text, cleaned):
            raise GeminiCorrectionError(
                "Text correction is temporarily unavailable.",
                detail=f"rejected implausible correction: raw_len={len(raw_text)} "
                       f"corrected_len={len(cleaned)}",
            )
        return cleaned

    raise GeminiCorrectionError(   # unreachable, keeps type-checkers happy
        "Text correction is temporarily unavailable.", detail=repr(last_error))


def correct_batch(texts: List[str], delay_sec: float = 0.0, **kwargs) -> List[str]:
    """Sequential helper for eval scripts (EVAL-05). One bad item raises immediately -
    catch per-item in the caller if partial results with skips are preferred instead."""
    out = []
    for i, t in enumerate(texts):
        out.append(correct_text(t, **kwargs))
        if delay_sec and i < len(texts) - 1:
            time.sleep(delay_sec)
    return out