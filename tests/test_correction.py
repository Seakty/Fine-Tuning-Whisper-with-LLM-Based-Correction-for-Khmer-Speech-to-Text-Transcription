"""Tests for src/correction (FR-4, FR-5).

Fast tests mock the Gemini SDK. The @live test calls the real API.
"""
import os
import sys
import types
from types import SimpleNamespace

import pytest

from src.correction import fallback, gemini_client
from src.correction.fallback import correct_with_fallback
from src.correction.gemini_client import (
    CORRECTION_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    GeminiConfigError,
    GeminiCorrectionError,
    _clean_response,
    _is_reasonable_correction,
    correct_text,
)

KHMER_SAMPLE = "សួស្តី បង តើ អ្នក សុខសប្បាយ ទេ"


# ---------------------------------------------------------------- fake Gemini SDK
@pytest.fixture
def fake_genai(monkeypatch):
    """Replace google.genai with a fake. Set `.responses` to a list of strings (returned as
    output_text) or Exceptions (raised); the last item repeats. `.calls` records requests."""
    state = SimpleNamespace(calls=[], responses=[])

    def create(**kwargs):
        state.calls.append(kwargs)
        item = state.responses[min(len(state.calls), len(state.responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(output_text=item)

    class FakeClient:
        def __init__(self, **kwargs):
            self.interactions = SimpleNamespace(create=create)

    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")
    gtypes.HttpOptions = lambda **kw: kw
    genai.Client = FakeClient
    genai.types = gtypes
    google = types.ModuleType("google")
    google.__path__ = []
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: None)   # instant retries
    return state


# ------------------------------------------------------------------ _clean_response
@pytest.mark.parametrize("raw, expected", [
    ("plain text", "plain text"),
    ('"quoted"', "quoted"),
    ("```fenced```", "fenced"),
    ("Corrected text: hello", "hello"),
    ('Corrected text: "hello"', "hello"),      # stacked prefix + quotes
    ("  spaced  ", "spaced"),
])
def test_clean_response(raw, expected):
    assert _clean_response(raw) == expected


# ------------------------------------------------------ _is_reasonable_correction
@pytest.mark.parametrize("raw_len, out_len, ok", [
    (100, 100, True),
    (100, 40, True),     # boundary: ratio 0.4 allowed
    (100, 30, False),    # too short (likely truncated or refused)
    (100, 250, True),    # boundary: ratio 2.5 allowed
    (100, 300, False),   # too long (likely added commentary)
    (5, 20, True),       # short input: judged by absolute diff (15 <= 20)
    (5, 30, False),      # short input: diff 25 > 20
    (100, 0, False),     # empty output
])
def test_is_reasonable_correction(raw_len, out_len, ok):
    assert _is_reasonable_correction("a" * raw_len, "b" * out_len) is ok


# ------------------------------------------------------------ correct_text (mocked)
def test_empty_input_returns_empty_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert correct_text("   ") == ""


def test_missing_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiConfigError) as exc:
        correct_text(KHMER_SAMPLE)
    assert isinstance(exc.value, GeminiCorrectionError)   # so the fallback catches it
    assert "GEMINI_API_KEY" in exc.value.detail


def test_success_cleans_output_and_sends_correct_request(fake_genai):
    fake_genai.responses = [f'"{KHMER_SAMPLE}"']
    out = correct_text(KHMER_SAMPLE, api_key="test-key")
    assert out == KHMER_SAMPLE
    assert len(fake_genai.calls) == 1
    call = fake_genai.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["system_instruction"] == CORRECTION_SYSTEM_PROMPT
    assert call["input"] == KHMER_SAMPLE
    assert call["generation_config"] == {"thinking_level": "low"}


def test_transient_errors_are_retried(fake_genai):
    fake_genai.responses = [Exception("503"), Exception("503"), KHMER_SAMPLE]
    assert correct_text(KHMER_SAMPLE, api_key="k") == KHMER_SAMPLE
    assert len(fake_genai.calls) == 3


def test_gives_up_after_max_retries(fake_genai):
    fake_genai.responses = [Exception("503")]
    with pytest.raises(GeminiCorrectionError) as exc:
        correct_text(KHMER_SAMPLE, api_key="k")
    assert len(fake_genai.calls) == 3   # 1 try + 2 retries
    assert "3 attempt" in exc.value.detail


def test_bad_request_is_not_retried(fake_genai):
    err = Exception("model not found")
    err.code = 404
    fake_genai.responses = [err]
    with pytest.raises(GeminiCorrectionError):
        correct_text(KHMER_SAMPLE, api_key="k")
    assert len(fake_genai.calls) == 1


def test_empty_response_raises_after_retries(fake_genai):
    fake_genai.responses = [""]
    with pytest.raises(GeminiCorrectionError) as exc:
        correct_text(KHMER_SAMPLE, api_key="k")
    assert "no usable text" in exc.value.detail
    assert len(fake_genai.calls) == 3


def test_implausible_correction_is_rejected(fake_genai):
    fake_genai.responses = ["x"]   # far shorter than the input
    with pytest.raises(GeminiCorrectionError) as exc:
        correct_text(KHMER_SAMPLE, api_key="k")
    assert "implausible" in exc.value.detail


# ------------------------------------------------------- correct_with_fallback (FR-5)
def test_fallback_on_gemini_error(monkeypatch):
    def fail(text, **kw):
        raise GeminiCorrectionError("generic", detail="rate limited")
    monkeypatch.setattr(fallback, "correct_text", fail)
    result = correct_with_fallback(f"  {KHMER_SAMPLE}  ")
    assert result.used_fallback is True
    assert result.text == KHMER_SAMPLE          # raw text, whitespace stripped
    assert result.error_detail == "rate limited"


def test_success_path_not_marked_as_fallback(monkeypatch):
    monkeypatch.setattr(fallback, "correct_text", lambda text, **kw: "corrected")
    result = correct_with_fallback("raw")
    assert (result.text, result.used_fallback, result.error_detail) == ("corrected", False, None)


def test_kwargs_pass_through_to_correct_text(monkeypatch):
    seen = {}
    def spy(text, **kw):
        seen.update(kw)
        return text
    monkeypatch.setattr(fallback, "correct_text", spy)
    correct_with_fallback("raw", model_name="m", timeout=3)
    assert seen == {"model_name": "m", "timeout": 3}


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_skips_gemini(monkeypatch, text):
    def boom(*a, **k):
        raise AssertionError("must not call Gemini for empty input")
    monkeypatch.setattr(fallback, "correct_text", boom)
    result = correct_with_fallback(text)
    assert result.text == "" and result.used_fallback is False


def test_no_api_key_end_to_end_falls_back(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = correct_with_fallback(KHMER_SAMPLE)
    assert result.used_fallback is True
    assert result.text == KHMER_SAMPLE
    assert "GEMINI_API_KEY" in result.error_detail


def test_unexpected_exceptions_still_propagate(monkeypatch):
    """Documented behavior: only GeminiCorrectionError is swallowed. handlers.py wraps the
    call in its own try/except (Card 24) so a bug here can't leave the chat stuck."""
    def bug(text, **kw):
        raise ValueError("real bug")
    monkeypatch.setattr(fallback, "correct_text", bug)
    with pytest.raises(ValueError):
        correct_with_fallback("raw")


# -------------------------------------------------------------- live (real API)
@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_live_gemini_correction():
    result = correct_with_fallback(KHMER_SAMPLE)
    assert not result.used_fallback, f"Gemini failed: {result.error_detail}"
    assert result.text.strip()
    print("\nraw      :", KHMER_SAMPLE)
    print("corrected:", result.text)