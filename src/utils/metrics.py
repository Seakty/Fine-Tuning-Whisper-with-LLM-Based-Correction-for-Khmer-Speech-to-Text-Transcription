"""
CER/WER scoring utilities for the Khmer STT project.

CER (Character Error Rate) is the primary metric for this project — Khmer script
has no spaces between words, so word-level boundaries used by WER are ambiguous.
WER is still computed and reported for comparability with prior published Khmer
ASR work (e.g. whisper-small-khmer-v2's reported WER).
"""

from jiwer import cer, wer


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate between a single reference (ground truth) and
    hypothesis (model prediction) string. Returns a float, e.g. 0.42 = 42% error.
    """
    return cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate between a single reference and hypothesis string.
    Returns a float, e.g. 0.62 = 62% error.
    """
    return wer(reference, hypothesis)


def compute_batch_cer(references: list[str], hypotheses: list[str]) -> float:
    """
    CER computed across a full list of reference/hypothesis pairs at once.
    This is the aggregate/overall CER — not an average of per-file CERs, since
    jiwer pools all edit distances and lengths together for the true overall rate.
    """
    return cer(references, hypotheses)


def compute_batch_wer(references: list[str], hypotheses: list[str]) -> float:
    """
    WER computed across a full list of reference/hypothesis pairs at once.
    """
    return wer(references, hypotheses)