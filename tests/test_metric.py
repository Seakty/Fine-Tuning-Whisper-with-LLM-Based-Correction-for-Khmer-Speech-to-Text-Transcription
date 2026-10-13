"""
Quick sanity check for src/utils/metrics.py — confirms CER/WER behave as
expected on a known input/output pair before we trust them on real data.
"""

import sys
sys.path.insert(0, "src")  # adjust if running from project root differently

from utils.metrics import compute_cer, compute_wer


def test_identical_strings_score_zero():
    ref = "សួស្តី ពិភពលោក"
    hyp = "សួស្តី ពិភពលោក"
    assert compute_cer(ref, hyp) == 0.0
    assert compute_wer(ref, hyp) == 0.0


def test_completely_different_strings_score_nonzero():
    ref = "សួស្តី ពិភពលោក"
    hyp = "ជំរាបសួរ"
    assert compute_cer(ref, hyp) > 0.0
    assert compute_wer(ref, hyp) > 0.0


def test_known_error_case():
    # One word wrong out of two — WER should be 0.5 exactly
    ref = "សួស្តី ពិភពលោក"
    hyp = "សួស្តី ខ្ញុំ"
    assert compute_wer(ref, hyp) == 0.5


if __name__ == "__main__":
    test_identical_strings_score_zero()
    test_completely_different_strings_score_nonzero()
    test_known_error_case()
    print("All metrics tests passed.")