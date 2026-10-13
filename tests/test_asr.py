"""Tests for src/asr/infer.py and src/asr/model.py (FR-3).

Most tests are fast: they inject a fake model or patch helpers, so no GPU or model
download is needed. Only the @slow tests load the real Whisper + LoRA adapter.
"""
import glob
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.asr import infer
from src.asr.infer import SAMPLE_RATE, max_new_tokens_for, split_audio
from src.asr.model import ASRError, check_adapter_trained, load_asr_model


def _noise(seconds, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, int(seconds * SAMPLE_RATE)).astype(np.float32)


# ------------------------------------------------------------------- split_audio
def test_short_audio_is_single_chunk():
    audio = _noise(10)
    chunks = split_audio(audio)
    assert len(chunks) == 1 and np.array_equal(chunks[0], audio)


def test_exactly_max_length_is_single_chunk():
    assert len(split_audio(_noise(28))) == 1


def test_long_audio_chunks_within_limit_and_lossless():
    audio = _noise(70)
    chunks = split_audio(audio)
    assert len(chunks) == 3
    assert all(len(c) <= int(infer.MAX_CHUNK_SEC * SAMPLE_RATE) for c in chunks)
    assert np.array_equal(np.concatenate(chunks), audio)  # nothing dropped or duplicated


def test_cuts_at_the_quiet_gap():
    audio = _noise(70)
    audio[24 * SAMPLE_RATE: 25 * SAMPLE_RATE] = 0.0   # 1 s pause inside the 22-28 s search window
    chunks = split_audio(audio)
    assert 24.0 <= len(chunks[0]) / SAMPLE_RATE <= 25.0


def test_tiny_tail_is_merged_into_previous_chunk():
    audio = _noise(28.5)
    audio[444800:448000] = 0.0   # pause at 27.8-28.0 s -> cut at ~27.81 s leaves a 0.69 s tail
    chunks = split_audio(audio)
    assert len(chunks) == 1 and len(chunks[0]) == len(audio)


# ------------------------------------------------------------- max_new_tokens_for
def test_token_budget_scales_with_duration():
    assert max_new_tokens_for(1, 3) == 120          # 1*40 + 80
    assert max_new_tokens_for(0, 3) == 80


def test_token_budget_capped_by_decoder_limit():
    assert max_new_tokens_for(100, 3) == infer.MAX_TARGET_TOKENS - 3 - 10


# --------------------------------------------------------- transcribe (fast, patched)
def test_silence_returns_empty_without_running_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model must not run on silence")
    monkeypatch.setattr(infer, "_transcribe_chunk", boom)

    for level in (0.0, 5e-4):   # digital silence and near-silence, both under SILENCE_PEAK
        audio = np.full(SAMPLE_RATE * 2, level, dtype=np.float32)
        monkeypatch.setattr(infer, "_load_audio", lambda p, a=audio: a)
        assert infer.transcribe("x.wav", asr=object()) == ""


def test_empty_audio_raises_asr_error(monkeypatch):
    monkeypatch.setattr(infer, "_load_audio", lambda p: np.array([], dtype=np.float32))
    with pytest.raises(ASRError) as exc:
        infer.transcribe("x.wav", asr=object())
    assert exc.value.user_message


def test_long_audio_is_chunked_and_joined(monkeypatch):
    monkeypatch.setattr(infer, "_load_audio", lambda p: _noise(70))
    counter = iter(range(1, 10))
    monkeypatch.setattr(infer, "_transcribe_chunk", lambda c, a: f"part{next(counter)}")
    assert infer.transcribe("x.wav", asr=object()) == "part1 part2 part3"


def test_empty_chunk_results_are_skipped(monkeypatch):
    monkeypatch.setattr(infer, "_load_audio", lambda p: _noise(70))
    results = iter(["a", "", "b"])
    monkeypatch.setattr(infer, "_transcribe_chunk", lambda c, a: next(results))
    assert infer.transcribe("x.wav", asr=object()) == "a b"   # no double space


# --------------------------------------------------------------- _load_audio
def test_load_audio_reads_16k_wav(tmp_path, make_wav):
    pytest.importorskip("librosa")
    p = make_wav(tmp_path / "a.wav", seconds=1, rate=16000, channels=1)
    assert len(infer._load_audio(p)) == 16000


def test_load_audio_missing_file_raises_asr_error(tmp_path):
    pytest.importorskip("librosa")
    with pytest.raises(ASRError) as exc:
        infer._load_audio(tmp_path / "nope.wav")
    assert exc.value.user_message


# ----------------------------------------------------------- _transcribe_chunk
def _fake_asr(generate):
    asr = MagicMock()
    asr.fp16 = False
    asr.device = "cpu"
    asr.prompt_length = 3
    asr.lock = threading.Lock()
    asr.model.generate = generate
    asr.processor.batch_decode.return_value = ["  ខ្ញុំ  "]
    return asr


def test_transcribe_chunk_returns_stripped_text():
    pytest.importorskip("torch")
    asr = _fake_asr(MagicMock(return_value="ids"))
    assert infer._transcribe_chunk(np.zeros(SAMPLE_RATE, np.float32), asr) == "ខ្ញុំ"


def test_generate_failure_becomes_asr_error():
    pytest.importorskip("torch")
    asr = _fake_asr(MagicMock(side_effect=RuntimeError("CUDA out of memory")))
    with pytest.raises(ASRError) as exc:
        infer._transcribe_chunk(np.zeros(SAMPLE_RATE, np.float32), asr)
    assert exc.value.user_message
    assert "out of memory" in exc.value.detail


# ---------------------------------------------------------- check_adapter_trained
class _FakePeft:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return self._params


def test_adapter_guard_accepts_trained_weights():
    torch = pytest.importorskip("torch")
    model = _FakePeft([("layer.lora_B.default.weight", torch.ones(4, 4))])
    assert check_adapter_trained(model) == pytest.approx(4.0)


def test_adapter_guard_rejects_untrained_zero_weights():
    torch = pytest.importorskip("torch")
    model = _FakePeft([("layer.lora_B.default.weight", torch.zeros(4, 4))])
    with pytest.raises(ASRError, match="") as exc:
        check_adapter_trained(model)
    assert "UNTRAINED" in exc.value.detail


def test_adapter_guard_rejects_missing_lora_layers():
    torch = pytest.importorskip("torch")
    model = _FakePeft([("layer.weight", torch.ones(2, 2))])
    with pytest.raises(ASRError) as exc:
        check_adapter_trained(model)
    assert "no lora_B" in exc.value.detail


def test_missing_adapter_dir_raises(tmp_path):
    for mod in ("torch", "peft", "transformers"):
        pytest.importorskip(mod)
    with pytest.raises(ASRError) as exc:
        load_asr_model(adapter_dir=str(tmp_path / "nope"))
    assert "adapter files not found" in exc.value.detail


# --------------------------------------------------- slow (real model, needs GPU)
@pytest.mark.slow
def test_real_model_loads_with_trained_adapter():
    from src.asr.model import get_asr
    asr = get_asr()   # runs check_adapter_trained internally; raises if untrained
    assert asr.prompt_length > 0


@pytest.mark.slow
def test_transcribes_real_eval_clip():
    files = sorted(glob.glob("data/eval/**/*.wav", recursive=True))
    if not files:
        pytest.skip("no .wav files under data/eval/")
    out = infer.transcribe(files[0])
    assert isinstance(out, str) and out.strip(), f"empty transcript for {files[0]}"
    assert any("\u1780" <= ch <= "\u17ff" for ch in out), f"no Khmer characters: {out!r}"