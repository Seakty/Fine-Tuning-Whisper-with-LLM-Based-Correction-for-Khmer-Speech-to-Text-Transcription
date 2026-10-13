"""Tests for src/audio/validate.py and src/audio/convert.py (FR-2, FR-7)."""
import inspect
import os
import shutil
import subprocess
import wave

import pytest

from src.audio import convert
from src.audio.convert import (
    TARGET_SAMPLE_RATE,
    AudioConversionError,
    convert_to_wav,
    prepare_audio,
    wav_duration_seconds,
)
from src.audio.validate import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    AudioValidationError,
    validate_audio_file,
    validate_metadata,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


# ------------------------------------------------------------------ validate_metadata
@pytest.mark.parametrize("ext", sorted(ALLOWED_EXTENSIONS))
def test_allowed_extensions_pass(ext):
    validate_metadata(f"clip{ext}", 1000)


def test_extension_check_is_case_insensitive():
    validate_metadata("CLIP.MP3", 1000)


@pytest.mark.parametrize("name", ["song.wma", "doc.pdf", "notes.txt", "video.mp4"])
def test_unsupported_extension_rejected(name):
    with pytest.raises(AudioValidationError) as exc:
        validate_metadata(name, 1000)
    assert exc.value.user_message
    assert "unsupported" in exc.value.detail


def test_no_filename_treated_as_ogg():
    validate_metadata(None, 1000)


def test_empty_file_rejected():
    with pytest.raises(AudioValidationError, match="empty"):
        validate_metadata("clip.ogg", 0)


def test_oversized_file_rejected():
    with pytest.raises(AudioValidationError, match="too large"):
        validate_metadata("clip.ogg", MAX_FILE_SIZE_BYTES + 1)


def test_file_at_exact_limit_passes():
    validate_metadata("clip.ogg", MAX_FILE_SIZE_BYTES)


def test_unknown_size_is_allowed():
    validate_metadata("clip.ogg", None)


@pytest.mark.skipif(
    "duration_sec" not in inspect.signature(validate_metadata).parameters,
    reason="duration check from Card 24 not applied yet",
)
def test_too_long_clip_rejected():
    with pytest.raises(AudioValidationError, match="too long"):
        validate_metadata("clip.ogg", 1000, 3600)


def test_error_carries_user_message_and_detail():
    with pytest.raises(AudioValidationError) as exc:
        validate_metadata("x.exe", 10)
    assert isinstance(exc.value.user_message, str) and exc.value.user_message
    assert isinstance(exc.value.detail, str) and exc.value.detail


# ---------------------------------------------------------------- validate_audio_file
def test_validate_audio_file_ok(tmp_path, make_wav):
    p = make_wav(tmp_path / "ok.wav", seconds=0.2)
    assert validate_audio_file(p) == p


def test_validate_audio_file_missing(tmp_path):
    with pytest.raises(AudioValidationError):
        validate_audio_file(tmp_path / "nope.wav")


def test_file_without_extension_rejected(tmp_path):
    """Regression test for the "can't read 'this' files" bug: Telegram downloads have
    no extension, so handlers.py must add one before validation."""
    p = tmp_path / "input_AgADabc"
    p.write_bytes(b"data")
    with pytest.raises(AudioValidationError):
        validate_audio_file(p)


def test_empty_file_on_disk_rejected(tmp_path):
    p = tmp_path / "empty.ogg"
    p.write_bytes(b"")
    with pytest.raises(AudioValidationError, match="empty"):
        validate_audio_file(p)


# ---------------------------------------------------------------------------- convert
def test_wav_duration_seconds(tmp_path, make_wav):
    p = make_wav(tmp_path / "a.wav", seconds=1.5, rate=16000, channels=1)
    assert wav_duration_seconds(p) == pytest.approx(1.5, abs=0.01)


@needs_ffmpeg
def test_convert_wav_to_16k_mono(tmp_path, make_wav):
    src = make_wav(tmp_path / "in.wav", seconds=1.0, rate=44100, channels=2)
    out = convert_to_wav(src, tmp_path / "out.wav")
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == TARGET_SAMPLE_RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # 16-bit PCM
    assert wav_duration_seconds(out) == pytest.approx(1.0, abs=0.05)


@needs_ffmpeg
def test_convert_does_not_overwrite_wav_input(tmp_path, make_wav):
    src = make_wav(tmp_path / "in.wav", seconds=0.5, rate=44100, channels=2)
    out = convert_to_wav(src)  # output_path=None
    assert out != src
    assert out.name.endswith("_16k.wav")
    with wave.open(str(src), "rb") as w:  # original untouched
        assert w.getframerate() == 44100


@needs_ffmpeg
def test_prepare_audio_wav_input_same_output_dir(tmp_path, make_wav):
    """Fails until the same-path fix in prepare_audio() is applied (input_X.wav with
    output_dir == its own folder must not use the input as the output)."""
    src = make_wav(tmp_path / "input_X.wav", seconds=0.5, rate=44100, channels=2)
    out = prepare_audio(src, str(tmp_path))
    assert out != src
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == TARGET_SAMPLE_RATE


@needs_ffmpeg
def test_prepare_audio_ogg_opus(tmp_path):
    """Telegram voice notes are Opus-in-Ogg; build one with ffmpeg and convert it."""
    ogg = tmp_path / "voice.ogg"
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "libopus", str(ogg)],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip("this ffmpeg build has no libopus encoder")
    out = prepare_audio(ogg, str(tmp_path / "out"))
    with wave.open(str(out), "rb") as w:
        assert (w.getframerate(), w.getnchannels()) == (TARGET_SAMPLE_RATE, 1)


@needs_ffmpeg
def test_corrupt_audio_raises_conversion_error(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(os.urandom(2000))
    with pytest.raises(AudioConversionError) as exc:
        prepare_audio(bad, str(tmp_path))
    assert exc.value.user_message
    assert not (tmp_path / "bad.wav").exists()  # partial output cleaned up


@needs_ffmpeg
def test_zero_length_wav_raises_conversion_error(tmp_path):
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
    with pytest.raises(AudioConversionError):
        prepare_audio(empty, str(tmp_path / "out"))


def test_missing_ffmpeg_raises_friendly_error(tmp_path, make_wav, monkeypatch):
    src = make_wav(tmp_path / "in.wav", seconds=0.1)
    monkeypatch.setattr(convert.shutil, "which", lambda _name: None)
    with pytest.raises(AudioConversionError) as exc:
        convert_to_wav(src, tmp_path / "out.wav")
    assert "ffmpeg" in exc.value.detail


def test_ffmpeg_timeout_raises_conversion_error(tmp_path, make_wav, monkeypatch):
    src = make_wav(tmp_path / "in.wav", seconds=0.1)
    monkeypatch.setattr(convert.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(convert.subprocess, "run", fake_run)
    with pytest.raises(AudioConversionError, match="too long"):
        convert_to_wav(src, tmp_path / "out.wav", timeout=1)