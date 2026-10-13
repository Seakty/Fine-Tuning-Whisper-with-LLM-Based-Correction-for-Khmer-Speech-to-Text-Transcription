"""Audio conversion (FR-2): any supported input -> 16 kHz mono 16-bit PCM .wav.

Uses the ffmpeg binary (must be installed on the host / in the Docker image), because
Telegram voice notes are Opus-in-Ogg, which librosa/soundfile can't reliably decode.

Quick manual test with a real voice note:
    python -m src.audio path/to/voice.ogg
"""
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

from .validate import AudioError, validate_audio_file

TARGET_SAMPLE_RATE = 16000  # Whisper's required input rate
CONVERT_TIMEOUT_SEC = 60


class AudioConversionError(AudioError):
    """ffmpeg missing, failed, timed out, or produced an empty result."""


def convert_to_wav(input_path, output_path=None, timeout: int = CONVERT_TIMEOUT_SEC) -> Path:
    """Convert `input_path` to 16 kHz mono .wav and return the output Path."""
    input_path = Path(input_path)

    if shutil.which("ffmpeg") is None:
        raise AudioConversionError(
            "Audio processing is temporarily unavailable. Please try again later.",
            detail="ffmpeg binary not found on PATH",
        )

    if output_path is None:
        output_path = input_path.with_suffix(".wav")
        if output_path == input_path:  # input already a .wav; don't overwrite it
            output_path = input_path.with_name(input_path.stem + "_16k.wav")
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(input_path),
        "-vn",                          # drop any video/cover-art stream
        "-ac", "1",                     # mono
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        raise AudioConversionError(
            "That audio took too long to process. Please try a shorter clip.",
            detail=f"ffmpeg timed out after {timeout}s",
        )

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise AudioConversionError(
            "I couldn't read that audio. The file may be corrupted - please try again.",
            detail=result.stderr.strip()[-500:],
        )

    # ffmpeg can "succeed" on a container with no samples -> zero-length wav.
    try:
        empty = not output_path.exists() or wav_duration_seconds(output_path) <= 0
    except (wave.Error, EOFError):
        empty = True
    if empty:
        output_path.unlink(missing_ok=True)
        raise AudioConversionError(
            "That file doesn't seem to contain any audio.",
            detail="output wav has zero duration",
        )

    return output_path


def wav_duration_seconds(wav_path) -> float:
    """Duration of a PCM .wav in seconds (stdlib only)."""
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def prepare_audio(input_path, output_dir: Optional[str] = None) -> Path:
    """Single entrypoint for the bot: validate, then convert. Returns the .wav Path.

    Raises AudioValidationError / AudioConversionError; both carry `.user_message`.
    """
    input_path = validate_audio_file(input_path)
    output_path = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / (input_path.stem + ".wav")
        if output_path == input_path:
            output_path = out_dir / (input_path.stem + "_16k.wav")
    return convert_to_wav(input_path, output_path)