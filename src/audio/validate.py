"""Audio validation (FR-7): reject unsupported or oversized files with a clear message.

Two layers, so the bot can fail fast:
  1. validate_metadata()   - checks filename + size *before* downloading (Telegram tells
                             you both up front via message.voice / message.audio).
  2. validate_audio_file() - checks the file on disk after download.

Whether the bytes are actually decodable audio is decided by ffmpeg in convert.py; a
corrupt file surfaces there as AudioConversionError.
"""
from pathlib import Path
from typing import Optional

# Telegram voice notes are Opus-in-Ogg (.ogg). Forwarded/uploaded audio can be the rest.
ALLOWED_EXTENSIONS = {".ogg", ".oga", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac"}

# Telegram's cloud Bot API only lets bots download files up to 20 MB.
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_DURATION_SEC = 300  # 5 min; lower this if your infer.py doesn't chunk long audio


class AudioError(Exception):
    """Base class. `user_message` is safe to send straight back to the Telegram user;
    `detail` is for logs only."""

    def __init__(self, user_message: str, detail: Optional[str] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class AudioValidationError(AudioError):
    """File type / size / emptiness problem."""


def validate_metadata(
    filename: Optional[str],
    size_bytes: Optional[int],
    duration_sec: Optional[float] = None,
) -> None:
    """Validate what Telegram tells us before downloading. Raises AudioValidationError."""
    # Voice notes usually arrive with no filename at all; treat that as .ogg.
    ext = Path(filename).suffix.lower() if filename else ".ogg"

    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS))
        raise AudioValidationError(
            f"Sorry, I can't read '{ext or 'this'}' files. "
            f"Please send a voice message or one of: {allowed}.",
            detail=f"unsupported extension {ext!r}",
        )

    if size_bytes is not None:
        if size_bytes <= 0:
            raise AudioValidationError(
                "That file looks empty. Please try sending it again.",
                detail="size_bytes <= 0",
            )
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise AudioValidationError(
                f"That file is too large ({size_bytes / 1024 / 1024:.1f} MB). "
                f"The limit is {MAX_FILE_SIZE_MB} MB.",
                detail=f"size {size_bytes} > {MAX_FILE_SIZE_BYTES}",
            )

    if duration_sec is not None and duration_sec > MAX_DURATION_SEC:
        raise AudioValidationError(
            f"That audio is too long ({duration_sec:.1f} sec). "
            f"The limit is {MAX_DURATION_SEC} sec.",
            detail=f"duration {duration_sec} > {MAX_DURATION_SEC}",
        )


def validate_audio_file(path) -> Path:
    """Validate a downloaded file on disk. Returns the Path if OK, raises otherwise."""
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(
            "I couldn't find the audio file. Please try sending it again.",
            detail=f"not a file: {path}",
        )
    validate_metadata(path.name, path.stat().st_size)
    return path